"""Cloud engine (extra: mineru): MinerU extraction service via signed
upload + batch poll, rate-limited by dual xtremeflow lanes (submit
50/min, status 1000/min — the official ceilings, per-minute numbers
divided by 60 into float rps). 429/5xx raise RetryException and retry
through auto_backoff; other 4xx (quota, auth) surface as hard errors.
Results are cached in-process keyed by (sha256, model_version) — memory
only, never disk, evicted by count and byte budget."""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import time
import zipfile
from collections import OrderedDict

import httpx

from docspectra.parser import (ImageAsset, MAX_ASSETS, MAX_ASSET_BYTES,
                               ParseResult, provider)
from docspectra.providers import traced

from xtremeflow.scheduler.rate_limit import RetryException, auto_backoff
from xtremeflow.scheduler.request import RequestRateScheduler  # deep import: not re-exported

BASE = 'https://mineru.net/api/v4'
POLL_INTERVAL = 2.0
_CACHE_MAX = 8  # ~1 MB per file — roughly 8 MB resident
_CACHE_BYTES = 8 * 1024 * 1024
_UPLOAD_RETRIES = 3
_MIME = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png',
         'gif': 'image/gif', 'webp': 'image/webp', 'svg': 'image/svg+xml',
         'bmp': 'image/bmp', 'tiff': 'image/tiff'}


class MineruError(Exception):
    """Hard error: quota, auth, timeout, or a failed extraction — falls
    through the chain like any provider exception."""


@provider('mineru', ('pdf', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx',
                     'png', 'jpg', 'jpeg', 'jp2', 'webp', 'gif', 'bmp', 'tiff'))
@traced
async def parse_mineru(data: bytes, *, name: str | None = None) -> ParseResult:
    token = os.environ.get('MINERU_API_TOKEN')
    if not token:
        raise MineruError('MINERU_API_TOKEN is not set')
    model = os.environ.get('MINERU_MODEL_VERSION', 'pipeline')
    key = (hashlib.sha256(data).hexdigest(), model)
    if cached := _cache_get(key):
        # per-call facade: share the immutable payload, never the cached
        # meta (a batch id for a batch this call never submitted)
        return ParseResult(text=cached.text, images=cached.images,
                           meta={'model_version': model, 'cache_hit': True})
    timeout = float(os.environ.get('MINERU_TIMEOUT', '300'))
    submit_lane, status_lane = _lanes()
    async with httpx.AsyncClient(timeout=30.0) as client:
        task = await submit_lane.start_task(_apply_url(client, token, model, name))
        batch_id, file_url = await task
        await _upload(client, file_url, data)
        zip_url = await _poll(client, token, batch_id, timeout, status_lane)
        if not zip_url:
            raise MineruError(f'mineru batch {batch_id} finished without a result zip')
        result_zip = await _download(client, zip_url)
        markdown, images, skipped = _unzip(result_zip)
    result = ParseResult(
        text=markdown, images=images,
        meta={'model_version': model, 'batch_id': batch_id, 'skipped_assets': skipped})
    _cache_put(key, result)
    return result


@auto_backoff()
async def _apply_url(client: httpx.AsyncClient, token: str, model: str,
                     name: str | None) -> tuple[str, str]:
    res = await client.post(f'{BASE}/file-urls/batch', headers=_headers(token),
                            json={'files': [{'name': name or 'file', 'data_id': 'f'}],
                                  'model_version': model})
    _check(res)
    payload = res.json()
    if payload.get('code') != 0:
        raise MineruError(f'mineru file-urls failed: {payload.get("msg")}')
    return payload['data']['batch_id'], payload['data']['file_urls'][0]


async def _upload(client: httpx.AsyncClient, url: str, data: bytes) -> None:
    """S3 single PUT is atomic — an interrupted PUT leaves no object, so
    retrying the same signed URL is safe. Outside the lanes: S3 does not
    count against mineru's submit rate."""
    for attempt in range(_UPLOAD_RETRIES):
        res = await client.put(url, content=data, timeout=120.0)
        if res.status_code == 429 or res.status_code >= 500:
            await asyncio.sleep(2 ** attempt)
            continue
        if res.status_code >= 400:
            raise MineruError(f'mineru upload failed: {res.status_code}: {res.text[:200]}')
        return
    raise MineruError(f'mineru upload failed after {_UPLOAD_RETRIES} attempts: {res.status_code}')


async def _poll(client: httpx.AsyncClient, token: str, batch_id: str,
                timeout: float, status_lane: RequestRateScheduler) -> str:
    deadline = time.monotonic() + timeout
    while True:
        task = await status_lane.start_task(_poll_once(client, token, batch_id))
        item = await task
        state = item['state']
        if state == 'done':
            return item.get('full_zip_url')
        if state == 'failed':
            raise MineruError(f'mineru extract failed: {item.get("err_msg")}')
        if time.monotonic() >= deadline:
            raise MineruError(f'mineru timed out after {timeout:.0f}s (state {state})')
        await asyncio.sleep(POLL_INTERVAL)


@auto_backoff()
async def _poll_once(client: httpx.AsyncClient, token: str, batch_id: str) -> dict:
    res = await client.get(f'{BASE}/extract-results/batch/{batch_id}',
                           headers=_headers(token))
    _check(res)
    payload = res.json()
    if payload.get('code') != 0:
        raise MineruError(f'mineru extract-results failed: {payload.get("msg")}')
    return payload['data']['extract_result'][0]


async def _download(client: httpx.AsyncClient, zip_url: str) -> bytes:
    res = await client.get(zip_url, timeout=120.0)
    if res.status_code >= 400:
        raise MineruError(f'mineru result download failed: {res.status_code}')
    return res.content


def _unzip(result_zip: bytes) -> tuple[str, list[ImageAsset], int]:
    """Result zip → markdown (full.md) + ImageAssets; assets reconcile
    with the markdown's ``![](images/...)`` references by name."""
    markdown = ''
    found = False
    images: list[ImageAsset] = []
    skipped = 0
    with zipfile.ZipFile(io.BytesIO(result_zip)) as z:
        for name in z.namelist():
            if name.endswith('full.md'):
                found = True
                markdown = z.read(name).decode('utf-8', errors='replace')
            elif name.startswith('images/') and not name.endswith('/'):
                if len(images) >= MAX_ASSETS or z.getinfo(name).file_size > MAX_ASSET_BYTES:
                    skipped += 1
                    continue
                ext = name.rsplit('.', 1)[-1].lower()
                images.append(ImageAsset(name=name, mime=_MIME.get(ext, 'application/octet-stream'),
                                         data=z.read(name), source=f'mineru {name}'))
    if not found:
        raise MineruError('mineru result zip has no full.md')
    return markdown, images, skipped


def _check(res: httpx.Response) -> None:
    if res.status_code >= 400:
        msg = f'mineru {res.status_code}: {res.text[:200]}'
        if res.status_code == 429 or res.status_code >= 500:
            retry_after = res.headers.get('Retry-After')
            raise RetryException(msg, retry_after=float(retry_after) if retry_after else None)
        raise MineruError(msg)


def _headers(token: str) -> dict:
    return {'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'}


_lanes_cache: tuple[RequestRateScheduler, RequestRateScheduler] | None = None


def _lanes() -> tuple[RequestRateScheduler, RequestRateScheduler]:
    global _lanes_cache
    if _lanes_cache is None:
        spec = dict(kv.split(':', 1) for kv in
                    os.environ.get('MINERU_CONCURRENCY', 'submit:50|status:1000').split('|'))
        missing = [k for k in ('submit', 'status') if k not in spec]
        if missing:
            raise ValueError(f'MINERU_CONCURRENCY is missing the {missing} lane')
        _lanes_cache = (
            RequestRateScheduler(max_rps=float(spec['submit']) / 60, max_concurrency=8),
            RequestRateScheduler(max_rps=float(spec['status']) / 60, max_concurrency=32),
        )
    return _lanes_cache


_cache: OrderedDict[tuple[str, str], ParseResult] = OrderedDict()
_cache_bytes = 0


def _result_size(result: ParseResult) -> int:
    return len(result.text) + sum(len(img.data) for img in result.images)


def _cache_get(key: tuple[str, str]) -> ParseResult | None:
    if cached := _cache.get(key):
        _cache.move_to_end(key)
    return cached


def _cache_put(key: tuple[str, str], result: ParseResult) -> None:
    global _cache_bytes
    _cache[key] = result
    _cache.move_to_end(key)
    _cache_bytes += _result_size(result)
    while len(_cache) > _CACHE_MAX or _cache_bytes > _CACHE_BYTES:
        _, old = _cache.popitem(last=False)
        _cache_bytes -= _result_size(old)
