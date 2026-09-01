"""Cloud engine (extra: mineru): MinerU extraction service via signed
upload + batch poll, rate-limited by dual xtremeflow lanes (submit
50/min, status 1000/min — the official ceilings, per-minute numbers
divided by 60 into float rps). 429/5xx raise RetryException and retry
through auto_backoff; other 4xx (quota, auth) surface as hard errors.
Results are cached via the shared ``_cloud`` LRU — memory only,
never disk."""

from __future__ import annotations

import asyncio
import io
import os
import zipfile

import httpx

from docspectra.parser import (ImageAsset, MAX_ASSETS, MAX_ASSET_BYTES,
                               ParseResult, provider)
from docspectra.providers import traced
from docspectra.providers._cloud import ResultCache, lanes, mime_for_ext, poll_until

from xtremeflow.scheduler.rate_limit import RetryException, auto_backoff
from xtremeflow.scheduler.request import RequestRateScheduler  # deep import: not re-exported

BASE = 'https://mineru.net/api/v4'
_UPLOAD_RETRIES = 3


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
    key = ResultCache.key(data, model)
    if hit := _cache.facade(key, model_version=model, cache_hit=True):
        return hit
    timeout = float(os.environ.get('MINERU_TIMEOUT', '300'))
    submit_lane, status_lane = lanes('MINERU_CONCURRENCY', 'submit:50|status:1000')
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
    _cache.put(key, result)
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
    async def once():
        item = await _poll_once(client, token, batch_id)
        return item['state'], item.get('err_msg'), item.get('full_zip_url')
    return await poll_until(status_lane, once, timeout=timeout,
                            label='mineru', error=MineruError)


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
                images.append(ImageAsset(name=name, mime=mime_for_ext(ext),
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


_cache = ResultCache()
