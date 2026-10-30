"""Cloud engine (extra: mineru): MinerU extraction service via signed
upload + batch poll, rate-limited by dual xtremeflow lanes (submit
50/min, status 1000/min — the official ceilings, per-minute numbers
divided by 60 into float rps). 429/5xx raise RetryException and retry
through auto_backoff; other 4xx (quota, auth) surface as hard errors.
Results are cached via the shared ``_cloud`` LRU — memory only,
never disk. full.md's HTML table blocks flatten to line text at the
boundary (``_markdown``)."""

from __future__ import annotations

import asyncio
import io
import os
import zipfile

import httpx

from docspectra.parser import (ImageAsset, MAX_ASSETS, MAX_ASSET_BYTES,
                               ParseResult, provider)
from docspectra.providers import traced
from docspectra.providers._cloud import (ResultCache, check_http, fetch,
                                         mime_for_ext, poll_until, run_job)
from docspectra.providers._markdown import flatten_tables

from xtremeflow.scheduler.rate_limit import RetryException, auto_backoff
from xtremeflow.scheduler.request import RequestRateScheduler  # deep import: not re-exported

BASE = 'https://mineru.net/api/v4'
_UPLOAD_RETRIES = 3
_AsyncClient = httpx.AsyncClient  # provider-local seam: tests patch this name


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

    async def flow(client, submit_lane, status_lane, timeout):
        batch_id, file_url = await (await submit_lane.start_task(
            _apply_url(client, token, model, name)))
        await _upload(client, file_url, data)  # outside the lanes
        zip_url = await _poll(client, token, batch_id, timeout, status_lane)
        if not zip_url:
            raise MineruError(
                f'mineru batch {batch_id} finished without a result zip')
        blob = await fetch(client, zip_url,
                           label='mineru result download',
                           error=MineruError, text=False)
        markdown, images, skipped = _unzip(blob)
        return ParseResult(
            text=flatten_tables(markdown), images=images,
            meta={'model_version': model, 'batch_id': batch_id,
                  'skipped_assets': skipped})

    return await run_job(
        cache=_cache, data=data, model=model, facet='model_version',
        timeout_env='MINERU_TIMEOUT',
        lanes_env='MINERU_CONCURRENCY', lanes_spec='submit:50|status:1000',
        client_factory=_AsyncClient, flow=flow)


@auto_backoff()
async def _apply_url(client: httpx.AsyncClient, token: str, model: str,
                     name: str | None) -> tuple[str, str]:
    res = await client.post(f'{BASE}/file-urls/batch', headers=_headers(token),
                            json={'files': [{'name': name or 'file', 'data_id': 'f'}],
                                  'model_version': model})
    check_http(res, label='mineru file-urls', error=MineruError, retry_5xx=True)
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
        try:
            check_http(res, label='mineru upload', error=MineruError,
                       retry_5xx=True)
            return
        except RetryException:
            if attempt + 1 == _UPLOAD_RETRIES:
                raise MineruError(
                    f'mineru upload failed after {_UPLOAD_RETRIES} '
                    f'attempts: {res.status_code}') from None
            await asyncio.sleep(2 ** attempt)


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
    check_http(res, label='mineru extract-results', error=MineruError,
               retry_5xx=True)
    payload = res.json()
    if payload.get('code') != 0:
        raise MineruError(f'mineru extract-results failed: {payload.get("msg")}')
    return payload['data']['extract_result'][0]


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


def _headers(token: str) -> dict:
    return {'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'}


_cache = ResultCache()
