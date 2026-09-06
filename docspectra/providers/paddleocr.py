"""Cloud engine (extra: paddleocr): the PaddleOCR official API over
direct httpx — multipart submit → poll → jsonl result. The API client
ships inside a distribution that drags a local-inference stack
(paddlex[ocr-core]: opencv & friends) the client never imports, so
the extra stays slim and calls the API itself. Rate-limited by dual
xtremeflow lanes (submit 60/min, status 600/min — the service
publishes no ceilings; conservative defaults, tunable via
PADDLEOCR_CONCURRENCY). 429s retry through auto_backoff; other errors
surface as hard errors. Bytes ride a multipart field; the result
jsonl and its assets download in memory. Per-model request options
and their measured rationale live in ``_options``; cache/lanes/poll
plumbing is shared (``_cloud``)."""

from __future__ import annotations

import asyncio
import json
import os

import httpx

from docspectra.parser import (ImageAsset, MAX_ASSETS, MAX_ASSET_BYTES,
                               ParseResult, provider)
from docspectra.providers import traced
from docspectra.providers._cloud import (ResultCache, lanes, mime_for_ext,
                                         poll_until)

from xtremeflow.scheduler.rate_limit import RetryException, auto_backoff
from xtremeflow.scheduler.request import RequestRateScheduler  # deep import: not re-exported

_OCR_MODEL = 'PP-OCRv6'  # the one ocr-pipeline model; the rest parse documents
_JOBS = 'https://paddleocr.aistudio-app.com/api/v2/ocr/jobs'
_AsyncClient = httpx.AsyncClient  # provider-local seam: tests patch this name


class PaddleOcrError(Exception):
    """Hard error: quota, auth, timeout, or a failed job — falls
    through the chain like any provider exception."""


def _options(model: str) -> dict:
    """Per-model request options, with the measured rationale — the
    single home for it. Orientation classify and unwarping are off:
    the server's defaults run them, and unwarping rewrites long
    posters into degraded markdown. The ocr pipeline's detection
    resolution rises to 4000 (the server's own max-side hard cap):
    the default 960 long-side limit downscales a tall poster ~5x and
    most of its text boxes with it. ``markdownIgnoreLabels`` is
    cleared — the server's default ignore list is what silences
    figure-adjacent text; the ocr pipeline has no markdown, so no
    ignore list applies (its textline-orientation classifier stays
    off)."""
    if model == _OCR_MODEL:
        return {'useDocOrientationClassify': False, 'useDocUnwarping': False,
                'useTextlineOrientation': False, 'textDetLimitSideLen': 4000,
                'textDetLimitType': 'max'}
    return {'useDocOrientationClassify': False, 'useDocUnwarping': False,
            'markdownIgnoreLabels': [], 'useChartRecognition': False}


def _data(res: httpx.Response) -> dict:
    """The response envelope's ``data`` — or a mapped error: 429s are
    rate limits (retried upstream through auto_backoff), the rest
    hard errors."""
    if res.status_code == 429:
        raise RetryException(f'paddleocr rate limited: {res.text[:200]}')
    if res.status_code >= 400:
        raise PaddleOcrError(f'paddleocr {res.status_code}: {res.text[:200]}')
    try:
        payload = res.json()
    except ValueError:
        raise PaddleOcrError(f'paddleocr malformed response: {res.text[:100]}')
    if payload.get('code', 0) not in (0, None):
        msg = payload.get('msg') or payload.get('errorMsg') or payload.get('message') or ''
        raise PaddleOcrError(f'paddleocr: {msg[:250]}')
    return payload['data']


def _headers(token: str) -> dict:
    return {'Authorization': f'Bearer {token}'}


@auto_backoff()
async def _submit(client: httpx.AsyncClient, token: str, model: str, data: bytes,
                  name: str | None) -> str:
    """One multipart submit → job id: the options ride a JSON-string
    form field, the file a multipart part (the API the SDK wraps)."""
    res = await client.post(_JOBS, headers=_headers(token),
                            data={'model': model,
                                  'optionalPayload': json.dumps(_options(model))},
                            files={'file': (name or 'file', data)},
                            timeout=300.0)
    return _data(res)['jobId']


@auto_backoff()
async def _status(client: httpx.AsyncClient, token: str, job_id: str) -> dict:
    res = await client.get(f'{_JOBS}/{job_id}', headers=_headers(token),
                           timeout=30.0)
    return _data(res)


async def _poll(client: httpx.AsyncClient, token: str, job_id: str, timeout: float,
                status_lane: RequestRateScheduler) -> str:
    """Drive the job to done on the status lane; the done check's
    payload is the result jsonl's url."""
    async def once():
        data = await _status(client, token, job_id)
        json_url = (data.get('resultUrl') or {}).get('jsonUrl') or ''
        return data.get('state'), data.get('errorMsg'), json_url
    url = await poll_until(status_lane, once, timeout=timeout,
                           label='paddleocr', error=PaddleOcrError)
    if not url:
        raise PaddleOcrError(f'paddleocr job {job_id} finished without a jsonl result')
    return url


async def _collect(model: str, lines: list[dict],
                   client: httpx.AsyncClient) -> tuple[str, list[ImageAsset], int]:
    """Jsonl lines → markdown + ImageAssets; assets reconcile with the
    markdown's ``![](...)`` references by the images dict each page
    ships. The ocr pipeline has no markdown — its recognized lines are
    the text, and it ships no content images (its images are
    visualizations)."""
    if model == _OCR_MODEL:
        recs = []
        for line in lines:
            for item in (line.get('result') or {}).get('ocrResults') or []:
                for t in (item.get('prunedResult') or {}).get('rec_texts') or []:
                    if t := t.strip():
                        recs.append(t)
        if not recs:
            raise PaddleOcrError('paddleocr result has no text lines')
        return '\n'.join(recs), [], 0
    pages = [item['markdown'] for line in lines
             for item in (line.get('result') or {}).get('layoutParsingResults') or []
             if item.get('markdown')]
    if not pages:
        raise PaddleOcrError('paddleocr result has no pages')
    pairs = [(path, url) for page in pages
             for path, url in (page.get('images') or {}).items()][:MAX_ASSETS]
    skipped = sum(len((page.get('images') or {})) for page in pages) - len(pairs)
    sem = asyncio.Semaphore(8)
    async def one(url):
        async with sem:
            return await _asset(client, url)
    payloads = await asyncio.gather(*(one(url) for _, url in pairs))
    images: list[ImageAsset] = []
    for (path, _), data in zip(pairs, payloads):
        if data is None or len(data) > MAX_ASSET_BYTES:
            skipped += 1
            continue
        images.append(ImageAsset(
            name=path, mime=mime_for_ext(path.rsplit('.', 1)[-1]),
            data=data, source=f'paddleocr {path}'))
    return '\n\n'.join(page.get('text') or '' for page in pages), images, skipped


async def _asset(client: httpx.AsyncClient, url: str) -> bytes | None:
    """One markdown image, in memory — None when the CDN declines, the
    fetch fails, or the body declares itself oversized (streamed, so
    the guard fires before the body is read); the parse records it as
    a skipped asset rather than failing the document over it."""
    try:
        async with client.stream('GET', url, timeout=30.0) as res:
            if res.status_code >= 400:
                return None
            if int(res.headers.get('Content-Length') or 0) > MAX_ASSET_BYTES:
                return None
            data = bytearray()
            async for chunk in res.aiter_bytes():
                data += chunk
                if len(data) > MAX_ASSET_BYTES:
                    return None
            return bytes(data)
    except (httpx.HTTPError, ValueError):
        return None


@provider('paddleocr', ('pdf', 'jpg', 'png', 'tiff', 'bmp'))
@traced
async def parse_paddleocr(data: bytes, *, name: str | None = None) -> ParseResult:
    token = os.environ.get('PADDLEOCR_API_TOKEN')
    if not token:
        raise PaddleOcrError('PADDLEOCR_API_TOKEN is not set')
    model = os.environ.get('PADDLEOCR_MODEL', _OCR_MODEL)
    key = ResultCache.key(data, model)
    if hit := _cache.facade(key, model=model, cache_hit=True):
        return hit
    timeout = float(os.environ.get('PADDLEOCR_TIMEOUT', '300'))
    submit_lane, status_lane = lanes('PADDLEOCR_CONCURRENCY', 'submit:60|status:600')
    async with _AsyncClient(timeout=30.0) as client:
        task = await submit_lane.start_task(_submit(client, token, model, data, name))
        job_id = await task
        jsonl_url = await _poll(client, token, job_id, timeout, status_lane)
        res = await client.get(jsonl_url, timeout=120.0)
        if res.status_code >= 400:
            raise PaddleOcrError(f'paddleocr result download failed: {res.status_code}')
        try:
            lines = [json.loads(line) for line in res.text.splitlines() if line.strip()]
        except json.JSONDecodeError as e:
            raise PaddleOcrError(f'paddleocr result payload is malformed: {e}') from e
        text, images, skipped = await _collect(model, lines, client)
    parsed = ParseResult(
        text=text, images=images,
        meta={'model': model, 'job_id': job_id, 'skipped_assets': skipped})
    _cache.put(key, parsed)
    return parsed


_cache = ResultCache()
