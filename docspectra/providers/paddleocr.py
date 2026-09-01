"""Cloud engine (extra: paddleocr): the PaddleOCR official API through
the official SDK (``AsyncPaddleOCRClient`` — submit, poll, parse),
rate-limited by dual xtremeflow lanes (the service publishes no
ceilings; conservative defaults, tunable via PADDLEOCR_CONCURRENCY).
SDK rate-limit errors retry through auto_backoff; other SDK errors
surface as hard errors. Bytes ride a temp file — the SDK's submit
takes a path — and assets download in memory over the aiohttp the SDK
ships. Per-model request options and their measured rationale live in
``_options``; cache/lanes/poll plumbing is shared (``_cloud``)."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import aiohttp

from docspectra.parser import (ImageAsset, MAX_ASSETS, MAX_ASSET_BYTES,
                               ParseResult, provider)
from docspectra.providers import traced
from docspectra.providers._cloud import ResultCache, lanes, mime_for_ext, poll_until

from paddleocr import (AsyncPaddleOCRClient, OCROptions, PPStructureV3Options,
                       PaddleOCRVLOptions)
from paddleocr._api_client.errors import (PollTimeoutError,  # not re-exported
                                          PaddleOCRAPIError, RateLimitError)

from xtremeflow.scheduler.rate_limit import RetryException, auto_backoff
from xtremeflow.scheduler.request import RequestRateScheduler  # deep import: not re-exported

_OCR_MODEL = 'PP-OCRv6'  # the one ocr-pipeline model; the rest parse documents


class PaddleOcrError(Exception):
    """Hard error: quota, auth, timeout, or a failed job — falls
    through the chain like any provider exception."""


def _options(model: str):
    """Per-model request options, with the measured rationale — the
    single home for it. Orientation classify and unwarping are off:
    the server's defaults run them, and unwarping rewrites long
    posters into degraded markdown. The ocr pipeline's detection
    resolution rises to 4000 (the server's own max-side hard cap):
    the default 960 long-side limit downscales a tall poster ~5x and
    most of its text boxes with it. ``markdown_ignore_labels`` is
    cleared — the server's default ignore list is what silences
    figure-adjacent text; the ocr pipeline has no markdown, so no
    ignore list applies (its textline-orientation classifier stays
    off)."""
    if model == _OCR_MODEL:
        return OCROptions(use_doc_orientation_classify=False,
                          use_doc_unwarping=False, use_textline_orientation=False,
                          text_det_limit_side_len=4000, text_det_limit_type='max')
    options = {'use_doc_orientation_classify': False, 'use_doc_unwarping': False,
               'markdown_ignore_labels': [], 'use_chart_recognition': False}
    return PPStructureV3Options(**options) if model == 'PP-StructureV3' \
        else PaddleOCRVLOptions(**options)


async def _via_sdk(call, *args, **kwargs):
    """One SDK call with its errors mapped to the chain contract: rate
    limits retry through auto_backoff, timeouts and the rest fall
    through as hard errors."""
    try:
        return await call(*args, **kwargs)
    except RateLimitError as e:
        raise RetryException(f'paddleocr rate limited: {e}') from e
    except PollTimeoutError as e:
        raise PaddleOcrError(f'paddleocr timed out: {e}') from e
    except PaddleOCRAPIError as e:
        raise PaddleOcrError(f'{type(e).__name__}: {e}') from e


@auto_backoff()
async def _submit(client: AsyncPaddleOCRClient, model: str, data: bytes,
                  name: str | None) -> str:
    """Bytes → temp file (the SDK's submit takes a path and reads it
    within the call) → job id. The file lives for one submit attempt —
    a retry mints a fresh one — and the suffix rides along: the service
    sniffs nothing."""
    fd, path = tempfile.mkstemp(suffix=Path(name or '').suffix)
    try:
        await asyncio.to_thread(os.write, fd, data)
        submit = client.submit_ocr if model == _OCR_MODEL else client.submit_document_parsing
        job = await _via_sdk(submit, model=model, file_path=path,
                             options=_options(model))
    finally:
        os.close(fd)
        await asyncio.to_thread(os.unlink, path)
    return job.job_id


@auto_backoff()
async def _status(client: AsyncPaddleOCRClient, job_id: str):
    return await _via_sdk(client.get_status, job_id)


async def _await_done(client: AsyncPaddleOCRClient, job_id: str, timeout: float,
                      status_lane: RequestRateScheduler) -> None:
    """Drive the job to done on the status lane — the SDK's own waiter
    would poll off-lane."""
    async def once():
        status = await _status(client, job_id)
        return status.state, status.error_msg, None
    await poll_until(status_lane, once, timeout=timeout,
                     label='paddleocr', error=PaddleOcrError)


async def _collect(model: str, result) -> tuple[str, list[ImageAsset], int]:
    """SDK result → markdown + ImageAssets; assets reconcile with the
    markdown's ``![](...)`` references by the images dict each page
    ships. The ocr pipeline has no markdown — its recognized lines are
    the text, and it ships no content images (its images are
    visualizations)."""
    if model == _OCR_MODEL:
        lines = [t.strip() for page in result.pages
                 for t in (page.pruned_result or {}).get('rec_texts') or []
                 if t.strip()]
        if not lines:
            raise PaddleOcrError('paddleocr result has no text lines')
        return '\n'.join(lines), [], 0
    pages = [page.markdown_text for page in result.pages]
    if not pages:
        raise PaddleOcrError('paddleocr result has no pages')
    pairs = [(asset_path, url) for page in result.pages
             for asset_path, url in page.markdown_images.items()][:MAX_ASSETS]
    skipped = sum(len(page.markdown_images) for page in result.pages) - len(pairs)
    # connector limit: MAX_ASSETS bounds the pair count, not the burst
    timeout = aiohttp.ClientTimeout(total=30)
    connector = aiohttp.TCPConnector(limit=8)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        payloads = await asyncio.gather(
            *(_asset(session, url) for _, url in pairs))
    images: list[ImageAsset] = []
    for (asset_path, _), data in zip(pairs, payloads):
        if data is None or len(data) > MAX_ASSET_BYTES:
            skipped += 1
            continue
        images.append(ImageAsset(
            name=asset_path, mime=mime_for_ext(asset_path.rsplit('.', 1)[-1]),
            data=data, source=f'paddleocr {asset_path}'))
    return '\n\n'.join(pages), images, skipped


async def _asset(session: aiohttp.ClientSession, url: str):
    """One markdown image, in memory — None when the CDN declines, the
    fetch fails, or the body declares itself oversized (checked before
    reading); the parse records it as a skipped asset rather than
    failing the document over it."""
    try:
        async with session.get(url) as resp:
            if resp.status >= 400:
                return None
            if int(resp.headers.get('Content-Length') or 0) > MAX_ASSET_BYTES:
                return None
            return await resp.read()
    except (aiohttp.ClientError, ValueError):
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
    client = AsyncPaddleOCRClient(token=token, poll_timeout=timeout)
    try:
        task = await submit_lane.start_task(_submit(client, model, data, name))
        job_id = await task
        await _await_done(client, job_id, timeout, status_lane)
        wait = client.wait_ocr_result if model == _OCR_MODEL \
            else client.wait_document_parsing_result
        result = await _via_sdk(wait, job_id)
        text, images, skipped = await _collect(model, result)
    finally:
        await client.close()
    parsed = ParseResult(
        text=text, images=images,
        meta={'model': model, 'job_id': job_id, 'skipped_assets': skipped})
    _cache.put(key, parsed)
    return parsed


_cache = ResultCache()
