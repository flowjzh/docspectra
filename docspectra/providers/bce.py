"""Cloud engine (extra: bce): Baidu Cloud Engine's document parsers
over direct httpx — oauth token, form-encoded task submit, task/query
poll, markdown download. One provider, one protocol, two products: the
endpoint path and the options knobs are all that differ between them
(the submit/poll/result envelope is field-identical). ``unlimited-ocr-
parser`` is the default — its markdown serializes table regions as
text lines rather than HTML tables, which is the shape the chunker
and the router address. ``paddle-vl-parser`` stays selectable via
``BCE_MODEL`` as the VL alternative. Per-product request options and
their rationale live in ``_options``; token/lanes/poll plumbing is
shared (``_cloud``); rate rejections and HTTP 429 retry through
auto_backoff, everything else surfaces as a hard error.

The declared types are the service's documented input families mapped
onto docspectra's canonical types. A deployment that wants bce off a
family pins that type's chain through ``DOCSPECTRA_PARSER_CONFIG``."""

from __future__ import annotations

import asyncio
import base64
import os
import time
from urllib.parse import urlencode

import httpx

from docspectra.parser import ParseResult, provider, sniff
from docspectra.providers import traced
from docspectra.providers._cloud import (ResultCache, fetch, json_body,
                                         poll_until, run_job)

from xtremeflow.scheduler.rate_limit import RetryException, auto_backoff
from xtremeflow.scheduler.request import RequestRateScheduler  # deep import: not re-exported

_UNLIMITED = 'unlimited-ocr-parser'  # default product
_PADDLE_VL = 'paddle-vl-parser'
_PRODUCTS = {
    _UNLIMITED: {'angle_adjust': False},
    _PADDLE_VL: {'angle_adjust': False, 'recognize_seal': False},
}
_TASKS = ('https://aip.baidubce.com/rest/2.0/brain/online/v2/'
          '{product}/task')
_QUERY = _TASKS + '/query'
_OAUTH = 'https://aip.baidubce.com/oauth/2.0/token'
_FORM = {'Content-Type': 'application/x-www-form-urlencoded'}
_AsyncClient = httpx.AsyncClient  # provider-local seam: tests patch this name

# transient rate rejections worth retrying: the openapi QPS code and the
# account's per-app concurrency ceiling
_RETRY_CODES = frozenset({18, 282118})


class BceError(Exception):
    """Hard error: quota, auth, timeout, or a failed task — falls
    through the chain like any provider exception."""


def _options(model: str) -> dict:
    """Per-product request options, with the rationale — the single
    home for it. Only deviations from a product's server defaults are
    pinned: server-side drift must not change extraction behavior.
    ``angle_adjust`` off on both products: resumes are upright, and a
    whole-page rotation is a failure mode the text pipeline cannot
    detect. paddle-vl runs seal recognition by default — no resume
    field can hold seal text, so it is pure latency."""
    return dict(_PRODUCTS[model])


def _data(res: httpx.Response) -> dict:
    """The response envelope's ``result`` — the shared HTTP ladder plus
    this service's business codes: the QPS/concurrency rejections retry
    through auto_backoff, the rest are hard errors."""
    payload = json_body(res, label='bce', error=BceError)
    code = payload.get('error_code') or 0
    msg = payload.get('error_msg') or ''
    if code in _RETRY_CODES:
        raise RetryException(f'bce rate limited: {msg[:200]}')
    if code:
        raise BceError(f'bce {code}: {msg[:250]}')
    return payload.get('result') or {}


_token: tuple[str, float] | None = None  # (access_token, monotonic expiry)
_token_lock = asyncio.Lock()


async def _access_token(client: httpx.AsyncClient, api_key: str,
                        secret_key: str) -> str:
    """The oauth token, reused to expiry (minus a minute of skew): every
    parse would otherwise pay the round trip, and a cold fleet's first
    burst would race N fetches into the account's ceiling."""
    global _token
    async with _token_lock:
        if _token and _token[1] > time.monotonic() + 60:
            return _token[0]
        res = await client.get(_OAUTH, params={
            'grant_type': 'client_credentials',
            'client_id': api_key, 'client_secret': secret_key}, timeout=30.0)
        payload = json_body(res, label='bce auth', error=BceError)
        if 'access_token' not in payload:
            raise BceError('bce auth response carries no access token')
        _token = (payload['access_token'],
                  time.monotonic() + int(payload.get('expires_in') or 0))
        return _token[0]


def _form(data: bytes, name: str | None, model: str) -> bytes:
    """The one form-encoded submit body, built once per job — the
    upload's only shape is base64 in ``file_data``. Encoding a 20 MB
    file is a second of CPU, and a rate-limit retry must not pay it
    again: the retry re-POSTs these bytes. The API derives the parser
    from the extension, so an unnamed file carries the one sniff
    already established."""
    suffix = sniff(data) or 'pdf'
    return urlencode({
        'file_data': base64.b64encode(data).decode(),
        'file_name': name or f'document.{suffix}',
        **_options(model),
    }).encode()


@auto_backoff()
async def _submit(client: httpx.AsyncClient, token: str, model: str,
                  body: bytes) -> str:
    """One submit → task id."""
    res = await client.post(
        _TASKS.format(product=model), params={'access_token': token},
        headers=_FORM, content=body, timeout=300.0)
    task_id = _data(res).get('task_id') or ''
    if not task_id:
        raise BceError('bce submit returned no task id')
    return task_id


@auto_backoff()
async def _status(client: httpx.AsyncClient, token: str, model: str,
                  task_id: str) -> dict:
    res = await client.post(
        _QUERY.format(product=model), params={'access_token': token},
        headers=_FORM, data={'task_id': task_id}, timeout=30.0)
    return _data(res)


async def _poll(client: httpx.AsyncClient, token: str, model: str,
                task_id: str, timeout: float,
                status_lane: RequestRateScheduler) -> dict:
    """Drive the task to success on the status lane; the done check's
    payload is the result envelope (markdown_url, cost_page_num)."""
    async def once():
        data = await _status(client, token, model, task_id)
        state = data.get('status')
        if state == 'success':
            return 'done', None, data
        if state == 'failed':
            return 'failed', data.get('task_error') or 'unknown error', None
        return state, None, None
    return await poll_until(status_lane, once, timeout=timeout,
                            label='bce', error=BceError)


async def _collect(client: httpx.AsyncClient, result: dict, model: str,
                   task_id: str) -> ParseResult:
    """The whole-document markdown — the parser's text contract; the
    products' structured result (pages/tables/cells) has no consumer in
    the text pipeline, so only the markdown url is fetched."""
    url = result.get('markdown_url') or ''
    if not url:
        raise BceError('bce task succeeded without a markdown url')
    text = await fetch(client, url, label='bce markdown download',
                       error=BceError)
    return ParseResult(
        text=text, images=[],
        meta={'model': model, 'job_id': task_id,
              'pages': int(result.get('cost_page_num') or 0)})


@provider('bce', ('pdf', 'ofd', 'doc', 'docx', 'ppt', 'pptx', 'txt', 'wps',
                  'ods', 'tsv', 'rtf', 'odt', 'odp',
                  'jpg', 'png', 'bmp', 'tiff', 'webp', 'gif', 'ico', 'svg'))
@traced
async def parse_bce(data: bytes, *, name: str | None = None) -> ParseResult:
    api_key = os.environ.get('BCE_API_KEY')
    secret_key = os.environ.get('BCE_SECRET_KEY')
    if not api_key or not secret_key:
        raise BceError('BCE_API_KEY / BCE_SECRET_KEY are not set')
    model = os.environ.get('BCE_MODEL', _UNLIMITED)
    if model not in _PRODUCTS:
        raise BceError(f'BCE_MODEL {model!r} is not a known product: '
                       f'{sorted(_PRODUCTS)}')

    async def flow(client, submit_lane, status_lane, timeout):
        # the token fetch and the body encode are independent; both stay
        # outside the submit lane, which only prices the API call itself
        body, token = await asyncio.gather(
            asyncio.to_thread(_form, data, name, model),
            _access_token(client, api_key, secret_key))
        task_id = await (await submit_lane.start_task(
            _submit(client, token, model, body)))
        result = await _poll(client, token, model, task_id, timeout,
                             status_lane)
        return await _collect(client, result, model, task_id)

    return await run_job(
        cache=_cache, data=data, model=model, facet='model',
        timeout_env='BCE_TIMEOUT',
        # the service caps both endpoints at 2 QPS (submit and result
        # query); one lane each, exactly at the ceiling — the retry
        # backoff absorbs a burst, not a lower steady rate
        lanes_env='BCE_CONCURRENCY', lanes_spec='submit:120|status:120',
        client_factory=_AsyncClient, flow=flow)


_cache = ResultCache()
