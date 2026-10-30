"""Shared plumbing behind the cloud engines — the byte-budget result
LRU, the dual rate lanes, the HTTP outcome ladder, the result
download, the job-poll state machine and the job lifecycle their flows
share. Local engines share ``pdfscan``; cloud ones share this.
Mirroring an engine's flow is fine; copying this file is not."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import os
import time
from collections import OrderedDict

import httpx

from docspectra.parser import ParseResult

from xtremeflow.scheduler.rate_limit import RetryException
from xtremeflow.scheduler.request import RequestRateScheduler  # deep import: not re-exported

MIME_BY_EXT = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png',
               'gif': 'image/gif', 'webp': 'image/webp', 'svg': 'image/svg+xml',
               'bmp': 'image/bmp', 'tiff': 'image/tiff'}


def mime_for_ext(ext: str) -> str:
    return MIME_BY_EXT.get(ext.lower(), 'application/octet-stream')


def check_http(res: httpx.Response, *, label: str, error,
               retry_5xx: bool = False) -> None:
    """The shared outcome ladder: 429 (and 5xx for services that use
    it as backpressure) is a retry, every other >=400 a hard error.
    ``error`` builds that engine's exception; ``Retry-After`` rides the
    retry when the service sends one."""
    if res.status_code < 400:
        return
    msg = f'{label} {res.status_code}: {res.text[:200]}'
    if res.status_code == 429 or (retry_5xx and res.status_code >= 500):
        retry_after = res.headers.get('Retry-After')
        raise RetryException(
            msg, retry_after=float(retry_after) if retry_after else None)
    raise error(msg)


def json_body(res: httpx.Response, *, label: str, error) -> dict:
    """The checked JSON body: ``check_http`` plus a malformed-body
    error. Envelope-level error codes stay with the engine — their
    field names and retry codes are the product's own."""
    check_http(res, label=label, error=error)
    try:
        return res.json()
    except ValueError:
        raise error(f'{label} malformed response: {res.text[:100]}')


async def fetch(client: httpx.AsyncClient, url: str, *, label: str, error,
                text: bool = True):
    """One result download (markdown, jsonl or zip) behind the shared
    status ladder — ``text`` picks ``res.text`` over ``res.content``.
    A result URL is fetched outside any retry wrapper, so a 429 here
    surfaces as the engine's hard error instead of a retry signal
    nothing can act on."""
    res = await client.get(url, timeout=120.0)
    try:
        check_http(res, label=label, error=error)
    except RetryException as e:
        raise error(str(e)) from e
    return res.text if text else res.content


@functools.cache
def lanes(env_var: str, default_spec: str
          ) -> tuple[RequestRateScheduler, RequestRateScheduler]:
    """The dual submit/status lanes one cloud engine drives its jobs
    through — memoized per (env_var, spec): each engine declares its
    spec once at the call site. The spec — ``'submit:N|status:N'``,
    per-minute ceilings divided by 60 into rps — rides ``env_var``;
    the 8/32 concurrency caps are deliberate shared policy, not part
    of the bargain."""
    spec = dict(kv.split(':', 1) for kv in
                os.environ.get(env_var, default_spec).split('|'))
    missing = [k for k in ('submit', 'status') if k not in spec]
    if missing:
        raise ValueError(f'{env_var} is missing the {missing} lane')
    return (RequestRateScheduler(max_rps=float(spec['submit']) / 60, max_concurrency=8),
            RequestRateScheduler(max_rps=float(spec['status']) / 60, max_concurrency=32))


async def poll_until(status_lane: RequestRateScheduler, once, *, timeout: float,
                     interval: float = 2.0, label: str, error):
    """Drive one cloud job to done on the status lane — the deadline,
    the failure branch, the poll cadence, and the lane discipline live
    here once. ``once`` runs a single status check and returns
    ``(state, error_msg, payload)``; the payload from the done check
    is returned to the engine (a result url, or None), and ``error``
    builds that engine's hard error."""
    deadline = time.monotonic() + timeout
    while True:
        task = await status_lane.start_task(once())
        state, error_msg, payload = await task
        if state == 'done':
            return payload
        if state == 'failed':
            raise error(f'{label} extract failed: {error_msg}')
        if time.monotonic() >= deadline:
            raise error(f'{label} timed out after {timeout:.0f}s (state {state})')
        await asyncio.sleep(interval)


async def run_job(*, cache, data: bytes, model: str, facet: str,
                  timeout_env: str, lanes_env: str, lanes_spec: str,
                  client_factory, flow) -> ParseResult:
    """The invariant part of one cloud job: the LRU facade (keyed by
    content and model), the timeout env, the dual lanes, one client, and
    the cache write. ``flow(client, submit_lane, status_lane, timeout)``
    runs the engine's own submit → poll → collect and returns the
    ParseResult — keeping the lane calls in the flow is what lets an
    engine step outside the lanes (mineru's S3 upload)."""
    key = ResultCache.key(data, model)
    if hit := cache.facade(key, **{facet: model}, cache_hit=True):
        return hit
    timeout = float(os.environ.get(timeout_env, '300'))
    submit_lane, status_lane = lanes(lanes_env, lanes_spec)
    async with client_factory(timeout=30.0) as client:
        result = await flow(client, submit_lane, status_lane, timeout)
    cache.put(key, result)
    return result


class ResultCache:
    """One engine's in-process result LRU keyed by (sha256, model) —
    memory only, never disk, evicted by count and byte budget (~1 MB
    per file, 8 MB resident)."""

    _MAX_ENTRIES = 8
    _MAX_BYTES = 8 * 1024 * 1024

    def __init__(self):
        self._entries: OrderedDict[tuple[str, str], ParseResult] = OrderedDict()
        self._bytes = 0

    @staticmethod
    def key(data: bytes, model: str) -> tuple[str, str]:
        return (hashlib.sha256(data).hexdigest(), model)

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0

    def facade(self, key: tuple[str, str], **meta) -> ParseResult | None:
        """The cached result shared as a per-call facade: the immutable
        payload (text, images) with call-specific meta — never the
        cached meta (a batch/job id for a job this call never
        submitted)."""
        if (cached := self.get(key)) is None:
            return None
        return ParseResult(text=cached.text, images=cached.images, meta=meta)

    @staticmethod
    def _size(result: ParseResult) -> int:
        return len(result.text) + sum(len(img.data) for img in result.images)

    def get(self, key: tuple[str, str]) -> ParseResult | None:
        if cached := self._entries.get(key):
            self._entries.move_to_end(key)
        return cached

    def put(self, key: tuple[str, str], result: ParseResult) -> None:
        self._entries[key] = result
        self._entries.move_to_end(key)
        self._bytes += self._size(result)
        while len(self._entries) > self._MAX_ENTRIES or self._bytes > self._MAX_BYTES:
            _, old = self._entries.popitem(last=False)
            self._bytes -= self._size(old)
