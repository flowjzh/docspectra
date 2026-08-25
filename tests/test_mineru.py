"""S15: the mineru cloud engine — upload→poll→zip flow, rate lanes,
retry, LRU — httpx MockTransport, zero network."""

import io
import zipfile

import httpx
import pytest

from docspectra.parser import ParseDeclinedError, configure, parse
from xtremeflow.scheduler.request import RequestRateScheduler

PDF = b'%PDF-1.4\nfake resume bytes'
PNG = b'\x89PNG\r\n\x1a\n' + b'x' * 32


def _zip(markdown: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('full.md', markdown)
        z.writestr('images/a.png', PNG)
    return buf.getvalue()


class _Flow:
    """Scripted MinerU: presigned URL, ``pending`` polls then done,
    optional one-shot 429s on file-urls or the upload, optional failure."""

    def __init__(self, markdown: str = '## result\n\nrecovered text',
                 pending: int = 1, retry_429: bool = False, upload_429: bool = False,
                 failed: str | None = None, zip_bytes: bytes | None = None):
        self.pending = pending
        self.retry_429 = retry_429
        self.upload_429 = upload_429
        self.failed = failed
        self.zip_bytes = zip_bytes if zip_bytes is not None else _zip(markdown)
        self.calls = 0
        self.file_urls_calls = 0
        self.upload_calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        path = request.url.path
        if path == '/api/v4/file-urls/batch':
            self.file_urls_calls += 1
            if self.retry_429 and self.file_urls_calls == 1:
                return httpx.Response(429, headers={'Retry-After': '0'}, json={'code': 429})
            return httpx.Response(200, json={'code': 0, 'data': {
                'batch_id': 'b1', 'file_urls': ['https://s3.example.com/upload']}})
        if path == '/upload':
            self.upload_calls += 1
            if self.upload_429 and self.upload_calls == 1:
                return httpx.Response(429, headers={'Retry-After': '0'})
            return httpx.Response(200, text='ok')
        if path == '/api/v4/extract-results/batch/b1':
            if self.failed:
                return httpx.Response(200, json={'code': 0, 'data': {'extract_result': [
                    {'state': 'failed', 'err_msg': self.failed}]}})
            if self.pending > 0:
                self.pending -= 1
                return httpx.Response(200, json={'code': 0, 'data': {'extract_result': [
                    {'state': 'pending'}]}})
            return httpx.Response(200, json={'code': 0, 'data': {'extract_result': [
                {'state': 'done', 'full_zip_url': 'https://s3.example.com/result.zip'}]}})
        if path == '/result.zip':
            return httpx.Response(200, content=self.zip_bytes)
        return httpx.Response(404, text=path)


@pytest.fixture
def env(monkeypatch):
    """Test token, clean LRU; patch_client swaps the provider's client."""
    monkeypatch.setenv('MINERU_API_TOKEN', 'test-token')
    from docspectra.providers import mineru
    mineru._cache.clear()
    yield mineru
    mineru._cache.clear()


@pytest.fixture
def patch_client(monkeypatch):
    def patch(handler):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: client)

    return patch


@pytest.fixture
def mineru_chain():
    configure({'pdf': ['mineru']})
    yield
    configure(None)


async def test_upload_poll_zip_flow(env, patch_client, mineru_chain):
    flow = _Flow()
    patch_client(flow)
    result = await parse(PDF, name='resume.pdf')
    assert result.provider == 'mineru'
    assert 'recovered text' in result.text
    assert result.meta['batch_id'] == 'b1'
    assert result.meta['model_version'] == 'pipeline'
    assert len(result.images) == 1
    asset = result.images[0]
    assert asset.name == 'images/a.png'
    assert asset.mime == 'image/png'
    assert asset.data == PNG
    assert asset.source == 'mineru images/a.png'


async def test_quota_rejection_is_a_hard_error(env, patch_client, mineru_chain):
    def handler(request):
        return httpx.Response(402, json={'code': 402, 'msg': 'quota exhausted'})

    patch_client(handler)
    with pytest.raises(ParseDeclinedError) as exc:
        await parse(PDF, name='resume.pdf')
    assert exc.value.attempts[0].outcome == 'error'
    assert '402' in str(exc.value.__cause__)


async def test_rate_limit_is_retried_then_succeeds(env, patch_client, mineru_chain):
    flow = _Flow(retry_429=True)
    patch_client(flow)
    result = await parse(PDF, name='resume.pdf')
    assert result.provider == 'mineru'
    assert flow.file_urls_calls == 2  # 429 then success
    assert 'recovered text' in result.text


async def test_upload_429_is_retried(env, patch_client, mineru_chain):
    flow = _Flow(upload_429=True)
    patch_client(flow)
    result = await parse(PDF, name='resume.pdf')
    assert result.provider == 'mineru'
    assert flow.upload_calls == 2  # 429 then success
    assert 'recovered text' in result.text


async def test_extract_failure_is_a_hard_error(env, patch_client, mineru_chain):
    patch_client(_Flow(failed='cannot parse document'))
    with pytest.raises(ParseDeclinedError) as exc:
        await parse(PDF, name='resume.pdf')
    assert 'cannot parse document' in str(exc.value.__cause__)


async def test_zip_without_markdown_is_a_hard_error(env, patch_client, mineru_chain):
    no_markdown = io.BytesIO()
    with zipfile.ZipFile(no_markdown, 'w') as z:
        z.writestr('images/a.png', PNG)
    patch_client(_Flow(zip_bytes=no_markdown.getvalue()))
    with pytest.raises(ParseDeclinedError) as exc:
        await parse(PDF, name='resume.pdf')
    assert 'no full.md' in str(exc.value.__cause__)


async def test_timeout_is_a_hard_error(env, patch_client, mineru_chain, monkeypatch):
    patch_client(_Flow(pending=99))
    monkeypatch.setenv('MINERU_TIMEOUT', '0')  # deadline passed right after the first poll
    with pytest.raises(ParseDeclinedError, match='timed out'):
        await parse(PDF, name='resume.pdf')


async def test_lru_makes_second_call_zero_requests(env, patch_client, mineru_chain):
    flow = _Flow()
    patch_client(flow)
    first = await parse(PDF, name='resume.pdf')
    calls_after_first = flow.calls
    second = await parse(PDF, name='resume.pdf')
    assert second.text == first.text
    assert second.meta['cache_hit'] is True
    assert 'batch_id' not in second.meta  # per-call facade, not call 1's meta
    assert flow.calls == calls_after_first


def test_ticket_spacing_pin():
    """50/min ÷ 60 → 1.2 s spacing. The frozen lib annotates max_rps
    Optional[int]; int coercion of 0.83 → 0 would disable limiting —
    this test fails loudly if a future version ever enforces the
    annotation."""
    scheduler = RequestRateScheduler(max_rps=50 / 60, max_concurrency=8)
    scheduler._reserve_ticket()
    wait = scheduler._reserve_ticket()
    assert 1.1 <= wait <= 1.3


def test_malformed_concurrency_spec(env, monkeypatch):
    monkeypatch.setenv('MINERU_CONCURRENCY', 'submit:50')  # missing status lane
    env._lanes_cache = None
    with pytest.raises(ValueError, match='status'):
        env._lanes()


async def test_missing_token_is_a_hard_error(env, mineru_chain, monkeypatch):
    monkeypatch.delenv('MINERU_API_TOKEN')
    with pytest.raises(ParseDeclinedError) as exc:
        await parse(PDF, name='resume.pdf')
    assert 'MINERU_API_TOKEN' in str(exc.value.__cause__)
