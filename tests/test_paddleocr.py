"""The paddleocr cloud engine over the official SDK — submit→poll→
wait flow, markdown_ignore_labels knob, rate lanes, retry, LRU — the
SDK client stubbed, zero network."""

from types import SimpleNamespace

import pytest

from docspectra.parser import ParseDeclinedError, configure, parse
from docspectra.providers.paddleocr import _options as paddle_options
from paddleocr import OCROptions, PPStructureV3Options, PaddleOCRVLOptions
from paddleocr._api_client.errors import (PaddleOCRAPIError,  # not re-exported
                                          RateLimitError)

from tests.helpers import PDF, PNG


def _page(markdown: str, images: dict | None = None, recs: list | None = None):
    pruned = {'rec_texts': recs} if recs is not None else None
    return SimpleNamespace(markdown_text=markdown,
                           markdown_images=images or {}, pruned_result=pruned)


class _StubClient:
    """Scripted AsyncPaddleOCRClient: submits recorded, status states
    popped in order, wait_* returns the canned pages. Asset fetch
    (``_asset``) is patched by the ``asset_bytes`` fixture."""

    def __init__(self, pages: list, states: list = ('running', 'done'),
                 fail_submit: Exception | None = None):
        self.pages = pages
        self.states = list(states)
        self.fail_submit = fail_submit
        self.submits = []  # (kind, model, file_path, options)
        self.status_calls = 0

    async def _submit(self, kind, model=None, file_path=None, options=None, **kw):
        self.submits.append((kind, model, file_path, options))
        if self.fail_submit is not None:
            err, self.fail_submit = self.fail_submit, None
            raise err
        return SimpleNamespace(job_id='j1')

    async def submit_document_parsing(self, **kw):
        return await self._submit('document', **kw)

    async def submit_ocr(self, **kw):
        return await self._submit('ocr', **kw)

    async def get_status(self, job_id):
        self.status_calls += 1
        state = self.states.pop(0) if self.states else 'done'
        if state == 'failed':
            return SimpleNamespace(state='failed', error_msg='cannot parse document')
        return SimpleNamespace(state=state, error_msg=None)

    async def wait_document_parsing_result(self, job_id):
        return SimpleNamespace(job_id=job_id, pages=self.pages)

    async def wait_ocr_result(self, job_id):
        return SimpleNamespace(job_id=job_id, pages=self.pages)

    async def close(self):
        pass


@pytest.fixture
def asset_bytes(monkeypatch):
    """Patch the in-memory asset fetch: every URL yields PNG."""
    async def fetch(session, url):
        return PNG

    monkeypatch.setattr('docspectra.providers.paddleocr._asset', fetch)


@pytest.fixture
def env(monkeypatch):
    """Test token and a clean LRU; install_client swaps the SDK client."""
    monkeypatch.setenv('PADDLEOCR_API_TOKEN', 'test-token')
    from docspectra.providers import paddleocr
    paddleocr._cache.clear()
    yield paddleocr
    paddleocr._cache.clear()


@pytest.fixture
def install_client(monkeypatch):
    def install(stub):
        monkeypatch.setattr('docspectra.providers.paddleocr.AsyncPaddleOCRClient',
                            lambda **kw: stub)
    return install


@pytest.fixture
def paddle_chain():
    configure({'pdf': ['paddleocr']})
    yield
    configure(None)


async def test_submit_poll_wait_flow(env, install_client, paddle_chain):
    stub = _StubClient([_page('', recs=['姓名 张三', '工作经历'])])
    install_client(stub)
    result = await parse(PDF, name='resume.pdf')
    assert result.provider == 'paddleocr'
    assert result.text == '姓名 张三\n工作经历'
    assert result.meta['job_id'] == 'j1'
    assert result.meta['model'] == 'PP-OCRv6'
    kind, model, path, options = stub.submits[0]
    assert kind == 'ocr' and model == 'PP-OCRv6' and path.endswith('.pdf')
    assert isinstance(options, OCROptions)
    assert options.use_doc_orientation_classify is False
    assert options.use_doc_unwarping is False
    assert options.text_det_limit_side_len == 4000
    assert options.text_det_limit_type == 'max'
    assert result.images == []


async def test_document_flow_wants_the_full_markdown(env, install_client, paddle_chain,
                                                     monkeypatch, asset_bytes):
    monkeypatch.setenv('PADDLEOCR_MODEL', 'PP-StructureV3')
    stub = _StubClient([_page('## name\n\nbody', images={'imgs/a.png': 'https://cdn/x.png'})])
    install_client(stub)
    result = await parse(PDF, name='resume.pdf')
    kind, model, path, options = stub.submits[0]
    assert kind == 'document' and model == 'PP-StructureV3' and path.endswith('.pdf')
    assert isinstance(options, PPStructureV3Options)
    assert options.markdown_ignore_labels == []
    assert options.use_doc_orientation_classify is False
    assert options.use_doc_unwarping is False
    assert options.use_chart_recognition is False
    assert '## name' in result.text
    assert len(result.images) == 1
    asset = result.images[0]
    assert asset.name == 'imgs/a.png'
    assert asset.mime == 'image/png'
    assert asset.data == PNG
    assert asset.source == 'paddleocr imgs/a.png'


def test_options_per_model():
    assert isinstance(structure := paddle_options('PP-StructureV3'), PPStructureV3Options)
    assert isinstance(vl := paddle_options('PaddleOCR-VL-1.6'), PaddleOCRVLOptions)
    assert structure.markdown_ignore_labels == []
    assert vl.markdown_ignore_labels == []
    assert structure.use_doc_unwarping is False
    assert vl.use_doc_unwarping is False
    ocr = paddle_options('PP-OCRv6')
    assert isinstance(ocr, OCROptions)
    assert ocr.use_doc_unwarping is False
    assert ocr.use_textline_orientation is False
    assert ocr.text_det_limit_side_len == 4000
    assert ocr.text_det_limit_type == 'max'


async def test_pages_join_in_order(env, install_client, paddle_chain, monkeypatch):
    monkeypatch.setenv('PADDLEOCR_MODEL', 'PP-StructureV3')
    install_client(_StubClient([_page('page one'), _page('page two')]))
    result = await parse(PDF, name='resume.pdf')
    assert result.text == 'page one\n\npage two'


async def test_asset_failure_is_skipped(env, install_client, paddle_chain, monkeypatch):
    monkeypatch.setenv('PADDLEOCR_MODEL', 'PP-StructureV3')

    async def failing_asset(session, url):
        return None

    monkeypatch.setattr('docspectra.providers.paddleocr._asset', failing_asset)
    stub = _StubClient([_page('body', images={'imgs/a.png': 'https://cdn/x.png'})])
    install_client(stub)
    result = await parse(PDF, name='resume.pdf')
    assert 'body' in result.text
    assert result.images == []
    assert result.meta['skipped_assets'] == 1


async def test_quota_rejection_is_a_hard_error(env, install_client, paddle_chain):
    install_client(_StubClient([], fail_submit=PaddleOCRAPIError('quota exhausted')))
    with pytest.raises(ParseDeclinedError) as exc:
        await parse(PDF, name='resume.pdf')
    assert exc.value.attempts[0].outcome == 'error'
    assert 'quota exhausted' in str(exc.value.__cause__)


async def test_rate_limit_is_retried_then_succeeds(env, install_client, paddle_chain):
    stub = _StubClient([_page('', recs=['recovered text'])],
                       fail_submit=RateLimitError('slow down'))
    install_client(stub)
    result = await parse(PDF, name='resume.pdf')
    assert result.provider == 'paddleocr'
    assert len(stub.submits) == 2  # 429 then success
    assert 'recovered text' in result.text


async def test_job_failure_is_a_hard_error(env, install_client, paddle_chain):
    install_client(_StubClient([_page('')], states=['failed']))
    with pytest.raises(ParseDeclinedError) as exc:
        await parse(PDF, name='resume.pdf')
    assert 'cannot parse document' in str(exc.value.__cause__)


async def test_timeout_is_a_hard_error(env, install_client, paddle_chain, monkeypatch):
    install_client(_StubClient([_page('')], states=['running'] * 99))
    monkeypatch.setenv('PADDLEOCR_TIMEOUT', '0')  # deadline passed right after the first poll
    with pytest.raises(ParseDeclinedError, match='timed out'):
        await parse(PDF, name='resume.pdf')


async def test_lru_makes_second_call_zero_requests(env, install_client, paddle_chain):
    stub = _StubClient([_page('', recs=['recovered text'])])
    install_client(stub)
    first = await parse(PDF, name='resume.pdf')
    second = await parse(PDF, name='resume.pdf')
    assert second.text == first.text
    assert second.meta['cache_hit'] is True
    assert 'job_id' not in second.meta  # per-call facade, not call 1's meta
    assert len(stub.submits) == 1


async def test_missing_token_is_a_hard_error(env, paddle_chain, monkeypatch):
    monkeypatch.delenv('PADDLEOCR_API_TOKEN')
    with pytest.raises(ParseDeclinedError) as exc:
        await parse(PDF, name='resume.pdf')
    assert 'PADDLEOCR_API_TOKEN' in str(exc.value.__cause__)
