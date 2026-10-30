"""The paddleocr cloud engine over direct httpx — submit→poll→result
flow, input-type model split, ocr-line geometry order, watermark
stamps, markdown_ignore_labels knob, rate lanes, retry, LRU — the wire
client faked, zero network."""

import json
import math

import httpx
import pytest

from docspectra.parser import ParseDeclinedError, parse
from docspectra.providers.paddleocr import (_drop_watermarks,
                                            _options as paddle_options,
                                            _reading_order)

from tests.helpers import PDF, PNG

JSONL_URL = 'https://result.local/out.jsonl'
ASSET_URL = 'https://cdn.example/x.png'


def _line(markdown: str | None = None, images: dict | None = None,
          recs: list | None = None, boxes: list | None = None,
          polys: list | None = None) -> dict:
    """One jsonl line — an ocrResults entry for recs, a
    layoutParsingResults entry for markdown/images."""
    if recs is not None:
        pruned = {'rec_texts': recs}
        if boxes is not None:
            pruned['rec_boxes'] = boxes
        if polys is not None:
            pruned['rec_polys'] = polys
        return {'result': {'ocrResults': [{'prunedResult': pruned}]}}
    return {'result': {'layoutParsingResults': [
        {'markdown': {'text': markdown, 'images': images or {}}},
    ]}}


class _FakeStream:
    """The stream contract ``_asset`` consumes, backed by one fake GET:
    status/headers surface before the (single-chunk) body."""

    def __init__(self, fake, url):
        self._fake = fake
        self._url = url
        self.status_code = None
        self.headers = {}

    async def __aenter__(self):
        res = await self._fake.get(self._url)
        self.status_code = res.status_code
        self.headers = res.headers
        self._content = res.content
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_bytes(self):
        yield self._content


class _FakeClient:
    """Scripted httpx.AsyncClient: submits recorded, status states
    popped in order, then the jsonl line body. Asset URLs serve PNG or
    the configured status. Asset fetch (``_asset``) needs no patch —
    the fake answers the CDN too. Signatures mirror the provider's
    calls, so a renamed kwarg fails loudly instead of vanishing."""

    def __init__(self, lines: list, states: list = ('running', 'done'),
                 fail_submit: tuple | None = None, asset_status: int = 200):
        self.lines = lines
        self.states = list(states)
        self.fail_submit = fail_submit
        self.asset_status = asset_status
        self.submits = []  # post kwargs: headers, data, files

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, *, headers=None, data=None, files=None, timeout=None):
        self.submits.append({'headers': headers, 'data': data, 'files': files})
        if self.fail_submit is not None:
            status, body = self.fail_submit
            self.fail_submit = None
            return httpx.Response(status, text=body)
        return httpx.Response(200, json={'code': 0, 'data': {'jobId': 'j1'}})

    def stream(self, method, url, *, headers=None, timeout=None):
        return _FakeStream(self, url)

    async def get(self, url, *, headers=None, timeout=None):
        if url == JSONL_URL:
            return httpx.Response(200, text='\n'.join(
                json.dumps(line) for line in self.lines))
        if url != ASSET_URL:  # the job status endpoint
            state = self.states.pop(0) if self.states else 'done'
            body = {'state': state}
            if state == 'failed':
                body['errorMsg'] = 'cannot parse document'
            if state == 'done':
                body['resultUrl'] = {'jsonUrl': JSONL_URL}
            return httpx.Response(200, json={'code': 0, 'data': body})
        if self.asset_status >= 400:
            return httpx.Response(self.asset_status)
        return httpx.Response(200, content=PNG,
                              headers={'Content-Length': str(len(PNG))})


@pytest.fixture
def install_client(patch_provider_client):
    def install(fake):
        patch_provider_client('paddleocr', fake)
    return install


@pytest.fixture
def env(monkeypatch):
    """Test token, no ambient model overrides, clean LRU;
    install_client fakes the wire."""
    monkeypatch.setenv('PADDLEOCR_API_TOKEN', 'test-token')
    monkeypatch.delenv('PADDLEOCR_MODEL', raising=False)
    monkeypatch.delenv('PADDLEOCR_IMAGE_MODEL', raising=False)
    from docspectra.providers import paddleocr
    paddleocr._cache.clear()
    yield paddleocr
    paddleocr._cache.clear()


@pytest.fixture
def paddle_chain(pin_chain):
    pin_chain({'pdf': ['paddleocr'], 'png': ['paddleocr']})


async def test_submit_poll_wait_flow(env, install_client, paddle_chain):
    fake = _FakeClient([_line(recs=['姓名 张三', '工作经历'])])
    install_client(fake)
    result = await parse(PNG, name='resume.png')
    assert result.provider == 'paddleocr'
    assert result.text == '姓名 张三\n工作经历'
    assert result.meta['job_id'] == 'j1'
    assert result.meta['model'] == 'PP-OCRv6'
    post = fake.submits[0]
    assert post['data']['model'] == 'PP-OCRv6'
    assert json.loads(post['data']['optionalPayload']) == paddle_options('PP-OCRv6')
    assert post['files']['file'][0] == 'resume.png'
    assert post['files']['file'][1] == PNG
    assert result.images == []


async def test_pdf_defaults_to_the_doc_model(env, install_client, paddle_chain):
    fake = _FakeClient([_line('## name\n\nbody')])
    install_client(fake)
    result = await parse(PDF, name='resume.pdf')
    post = fake.submits[0]
    assert post['data']['model'] == 'PP-StructureV3'
    assert result.meta['model'] == 'PP-StructureV3'
    assert json.loads(post['data']['optionalPayload']) == paddle_options('PP-StructureV3')


async def test_document_flow_wants_the_full_markdown(env, install_client,
                                                     paddle_chain):
    fake = _FakeClient([_line('## name\n\nbody', images={'imgs/a.png': ASSET_URL})])
    install_client(fake)
    result = await parse(PDF, name='resume.pdf')
    assert '## name' in result.text
    assert len(result.images) == 1
    asset = result.images[0]
    assert asset.name == 'imgs/a.png'
    assert asset.mime == 'image/png'
    assert asset.data == PNG
    assert asset.source == 'paddleocr imgs/a.png'


def test_options_per_model():
    assert paddle_options('PP-StructureV3') == {
        'useDocOrientationClassify': False, 'useDocUnwarping': False,
        'markdownIgnoreLabels': [], 'useChartRecognition': False}
    assert paddle_options('PaddleOCR-VL-1.6') == paddle_options('PP-StructureV3')
    assert paddle_options('PP-OCRv6') == {
        'useDocOrientationClassify': False, 'useDocUnwarping': False,
        'useTextlineOrientation': False, 'textDetLimitSideLen': 4000,
        'textDetLimitType': 'max'}


async def test_pages_join_in_order(env, install_client, paddle_chain):
    fake = _FakeClient([_line('page one'), _line('page two')])
    install_client(fake)
    result = await parse(PDF, name='resume.pdf')
    assert result.text == 'page one\n\npage two'


async def test_asset_failure_is_skipped(env, install_client, paddle_chain):
    fake = _FakeClient([_line('body', images={'imgs/a.png': ASSET_URL})],
                       asset_status=500)
    install_client(fake)
    result = await parse(PDF, name='resume.pdf')
    assert 'body' in result.text
    assert result.images == []
    assert result.meta['skipped_assets'] == 1


async def test_quota_rejection_is_a_hard_error(env, install_client, paddle_chain):
    fake = _FakeClient([], fail_submit=(402, 'quota exhausted'))
    install_client(fake)
    with pytest.raises(ParseDeclinedError) as exc:
        await parse(PDF, name='resume.pdf')
    assert exc.value.attempts[0].outcome == 'error'
    assert 'quota exhausted' in str(exc.value.__cause__)


async def test_rate_limit_is_retried_then_succeeds(env, install_client, paddle_chain):
    fake = _FakeClient([_line(recs=['recovered text'])],
                       fail_submit=(429, 'slow down'))
    install_client(fake)
    result = await parse(PNG, name='resume.png')
    assert result.provider == 'paddleocr'
    assert len(fake.submits) == 2  # 429 then success
    assert 'recovered text' in result.text


async def test_job_failure_is_a_hard_error(env, install_client, paddle_chain):
    fake = _FakeClient([_line('')], states=['failed'])
    install_client(fake)
    with pytest.raises(ParseDeclinedError) as exc:
        await parse(PDF, name='resume.pdf')
    assert 'cannot parse document' in str(exc.value.__cause__)


async def test_timeout_is_a_hard_error(env, install_client, paddle_chain, monkeypatch):
    fake = _FakeClient([_line('')], states=['running'] * 99)
    install_client(fake)
    monkeypatch.setenv('PADDLEOCR_TIMEOUT', '0')  # deadline passed right after the first poll
    with pytest.raises(ParseDeclinedError, match='timed out'):
        await parse(PDF, name='resume.pdf')


async def test_lru_makes_second_call_zero_requests(env, install_client, paddle_chain):
    fake = _FakeClient([_line(recs=['recovered text'])])
    install_client(fake)
    first = await parse(PNG, name='resume.png')
    second = await parse(PNG, name='resume.png')
    assert second.text == first.text
    assert second.meta['cache_hit'] is True
    assert 'job_id' not in second.meta  # per-call facade, not call 1's meta
    assert len(fake.submits) == 1


def test_reading_order_splits_column_streams():
    # a two-column page in scrambled detection order: the gutter splits
    # the streams, each reads top-down, left column first
    boxes = [[430, 300, 1100, 340], [40, 400, 360, 440],
             [40, 100, 360, 140], [430, 100, 1100, 140]]
    assert _reading_order(boxes, ['R2', 'L2', 'L1', 'R1']) == ['L1', 'L2', 'R1', 'R2']


def test_reading_order_falls_back_to_y_order():
    # a full-width box crosses every mid-page x-run, so no gutter
    # qualifies and the geometric order decides
    boxes = [[100, 300, 1100, 340], [40, 100, 360, 140], [430, 200, 1100, 240]]
    assert _reading_order(boxes, ['c', 'a', 'b']) == ['a', 'b', 'c']


def test_reading_order_without_boxes_keeps_detection_order():
    assert _reading_order([], ['b', 'a', '']) == ['b', 'a']


async def test_model_env_overrides_route_both_ways(env, install_client,
                                                   paddle_chain, monkeypatch):
    monkeypatch.setenv('PADDLEOCR_MODEL', 'PP-OCRv6')
    fake = _FakeClient([_line(recs=['scanned pdf line'])])
    install_client(fake)
    result = await parse(PDF, name='resume.pdf')
    assert fake.submits[0]['data']['model'] == 'PP-OCRv6'
    assert result.text == 'scanned pdf line'

    monkeypatch.setenv('PADDLEOCR_IMAGE_MODEL', 'PP-StructureV3')
    fake = _FakeClient([_line('## scanned image')])
    install_client(fake)
    result = await parse(PNG, name='resume.png')
    assert fake.submits[0]['data']['model'] == 'PP-StructureV3'
    assert result.text == '## scanned image'


async def test_ocr_lines_read_in_geometry_order(env, install_client, paddle_chain):
    boxes = [[430, 100, 1100, 140], [40, 100, 360, 140]]
    fake = _FakeClient([_line(recs=['R1', 'L1'], boxes=boxes)])
    install_client(fake)
    result = await parse(PNG, name='resume.png')
    assert result.text == 'L1\nR1'


def _poly(x: float, y: float, deg: float) -> list:
    """One text line's quad: a 200×30 line whose top edge runs from
    (x, y) at `deg` — the rec_polys wire shape."""
    dx, dy = 200 * math.cos(math.radians(deg)), 200 * math.sin(math.radians(deg))
    nx, ny = -30 * math.sin(math.radians(deg)), 30 * math.cos(math.radians(deg))
    return [[x, y], [x + dx, y + dy], [x + dx + nx, y + dy + ny], [x + nx, y + ny]]


def test_watermark_stamps_are_dropped():
    # the calibration bands: body text under 2°, site stamps ~22-27° — a
    # diagonal line is a stamp whether or not it crosses a body line
    boxes = [[100, 100, 300, 130], [40, 200, 281, 285], [300, 300, 500, 330]]
    polys = [_poly(100, 100, 0), _poly(40, 200, 25), _poly(300, 300, 0)]
    assert _drop_watermarks(boxes, polys, ['body', 'stamp', 'tail']) == (
        [[100, 100, 300, 130], [300, 300, 500, 330]], ['body', 'tail'])


def test_vertical_tabs_and_slight_skew_survive():
    boxes = [[30, 100, 60, 300], [100, 500, 300, 530]]
    polys = [_poly(30, 100, 90), _poly(100, 500, 3)]
    assert _drop_watermarks(boxes, polys, ['tab', 'body']) == (
        boxes, ['tab', 'body'])


def test_without_polys_nothing_is_dropped():
    boxes, texts = [[0, 0, 10, 10]], ['a']
    assert _drop_watermarks(boxes, [], texts) == (boxes, texts)


async def test_stamp_lines_never_reach_the_text(env, install_client, paddle_chain):
    fake = _FakeClient([_line(
        recs=['正文一行', 'Watermark stamp'],
        boxes=[[40, 100, 360, 140], [430, 300, 708, 390]],
        polys=[_poly(40, 100, 0), _poly(430, 300, 25)])])
    install_client(fake)
    result = await parse(PNG, name='resume.png')
    assert result.text == '正文一行'


async def test_missing_token_is_a_hard_error(env, paddle_chain, monkeypatch):
    monkeypatch.delenv('PADDLEOCR_API_TOKEN')
    with pytest.raises(ParseDeclinedError) as exc:
        await parse(PDF, name='resume.pdf')
    assert 'PADDLEOCR_API_TOKEN' in str(exc.value.__cause__)
