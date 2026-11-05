"""The bce cloud engine over direct httpx — token → submit → poll →
markdown flow, product selection and options, envelope error mapping,
retries, LRU — the wire client faked, zero network."""

import base64
from urllib.parse import parse_qs

import httpx
import pytest

from docspectra.parser import ParseDeclinedError, parse

from tests.helpers import PDF

OAUTH_URL = 'https://aip.baidubce.com/oauth/2.0/token'
MARKDOWN_URL = 'https://result.local/doc.md'
STRUCTURED_URL = 'https://result.local/doc.json'
MARKDOWN = '# Test Candidate\n\n2021/03~ Present Director- Supply Chain'
TASK_URL = ('https://aip.baidubce.com/rest/2.0/brain/online/v2/'
            'unlimited-ocr-parser/task')
PADDLE_VL_URL = ('https://aip.baidubce.com/rest/2.0/brain/online/v2/'
                 'paddle-vl-parser/task')


def _structured(*pages: tuple) -> dict:
    """A structured-result payload from page layout lists — each
    argument is one page's ``(type, text)`` layout pairs."""
    return {'pages': [{'layouts': [
        {'type': t, 'text': x} for t, x in page]} for page in pages]}


class _FakeClient:
    """Scripted httpx.AsyncClient: one token GET, submits recorded,
    task states popped in order, then the markdown body (and the
    structured payload when one is installed). Signatures mirror the
    provider's calls, so a renamed kwarg fails loudly."""

    def __init__(self, markdown: str | None = MARKDOWN,
                 states: tuple = ('running', 'success'),
                 fail_submit: int | dict | None = None,
                 auth: dict | None = None,
                 structured: dict | str | None = None,
                 fail_structured: bool = False):
        self.markdown = markdown
        self.states = list(states)
        self.fail_submit = fail_submit
        self.auth = auth or {'access_token': 'tok-1', 'expires_in': 2592000}
        self.structured = structured
        self.fail_structured = fail_structured
        self.submits = []  # post kwargs per task call
        self.downloads = []  # result urls fetched, in order
        self.token_gets = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, *, params=None, timeout=None):
        if url == OAUTH_URL:
            self.token_gets += 1
            return httpx.Response(200, json=self.auth)
        if url == MARKDOWN_URL:
            self.downloads.append(url)
            return httpx.Response(200, text=self.markdown)
        if url == STRUCTURED_URL:
            self.downloads.append(url)
            if self.fail_structured:
                return httpx.Response(404, text='gone')
            if isinstance(self.structured, str):
                return httpx.Response(200, text=self.structured)
            return httpx.Response(200, json=self.structured)
        raise AssertionError(f'unexpected GET {url}')

    async def post(self, url, *, params=None, headers=None, data=None,
                   content=None, timeout=None):
        if url.endswith('/query'):
            state = self.states.pop(0) if self.states else 'success'
            body = {'task_id': data['task_id'], 'status': state,
                    'cost_page_num': 3}
            if state == 'failed':
                body['task_error'] = 'parse failed'
            if state == 'success' and self.markdown is not None:
                body['markdown_url'] = MARKDOWN_URL
                if self.structured is not None:
                    body['parse_result_url'] = STRUCTURED_URL
            return httpx.Response(200, json={
                'error_code': 0, 'error_msg': 'success',
                'result': body})
        self.submits.append({'url': url, 'params': params,
                             'headers': headers, 'content': content})
        if self.fail_submit is not None:
            payload = self.fail_submit
            self.fail_submit = None
            if isinstance(payload, int):
                return httpx.Response(payload, text='too fast')
            return httpx.Response(200, json=payload)
        return httpx.Response(200, json={
            'error_code': 0, 'error_msg': '',
            'result': {'task_id': 'task-1'}})


@pytest.fixture
def env(monkeypatch):
    """Test credentials, no ambient overrides, clean LRU and token
    memo; install_client fakes the wire."""
    monkeypatch.setenv('BCE_API_KEY', 'key-1')
    monkeypatch.setenv('BCE_SECRET_KEY', 'secret-1')
    monkeypatch.delenv('BCE_MODEL', raising=False)
    monkeypatch.delenv('BCE_TIMEOUT', raising=False)
    monkeypatch.delenv('BCE_INCLUDE_HEADER', raising=False)
    from docspectra.providers import bce
    bce._cache.clear()
    bce._token = None
    yield bce
    bce._cache.clear()
    bce._token = None


@pytest.fixture
def install_client(patch_provider_client):
    def install(fake):
        patch_provider_client('bce', fake)
    return install


@pytest.fixture
def bce_chain(pin_chain):
    pin_chain({'pdf': ['bce']})


async def test_submit_poll_download_flow(env, install_client, bce_chain):
    fake = _FakeClient()
    install_client(fake)
    result = await parse(PDF, name='resume.pdf')
    assert result.provider == 'bce'
    assert result.text == MARKDOWN
    assert result.meta['model'] == 'unlimited-ocr-parser'
    assert result.meta['job_id'] == 'task-1'
    assert result.meta['pages'] == 3
    assert result.images == []
    submit = fake.submits[0]
    assert submit['url'] == TASK_URL
    assert submit['params'] == {'access_token': 'tok-1'}
    assert submit['headers']['Content-Type'] == 'application/x-www-form-urlencoded'
    fields = parse_qs(submit['content'].decode())
    assert base64.b64decode(fields['file_data'][0]) == PDF
    assert fields['file_name'] == ['resume.pdf']
    assert fields['angle_adjust'] == ['False']
    # the oauth token rode every task call
    assert fake.token_gets == 1


async def test_env_override_swaps_the_product(env, install_client,
                                              bce_chain, monkeypatch):
    monkeypatch.setenv('BCE_MODEL', 'paddle-vl-parser')
    fake = _FakeClient()
    install_client(fake)
    result = await parse(PDF, name='resume.pdf')
    assert fake.submits[0]['url'] == PADDLE_VL_URL
    fields = parse_qs(fake.submits[0]['content'].decode())
    assert fields['recognize_seal'] == ['False']
    assert result.meta['model'] == 'paddle-vl-parser'


async def test_unknown_product_is_a_hard_error(env, install_client,
                                               bce_chain, monkeypatch):
    monkeypatch.setenv('BCE_MODEL', 'no-such-parser')
    install_client(_FakeClient())
    with pytest.raises(ParseDeclinedError, match='not a known product'):
        await parse(PDF, name='resume.pdf')


async def test_token_is_reused_across_calls(env, install_client, bce_chain):
    fake = _FakeClient()
    install_client(fake)
    first = await parse(PDF, name='a.pdf')
    env._cache.clear()  # force the second call through the wire
    second = await parse(PDF, name='b.pdf')
    assert fake.token_gets == 1  # the memo served call two
    assert second.text == first.text
    assert len(fake.submits) == 2


async def test_lru_makes_second_call_zero_requests(env, install_client,
                                                  bce_chain):
    fake = _FakeClient()
    install_client(fake)
    first = await parse(PDF, name='resume.pdf')
    second = await parse(PDF, name='resume.pdf')
    assert second.text == first.text
    assert second.meta['cache_hit'] is True
    assert 'job_id' not in second.meta  # per-call facade, not call 1's meta
    assert len(fake.submits) == 1


@pytest.mark.parametrize('fake_kwargs, match', [
    ({'auth': {'error': 'invalid_client'}}, 'access token'),
    ({'fail_submit': {'error_code': 6,
                      'error_msg': 'No permission to access data'}},
     'No permission'),
    ({'states': ['failed']}, 'parse failed'),
    ({'fail_submit': {'error_code': 0, 'error_msg': '', 'result': {}}},
     'no task id'),
    ({'markdown': None}, 'markdown url'),
], ids=['auth', 'no-permission', 'task-failure', 'no-task-id',
        'no-markdown-url'])
async def test_failures_are_hard_errors(env, install_client, bce_chain,
                                        fake_kwargs, match):
    install_client(_FakeClient(**fake_kwargs))
    with pytest.raises(ParseDeclinedError, match=match):
        await parse(PDF, name='resume.pdf')


@pytest.mark.parametrize('failure', [
    429, {'error_code': 18, 'error_msg': 'QPS limit reached'},
], ids=['http-429', 'qps-code'])
async def test_rate_limits_are_retried(env, install_client, bce_chain,
                                       failure):
    fake = _FakeClient(fail_submit=failure)
    install_client(fake)
    result = await parse(PDF, name='resume.pdf')
    assert len(fake.submits) == 2  # rejection, then success
    assert result.text == MARKDOWN


async def test_timeout_is_a_hard_error(env, install_client, bce_chain,
                                       monkeypatch):
    fake = _FakeClient(states=['pending'] * 99)
    install_client(fake)
    monkeypatch.setenv('BCE_TIMEOUT', '0')  # deadline passes after poll one
    with pytest.raises(ParseDeclinedError, match='timed out'):
        await parse(PDF, name='resume.pdf')


async def test_missing_credentials_are_a_hard_error(env, bce_chain,
                                                    monkeypatch):
    monkeypatch.delenv('BCE_API_KEY')
    with pytest.raises(ParseDeclinedError) as exc:
        await parse(PDF, name='resume.pdf')
    assert 'BCE_API_KEY' in str(exc.value.__cause__)


async def test_header_layouts_are_prepended_by_default(env, install_client,
                                                       bce_chain):
    # an identity block filed under header never reaches the markdown
    install_client(_FakeClient(structured=_structured(
        (('header', 'Header Candidate\nEmail: hc@example.com'),
         ('title', 'SUMMARY')),
        (('title', 'EXPERIENCE'),),
    )))
    result = await parse(PDF, name='resume.pdf')
    assert result.text == ('Header Candidate\nEmail: hc@example.com'
                           f'\n\n{MARKDOWN}')


async def test_no_repeat_collapses_a_running_header(env, install_client,
                                                    bce_chain):
    install_client(_FakeClient(structured=_structured(
        (('header', 'Internal Use Only'), ('title', 'SUMMARY')),
        (('header', 'Internal Use Only'), ('title', 'MORE')),
        (('header', 'Internal Use Only'),),
    )))
    result = await parse(PDF, name='resume.pdf')
    assert result.text == f'Internal Use Only\n\n{MARKDOWN}'


async def test_no_repeat_keeps_nonconsecutive_headers(env, install_client,
                                                      bce_chain):
    install_client(_FakeClient(structured=_structured(
        (('header', 'Chapter One'),),
        (('title', 'BODY'),),
        (('header', 'Chapter One'),),
    )))
    result = await parse(PDF, name='resume.pdf')
    assert result.text == f'Chapter One\n\nChapter One\n\n{MARKDOWN}'


async def test_all_keeps_every_occurrence(env, install_client, bce_chain,
                                          monkeypatch):
    monkeypatch.setenv('BCE_INCLUDE_HEADER', 'all')
    install_client(_FakeClient(structured=_structured(
        (('header', 'Internal Use Only'),),
        (('header', 'Internal Use Only'),),
    )))
    result = await parse(PDF, name='resume.pdf')
    assert result.text == ('Internal Use Only\n\nInternal Use Only'
                           f'\n\n{MARKDOWN}')


async def test_none_never_downloads_the_structured_result(
        env, install_client, bce_chain, monkeypatch):
    monkeypatch.setenv('BCE_INCLUDE_HEADER', 'none')
    fake = _FakeClient(structured=_structured(
        (('header', 'Header Candidate'),)))
    install_client(fake)
    result = await parse(PDF, name='resume.pdf')
    assert result.text == MARKDOWN
    assert fake.downloads == [MARKDOWN_URL]
    assert result.meta['model'] == 'unlimited-ocr-parser:none'


async def test_structured_download_failure_degrades_to_markdown(
        env, install_client, bce_chain):
    install_client(_FakeClient(structured={}, fail_structured=True))
    result = await parse(PDF, name='resume.pdf')
    assert result.text == MARKDOWN


async def test_malformed_structured_payload_degrades_to_markdown(
        env, install_client, bce_chain):
    install_client(_FakeClient(structured='not json'))
    result = await parse(PDF, name='resume.pdf')
    assert result.text == MARKDOWN


async def test_unknown_header_mode_is_a_hard_error(env, install_client,
                                                   bce_chain, monkeypatch):
    monkeypatch.setenv('BCE_INCLUDE_HEADER', 'some')
    install_client(_FakeClient())
    with pytest.raises(ParseDeclinedError) as exc:
        await parse(PDF, name='resume.pdf')
    assert 'is not a mode' in str(exc.value.__cause__)


async def test_a_mode_flip_does_not_serve_the_cached_other_mode(
        env, install_client, bce_chain, monkeypatch):
    fake = _FakeClient(structured=_structured(
        (('header', 'Header Candidate'),)))
    install_client(fake)
    first = await parse(PDF, name='resume.pdf')
    assert first.text == f'Header Candidate\n\n{MARKDOWN}'
    monkeypatch.setenv('BCE_INCLUDE_HEADER', 'none')
    second = await parse(PDF, name='resume.pdf')
    assert second.text == MARKDOWN  # fresh key, not first's cached text
    assert len(fake.submits) == 2  # the flip went to the wire
    assert 'cache_hit' not in second.meta
