"""Chain scheduling: decline fall-through, exhaustion, config validation,
routing, name normalization — fake providers, zero network."""

import json

import pytest

from docspectra import parser as core
from docspectra.parser import ParseDeclinedError, configure, parse
from docspectra.parser.core import _chain_for
from tests.fake_providers import BEHAVIORS
from tests.helpers import OLE2, PDF, zip_with

TXT = '姓名：张三'.encode('utf-8')


@pytest.fixture
def fakes():
    BEHAVIORS.clear()
    configure(None)  # unpin from any earlier test
    yield BEHAVIORS
    BEHAVIORS.clear()
    configure(None)


async def test_chain_falls_through_on_decline(fakes):
    fakes['fake-a'] = 'decline'
    configure({'pdf': ['fake-a', 'fake-b']})
    result = await parse(PDF, name='resume.pdf')
    assert result.provider == 'fake-b'
    assert result.text.startswith('fake-b:')
    assert [(a.provider, a.outcome, a.reason) for a in result.attempts] == [
        ('fake-a', 'declined', 'fake-a-gate'),
        ('fake-b', 'success', None),
    ]


async def test_exhaustion_records_every_attempt(fakes):
    configure({'pdf': ['fake-a', 'fake-b']})
    fakes['fake-a'] = 'decline'
    fakes['fake-b'] = 'decline'
    with pytest.raises(ParseDeclinedError) as exc:
        await parse(PDF, name='resume.pdf')
    assert [(a.provider, a.outcome, a.reason) for a in exc.value.attempts] == [
        ('fake-a', 'declined', 'fake-a-gate'),
        ('fake-b', 'declined', 'fake-b-gate'),
    ]
    assert 'fake-a' in str(exc.value) and 'fake-b' in str(exc.value)


async def test_hard_error_falls_through(fakes):
    fakes['fake-a'] = 'raise'
    configure({'pdf': ['fake-a', 'fake-b']})
    result = await parse(PDF, name='resume.pdf')
    assert result.provider == 'fake-b'
    assert result.attempts[0].outcome == 'error'
    assert 'RuntimeError' in result.attempts[0].reason


async def test_exhaustion_after_errors_chains_the_cause(fakes):
    fakes['fake-a'] = 'raise'
    configure({'pdf': ['fake-a']})
    with pytest.raises(ParseDeclinedError) as exc:
        await parse(PDF, name='resume.pdf')
    assert isinstance(exc.value.__cause__, RuntimeError)
    assert exc.value.attempts[0].outcome == 'error'


async def test_configured_but_not_installed_is_an_error(monkeypatch):
    """A configured provider the sweep skipped (extra absent) errors at
    the startup hook — simulated by unregistering mineru."""
    monkeypatch.setenv('DOCSPECTRA_PARSER_CONFIG', json.dumps({'pdf': ['mineru']}))
    monkeypatch.delitem(core.core._providers, 'mineru', raising=False)
    with pytest.raises(ValueError, match='mineru.*not installed'):
        configure(None)  # fail-fast at the startup hook


async def test_unknown_provider_name_is_an_error(monkeypatch):
    monkeypatch.setenv('DOCSPECTRA_PARSER_CONFIG', json.dumps({'pdf': ['bogus']}))
    with pytest.raises(ValueError, match='bogus.*known'):
        configure(None)


async def test_malformed_config_is_an_error(monkeypatch):
    monkeypatch.setenv('DOCSPECTRA_PARSER_CONFIG', 'not json')
    with pytest.raises(ValueError, match='DOCSPECTRA_PARSER_CONFIG'):
        configure(None)
    monkeypatch.setenv('DOCSPECTRA_PARSER_CONFIG', json.dumps({'pdf': 'fake-a'}))
    with pytest.raises(ValueError, match='list of engine names'):
        configure(None)


async def test_hyphen_underscore_normalization(fakes):
    fakes['fake-a'] = 'decline'
    fakes['fake-b'] = 'decline'
    configure({'pdf': ['fake_a', 'fake-b', 'fake_c']})  # underscore & hyphen mix
    result = await parse(PDF, name='resume.pdf')
    assert result.provider == 'fake-c'  # underscore-declared engine via hyphen config
    assert [a.provider for a in result.attempts] == ['fake-a', 'fake-b', 'fake-c']


async def test_binary_extension_mismatch_raises_before_chain(fakes):
    """The declared-binary gate fires before the chain even when the magic
    is conclusive — the declaration wins the diagnosis, not the content."""
    configure({'pdf': ['fake-a'], 'docx': ['fake-b']})
    with pytest.raises(ValueError, match='file content is not a docx'):
        await parse(PDF, name='resume.docx')  # pdf magic, docx name
    with pytest.raises(ValueError, match='file content is not a pdf'):
        await parse(TXT, name='resume.pdf')  # no magic → declared gate


async def test_binary_content_with_text_name_routes_by_magic(fakes):
    configure({'pdf': ['fake-a']})
    result = await parse(PDF, name='resume.txt')  # text names are never gated
    assert result.provider == 'fake-a'


async def test_extension_alias_fallback(fakes):
    configure({'txt': ['txt'], 'jpg': ['fake_c']})
    result = await parse(b'\xff\xd8\xff\xe0', name='photo.jpeg')  # jpeg → jpg
    assert result.provider == 'fake-c'


async def test_derived_default_chain_in_registration_order(fakes):
    fakes['fake-a'] = 'decline'
    fakes['fake-b'] = 'decline'
    result = await parse(PDF, name='resume.pdf')  # unconfigured → derived [fake-a, fake-b, fake-c]
    assert result.provider == 'fake-c'


async def test_builtin_pdf_chain_ranks_paddleocr_before_mineru(fakes):
    """Both extras share deps, so import success can't rank them — the
    builtin order must keep paddleocr ahead of mineru or a paddleocr-only
    install silently routes scanned pdfs to mineru."""
    chain = _chain_for('pdf')
    assert chain.index('paddleocr') < chain.index('mineru')


async def test_zip_routes_docx_xlsx_pptx(fakes):
    configure({'docx': ['fake-a'], 'xlsx': ['fake-b'], 'pptx': ['fake_c']})
    assert (await parse(zip_with('word/document.xml'), name='f.docx')).provider == 'fake-a'
    assert (await parse(zip_with('xl/workbook.xml'), name='f.xlsx')).provider == 'fake-b'
    assert (await parse(zip_with('ppt/presentation.xml'), name='f.pptx')).provider == 'fake-c'
    # plain zip: no office directory → the declared-binary gate rejects it
    with pytest.raises(ValueError, match='file content is not a docx'):
        await parse(zip_with('data.bin'), name='f.docx')


async def test_ole2_routes_to_legacy_doc_chain(fakes):
    configure({'doc': ['fake-b']})
    result = await parse(OLE2, name='resume.xls')
    assert result.provider == 'fake-b'  # magic wins over the xls extension


async def test_txt_passthrough():
    result = await parse(TXT, name='resume.txt')
    assert result.text == '姓名：张三'
    assert result.provider == 'txt'
    assert result.format == 'markdown'
    assert result.images == []
    assert result.attempts == [core.Attempt('txt', 'success', None)]
    assert (await parse('# 标题'.encode(), name='resume.md')).provider == 'txt'


async def test_binary_as_txt_declines():
    with pytest.raises(ParseDeclinedError, match='utf-8'):
        await parse(b'\x00\xff\xfe', name='f.txt')


async def test_unroutable_input_raises():
    with pytest.raises(ValueError, match='cannot determine'):
        await parse(b'\x00\x01\x02\x03')
    with pytest.raises(ValueError, match='pdfx'):
        await parse(b'x', name='f.pdfx')


async def test_parse_path_convenience(tmp_path):
    file = tmp_path / 'resume.txt'
    file.write_bytes(TXT)
    result = await core.parse_path(file)
    assert result.provider == 'txt' and result.text == '姓名：张三'
