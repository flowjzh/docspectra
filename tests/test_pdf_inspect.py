"""S13: the pdf-inspect engine — gates, post-processing, image assets.

Synthetic PDFs exercise what is synthesizable; the three gate routing
proofs (fingerprint, ratio, classify) assert in the S16 real-chain
eval."""

import io

import pytest
from pypdf import PdfReader

from docspectra.parser import ParseDeclinedError, configure, parse
from docspectra.providers.pdfscan import clean, ratio_hit, strip_hit
from tests.pdf_builder import make_pdf, raster_pdf, strip_pdf

CLEAN = make_pdf()
RASTER = raster_pdf()
STRIP = strip_pdf()
HEADSHOT = make_pdf(image=(80, 400), image_rect=(250, 600, 60, 120))
TINY_IMAGE = make_pdf(image=(6, 6), image_rect=(250, 600, 12, 12))
RATIO = make_pdf(text_lines=1, path_ops=400)


@pytest.fixture
def pdf_chain():
    configure({'pdf': ['pdf-inspect']})
    yield
    configure(None)


async def test_clean_text_pdf_parses(pdf_chain):
    result = await parse(CLEAN, name='resume.pdf')
    assert result.provider == 'pdf-inspect'
    assert 'sample line 001' in result.text
    assert result.meta['pages'] == 1
    assert result.meta['pdf_type'] == 'text_based'
    assert result.meta['encoding_issues'] is False


async def test_raster_page_declines_classify_gate(pdf_chain):
    with pytest.raises(ParseDeclinedError) as exc:
        await parse(RASTER, name='resume.pdf')
    assert exc.value.attempts[0].outcome == 'declined'
    assert exc.value.attempts[0].reason == 'classify_gate'


async def test_word_strip_declines_fingerprint_gate(pdf_chain):
    with pytest.raises(ParseDeclinedError) as exc:
        await parse(STRIP, name='resume.pdf')
    assert exc.value.attempts[0].reason == 'fingerprint_gate'


async def test_fingerprint_dimensions(pdf_chain):
    reader = PdfReader(io.BytesIO(STRIP))
    assert strip_hit(reader.pages[0])
    reader = PdfReader(io.BytesIO(HEADSHOT))  # tall portrait image passes
    assert not strip_hit(reader.pages[0])
    reader = PdfReader(io.BytesIO(CLEAN))
    assert not strip_hit(reader.pages[0])


def test_ratio_gate_direct():
    ratio_text = len('sample line 001 of text')  # the single text line
    assert ratio_hit(PdfReader(io.BytesIO(RATIO)), ratio_text) >= 200
    clean_text = len('sample line 001 of text') * 25
    assert ratio_hit(PdfReader(io.BytesIO(CLEAN)), clean_text) < 200


def test_ratio_merges_multi_part_content_stream():
    pdf = make_pdf(text_lines=1, path_ops=400, stream_parts=2)
    nchars = len('sample line 001 of text')
    assert ratio_hit(PdfReader(io.BytesIO(pdf)), nchars) >= 200  # both parts counted


def test_post_processing_strips_u_and_restores_bullets():
    markdown = '## <u>技能特长</u>\nl Manage the office\nl Lead the team\nplain line'
    assert clean(markdown) == '## 技能特长\n• Manage the office\n• Lead the team\nplain line'
    assert clean('no tags here') == 'no tags here'


async def test_small_image_carried_as_asset(pdf_chain):
    result = await parse(TINY_IMAGE, name='resume.pdf')
    assert len(result.images) == 1
    asset = result.images[0]
    assert asset.name == 'p1-Im1'
    assert asset.mime == 'application/octet-stream'  # flate raw pixels
    assert asset.source == 'pdf p1 /Im1'
    assert len(asset.data) == 6 * 6 * 3
    assert result.meta['skipped_assets'] == 0
