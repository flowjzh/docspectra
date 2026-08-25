"""Magic-byte sniffing and extension fallback."""

import pytest

from docspectra.parser import sniff, validate_declared
from tests.helpers import OLE2, PDF, zip_with


@pytest.mark.parametrize('data, expected', [
    (b'\xff\xd8\xff\xe0', 'jpg'),
    (b'\x89PNG\r\n\x1a\n', 'png'),
    (b'GIF87a\x00', 'gif'),
    (b'GIF89a\x00', 'gif'),
    (b'BM\x00\x00\x00\x00', 'bmp'),
    (b'II*\x00\x08', 'tiff'),
    (b'MM\x00*\x00', 'tiff'),
    (b'RIFF\x00\x00\x00\x00WEBP', 'webp'),
    (b'\x00\x00\x00\x0cjP  \r\n\x87\n', 'jp2'),
])
def test_image_signatures(data: bytes, expected: str):
    assert sniff(data) == expected


def test_pdf_and_ole2_magic():
    assert sniff(PDF) == 'pdf'
    assert sniff(OLE2) == 'doc'  # coarse legacy chain


def test_zip_office_types():
    assert sniff(zip_with('word/document.xml')) == 'docx'
    assert sniff(zip_with('xl/workbook.xml')) == 'xlsx'
    assert sniff(zip_with('ppt/presentation.xml')) == 'pptx'
    assert sniff(zip_with('word/a.xml', 'ppt/b.xml')) == 'docx'  # first match wins


def test_plain_zip_falls_back_to_extension():
    assert sniff(zip_with('data.bin'), name='a.docx') == 'docx'
    assert sniff(zip_with('data.bin')) is None


def test_extension_fallback_and_alias_normalization():
    assert sniff(b'no magic', name='photo.jpeg') == 'jpg'
    assert sniff(b'no magic', name='photo.tif') == 'tiff'
    assert sniff(b'no magic', name='page.htm') == 'html'
    assert sniff(b'no magic', name='resume.txt') == 'txt'
    assert sniff(b'no magic', name='resume.markdown') == 'markdown'
    assert sniff(b'no magic', name='resume') is None
    assert sniff(b'no magic') is None


def test_extension_is_case_insensitive():
    assert sniff(b'no magic', name='RESUME.PDF') == 'pdf'


def test_bytes_win_on_conflict():
    assert sniff(PDF, name='a.docx') == 'pdf'
    assert sniff(OLE2, name='a.xlsx') == 'doc'
    assert sniff(b'\xff\xd8\xff', name='a.png') == 'jpg'


@pytest.mark.parametrize('data, name, declared', [
    (b'plain text', 'resume.pdf', 'pdf'),
    (PDF, 'a.docx', 'docx'),
    (OLE2, 'a.xlsx', 'xlsx'),
    (zip_with('word/document.xml'), 'a.doc', 'doc'),
    (zip_with('ppt/presentation.xml'), 'f.docx', 'docx'),
    (zip_with('data.bin'), 'a.docx', 'docx'),
    (b'PK\x03\x04' + b'\x00' * 64, 'a.docx', 'docx'),
    (b'\xff\xd8\xff\xe0', 'photo.png', 'png'),
    (b'no magic', 'photo.jpeg', 'jpg'),
    (b'no magic', 'scan.tif', 'tiff'),
])
def test_declared_binary_mismatch_raises(data: bytes, name: str, declared: str):
    with pytest.raises(ValueError, match=f'file content is not a {declared}'):
        validate_declared(data, name)


def test_declared_binary_valid_content_passes():
    assert validate_declared(PDF, 'resume.pdf') is None
    assert validate_declared(PDF, 'RESUME.PDF') is None  # case-insensitive
    assert validate_declared(zip_with('xl/workbook.xml'), 'a.xlsx') is None
    assert validate_declared(b'\x89PNG\r\n\x1a\n', 'p.png') is None


def test_ole2_family_validates_doc_ppt_xls():
    # the magic is coarse — any OLE2 content validates all three legacy names
    assert all(validate_declared(OLE2, f'a.{ext}') is None
               for ext in ('doc', 'ppt', 'xls'))


def test_alias_extensions_validate_the_canonical_type():
    assert validate_declared(b'\xff\xd8\xff\xe0', 'photo.jpeg') is None
    assert validate_declared(b'II*\x00\x08', 'scan.tif') is None


@pytest.mark.parametrize('name', ['resume.txt', 'a.md', 'a.markdown', 'page.htm',
                                  'backup.zip', 'f.pdfx', 'resume', None])
def test_text_and_unknown_extensions_are_not_validated(name: str | None):
    assert validate_declared(b'anything', name) is None


def test_valid_binary_with_text_extension_not_validated():
    # bytes-win direction preserved: no gate for text names
    assert validate_declared(PDF, 'resume.txt') is None
