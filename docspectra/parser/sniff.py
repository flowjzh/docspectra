"""Document-type sniffing: magic bytes first, extension fallback, no
dependencies. Bytes win on conflict for routing; a filename declaring a
binary extension is gated by :func:`validate_declared` before any
provider runs.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

_PDF = b'%PDF-'
_ZIP = b'PK\x03\x04'
_OLE2 = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'

_IMAGES = (
    (b'\xff\xd8\xff', 'jpg'),
    (b'\x89PNG\r\n\x1a\n', 'png'),
    (b'GIF87a', 'gif'),
    (b'GIF89a', 'gif'),
    (b'BM', 'bmp'),
    (b'II*\x00', 'tiff'),
    (b'MM\x00*', 'tiff'),
    (b'\x00\x00\x00\x0cjP  \r\n\x87\n', 'jp2'),
)

# extension aliases, normalized to the canonical type
_ALIASES = {'jpeg': 'jpg', 'tif': 'tiff', 'htm': 'html'}

# OLE2 magic is coarse — the legacy office extensions form one family
_FAMILY = {'doc': frozenset({'doc', 'ppt', 'xls'})}

# types whose declared extension must verify against content magic: the
# magic tables' codomain (zip parts + images + webp), family expanded
_BINARY_TYPES = ({'pdf', 'docx', 'xlsx', 'pptx', 'webp'}
                 | {t for _, t in _IMAGES} | set().union(*_FAMILY.values()))


def sniff(data: bytes, name: str | None = None) -> str | None:
    """Route bytes to a canonical document type. ``name`` feeds the
    extension fallback when magic is inconclusive; None when neither
    yields a type."""
    if data.startswith(_PDF):
        return 'pdf'
    if data.startswith(_ZIP):
        return _zip_type(data) or _extension_type(name)
    if data.startswith(_OLE2):
        return 'doc'  # coarse: doc/ppt/xls share the legacy chain
    if t := _image_type(data):
        return t
    return _extension_type(name)


def validate_declared(data: bytes, name: str | None = None) -> None:
    """Gate a declared binary extension against content magic: raise
    ValueError('file content is not a XXX') — XXX the extension's
    canonical type — when the bytes do not verify. Text and unknown
    extensions are never validated; the extension fallback owns them."""
    declared = _extension_type(name)
    if declared not in _BINARY_TYPES or _matches(data, declared):
        return
    raise ValueError(f'file content is not a {declared}')


def _matches(data: bytes, declared: str) -> bool:
    magic = sniff(data)  # no name: pure magic, extension fallback off
    return magic is not None and declared in _FAMILY.get(magic, {magic})


def _zip_type(data: bytes) -> str | None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for info in z.infolist():
                if info.filename.startswith('word/'):
                    return 'docx'
                if info.filename.startswith('xl/'):
                    return 'xlsx'
                if info.filename.startswith('ppt/'):
                    return 'pptx'
    except (zipfile.BadZipFile, OSError):
        return None
    return None


def _image_type(data: bytes) -> str | None:
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return 'webp'
    for magic, t in _IMAGES:
        if data.startswith(magic):
            return t
    return None


def _extension_type(name: str | None) -> str | None:
    ext = Path(name or '').suffix.lower().lstrip('.')
    return _ALIASES.get(ext, ext) or None
