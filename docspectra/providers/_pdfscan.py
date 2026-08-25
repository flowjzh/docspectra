"""Shared plumbing behind the local pdf engines (pdf-inspect and anydoc):
the post-classify tail — two byte-side decline gates over pypdf (the
word-strip fingerprint and the content-stream ratio) plus one text-side
gate over the engine markdown (the table wall), the XObject image
collection, and the markdown clean-up — pure pypdf plus regex, so either
extra can host it without the other's engine."""

from __future__ import annotations

import io
import re

from pypdf import PdfReader

from docspectra.parser import (Declined, ImageAsset, MAX_ASSETS,
                               MAX_ASSET_BYTES, ParseResult)

RATIO_THRESHOLD = 200  # B/char; measured 4,968 vs clean max 82 (parser.md §1)
STRIP_W, STRIP_H = 80, 100  # pixel dims of word-shaped raster strips
TABLE_SHARE_MIN = 0.7  # of all text chars; judgment call — measured wall
# 0.99, corpus real tables a minority share (parser.md §1)
WALL_ROW_CHARS = 500  # chars in one row; the engine's own paragraph-wall
# constant, measured wall row ~1,300 (parser.md §1)
_MIME = {'/DCTDecode': 'image/jpeg', '/JPXDecode': 'image/jp2',
         '/CCITTFaxDecode': 'image/tiff'}

_U_TAG = re.compile(r'</?u>')
_BULLET = re.compile(r'^l ', re.MULTILINE)  # the engine renders bullets as 'l'


def scan(data: bytes, text: str, **meta) -> ParseResult | Declined:
    """The shared post-classify tail: the two pypdf gates and the
    text-side table wall, image collection, and the cleaned ParseResult.
    ``meta`` adds engine-specific diagnostics between ``pages`` and
    ``skipped_assets``."""
    reader = PdfReader(io.BytesIO(data))
    for page in reader.pages:
        if strip_hit(page):
            return Declined('fingerprint_gate')
    if ratio_hit(reader, len(text)) >= RATIO_THRESHOLD:
        return Declined('ratio_gate')
    if table_wall_hit(text):
        return Declined('table_wall_gate')
    images, skipped = collect_images(reader)
    return ParseResult(text=clean(text), images=images,
                       meta={'pages': len(reader.pages), **meta,
                             'skipped_assets': skipped})


def clean(markdown: str) -> str:
    return _BULLET.sub('• ', _U_TAG.sub('', markdown))


def table_wall_hit(text: str) -> bool:
    """One markdown table swallowing the page's flowing prose — the
    layout-band false positive of geometry-based table detection (a
    borderless template frame reconstructed as a grid, reading order
    scrambled). Both conditions matter: a document that merely carries
    a real table stays, and a page of genuine data cells (high table
    share, short rows) stays."""
    lines = [line for line in text.splitlines() if line.strip()]
    rows = [line for line in lines
            if line.startswith('|') and line.strip('|- ')]
    total = sum(map(len, lines))
    return (bool(rows)
            and sum(map(len, rows)) / total >= TABLE_SHARE_MIN
            and max(map(len, rows)) > WALL_ROW_CHARS)


def strip_hit(page) -> bool:
    """A wide flat image XObject — a rasterized text strip among a real
    text layer (the raster-strip class). Missing height counts as the threshold."""
    return any(obj.get('/Width', 0) >= STRIP_W and obj.get('/Height', STRIP_H) <= STRIP_H
               for _, obj in _iter_images(page))


def ratio_hit(reader: PdfReader, nchars: int) -> float:
    """Content-stream bytes per delivered text char; outlined glyphs
    (the outlined-glyph class) inflate the stream without any text to show for it."""
    nbytes = 0
    for page in reader.pages:
        if (contents := page.get_contents()) is not None:
            nbytes += len(contents.get_data())
    return nbytes / nchars if nchars else float('inf')


def collect_images(reader: PdfReader) -> tuple[list[ImageAsset], int]:
    images, skipped = [], 0
    for page_no, page in enumerate(reader.pages, 1):
        for key, obj in _iter_images(page):
            if len(images) >= MAX_ASSETS:
                skipped += 1
                continue
            try:
                data = obj.get_data()
            except Exception:
                skipped += 1
                continue
            if len(data) > MAX_ASSET_BYTES:
                skipped += 1
                continue
            # names are opaque per-engine keys: the engine emits no
            # image references, so nothing in the text reconciles to them
            images.append(ImageAsset(
                name=f'p{page_no}-{str(key).lstrip("/")}',
                mime=_MIME.get(str(obj.get('/Filter', '')), 'application/octet-stream'),
                data=data, source=f'pdf p{page_no} {key}'))
    return images, skipped


def _iter_images(page):
    """Yield (key, ImageObject) pairs from the page's own XObject dict —
    the measured walk behind the fingerprint gate (parser.md §1)."""
    resources = _deref(page.get('/Resources') or {})
    for key, ref in _deref(resources.get('/XObject') or {}).items():
        obj = _deref(ref)
        if obj.get('/Subtype') == '/Image':
            yield key, obj


def _deref(obj):
    return obj.get_object() if hasattr(obj, 'get_object') else obj
