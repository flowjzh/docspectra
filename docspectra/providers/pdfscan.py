"""Shared plumbing behind the local pdf engines (pdf-inspect and anydoc):
the ``scan()`` tail runs the decline gates over pypdf (the word-strip
fingerprint and the content-stream ratio, both byte-side; the emission
order against the layout-mode line order, text-side), the shared wall
gate (predicate in ``_markdown``, engine markdown not pypdf), the
XObject image collection, and the markdown clean-up — pypdf plus regex
so either extra can host it without the other's engine."""

from __future__ import annotations

import io
import re

from pypdf import PdfReader

from docspectra.parser import (Declined, ImageAsset, MAX_ASSETS,
                               MAX_ASSET_BYTES, ParseResult)
from docspectra.providers._markdown import table_wall_hit

RATIO_THRESHOLD = 200  # B/char; measured 4,968 vs clean max 82 (parser.md §1)
STRIP_W, STRIP_H = 80, 100  # pixel dims of word-shaped raster strips
# reference-line ranks between consecutive emissions
LAYOUT_LINE_JUMP = 4
# measured 57% of pairs scrambled vs ≤11% clean, two-column info grids
# included (parser.md §1)
LAYOUT_JUMP_FRAC = 0.30
# fewer aligned pairs and the gate has no opinion — a compensating
# control never declines on evidence it does not have
LAYOUT_MIN_PAIRS = 8
_MIME = {'/DCTDecode': 'image/jpeg', '/JPXDecode': 'image/jp2',
         '/CCITTFaxDecode': 'image/tiff'}

_U_TAG = re.compile(r'</?u>')
_BULLET = re.compile(r'^l ', re.MULTILINE)  # the engine renders bullets as 'l'
_ROW_SEP = re.compile(r'^\|[\s\-|]*\|$')


def scan(data: bytes, text: str, **meta) -> ParseResult | Declined:
    """The shared post-classify tail: the decline gates (fingerprint,
    ratio, table wall, layout), image collection, and the cleaned
    ParseResult. ``meta`` adds engine-specific diagnostics between
    ``pages`` and ``skipped_assets``."""
    reader = PdfReader(io.BytesIO(data))
    for page in reader.pages:
        if strip_hit(page):
            return Declined('fingerprint_gate')
    if ratio_hit(reader, len(text)) >= RATIO_THRESHOLD:
        return Declined('ratio_gate')
    if table_wall_hit(text):
        return Declined('table_wall_gate')
    if layout_hit(reader, text):
        return Declined('layout_gate')
    images, skipped = collect_images(reader)
    return ParseResult(text=clean(text), images=images,
                       meta={'pages': len(reader.pages), **meta,
                             'skipped_assets': skipped})


def clean(markdown: str) -> str:
    return _BULLET.sub('• ', _U_TAG.sub('', markdown))


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


def layout_hit(reader: PdfReader, text: str) -> bool:
    """The scrambled-layout class: a clean text layer read
    text-complete but with wide justified lines emitted after the short
    ones, dissociating entries from their dates and bullets. Emission
    units (paragraphs, table rows) align to the layout-mode line order
    by longest containment; the travel fraction decides, and thin
    evidence passes."""
    units = _units(text)
    if len(units) <= LAYOUT_MIN_PAIRS:  # max travels (units − 1) < 8
        return False
    refs = _layout_lines(reader)
    unused = set(range(len(refs)))
    seq = []
    for unit in units:
        cand = [j for j in unused if refs[j] in unit]
        if cand:
            j = max(cand, key=lambda j: len(refs[j]))
            unused.discard(j)
            seq.append(j)
    travels = [abs(b - a) for a, b in zip(seq, seq[1:])]
    if len(travels) < LAYOUT_MIN_PAIRS:
        return False
    jumps = sum(1 for d in travels if d > LAYOUT_LINE_JUMP)
    return jumps / len(travels) > LAYOUT_JUMP_FRAC


def _layout_lines(reader: PdfReader) -> list[str]:
    """Visual-order normalized lines from pypdf's layout mode, deduped
    (repeated page furniture collapses onto its first occurrence). One
    unreadable page voids the reference — the gate passes."""
    out, seen = [], set()
    for page in reader.pages:
        try:
            text = page.extract_text(extraction_mode='layout')
        except Exception:
            return []
        for line in text.splitlines():
            if len(n := _norm(line)) >= 4 and n not in seen:
                seen.add(n)
                out.append(n)
    return out


def _units(text: str) -> list[str]:
    out = []
    for block in text.split('\n\n'):
        if block.lstrip().startswith('|'):
            out += [line for line in block.splitlines()
                    if line.startswith('|') and not _ROW_SEP.match(line)]
        else:
            out.append(block)
    return [unit for unit in map(_strip_md, out) if len(unit) >= 4]


def _strip_md(markdown: str) -> str:
    return _norm(_U_TAG.sub('', re.sub(r'^[#>\s]+', '', markdown))
                 .replace('*', '').replace('`', '').replace('|', ''))


def _norm(text: str) -> str:
    return re.sub(r'\s+', '', text)


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
