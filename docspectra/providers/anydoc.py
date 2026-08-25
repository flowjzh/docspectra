"""Local superset engine (extra: anydoc): firecrawl-anydoc (Rust) over
pdf plus every office family. The pdf branch runs the shared decline
gates — classify from the engine's own ImageBased/Scanned rejection,
the rest from ``_pdfscan``; office formats (doc/docx, xls/xlsx, ppt/pptx,
odt/ods/odp, rtf, epub, csv) convert to markdown with their embedded
images carried as ImageAssets. Docx headers and footers are page
furniture the engine drops by design."""

from __future__ import annotations

import anydoc

from docspectra.parser import (Declined, ImageAsset, MAX_ASSETS,
                               MAX_ASSET_BYTES, ParseResult, provider,
                               sniff)
from docspectra.providers import traced
from docspectra.providers._pdfscan import scan

# sniff type → engine Format for the office branch; None lets the engine
# detect from the bytes (the OLE2 family — 'doc' covers legacy doc/xls/ppt
# alike, and xls has no Format name of its own). An explicit map, not a
# passthrough: a sniff/engine vocabulary drift must fail here, not inside
# the engine as a runtime error.
_FORMAT: dict[str, str | None] = {
    'docx': 'docx', 'doc': None,
    'ppt': 'ppt', 'pptx': 'pptx', 'xls': None, 'xlsx': 'xlsx',
    'odt': 'odt', 'ods': 'ods', 'odp': 'odp',
    'rtf': 'rtf', 'epub': 'epub', 'csv': 'csv',
}


@provider('anydoc', ('pdf', *_FORMAT))
@traced
async def parse_anydoc(data: bytes, *, name: str | None = None) -> ParseResult | Declined:
    # core already sniffed for routing; re-sniffed here because the
    # provider contract carries (data, name) and the engine Format and
    # the pdf/office branch both hang off the type
    if (doc_type := sniff(data, name)) == 'pdf':
        return _parse_pdf(data)
    return _parse_office(data, doc_type)


def _parse_pdf(data: bytes) -> ParseResult | Declined:
    try:
        text = anydoc.to_markdown_bytes(data)
    except anydoc.UnsupportedError:
        # the engine's own rejection: ImageBased/Scanned pages need OCR
        return Declined('classify_gate')
    return scan(data, text)


def _parse_office(data: bytes, doc_type: str) -> ParseResult:
    # signature-less formats (csv) must name their format explicitly
    fmt = _FORMAT[doc_type]
    text = anydoc.to_markdown_bytes(data, fmt)
    images, skipped = [], 0
    for asset in anydoc.to_document(data, fmt).assets:
        if len(images) >= MAX_ASSETS or len(asset.data) > MAX_ASSET_BYTES:
            skipped += 1
            continue
        images.append(ImageAsset(
            name=asset.origin_part, mime=asset.media_type, data=asset.data,
            source=f'anydoc {asset.origin_part}'))
    return ParseResult(text=text, images=images,
                       meta={'skipped_assets': skipped})
