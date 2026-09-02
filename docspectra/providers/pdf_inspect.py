"""Local pdf engine (extra: pdf-inspect): pdf-inspector extraction with
a sampling-classify decline gate (raster pages); the fingerprint/ratio
gates, image collection, and post-processing live in ``pdfscan``."""

from __future__ import annotations

import pdf_inspector

from docspectra.parser import Declined, ParseResult, provider
from docspectra.providers import traced
from docspectra.providers.pdfscan import scan


@provider('pdf-inspect', ('pdf',))
@traced
async def parse_pdf(data: bytes, *, name: str | None = None) -> ParseResult | Declined:
    det = pdf_inspector.process_pdf_bytes(data)
    if det.pages_needing_ocr:
        return Declined('classify_gate')
    return scan(data, det.markdown or '', pdf_type=str(det.pdf_type),
                encoding_issues=det.has_encoding_issues)
