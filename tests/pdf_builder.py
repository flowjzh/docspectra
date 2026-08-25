"""Synthetic PDF builder for engine tests (pypdf, zero network).

Covers the synthesizable gate classes: dense-text clean files, raster
pages (sparse text + image), word-strip images, and path-heavy content
streams for the ratio gate."""

from __future__ import annotations

import io
import zlib

from pypdf import PdfWriter
from pypdf.generic import ArrayObject, DictionaryObject, NameObject, NumberObject, StreamObject

_FONT = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                          NameObject('/Subtype'): NameObject('/Type1'),
                          NameObject('/BaseFont'): NameObject('/Helvetica')})


def make_pdf(*, text_lines: int = 25, image: tuple[int, int] | None = None,
             image_rect: tuple[int, int, int, int] = (50, 400, 200, 60),
             path_ops: int = 0, stream_parts: int = 1) -> bytes:
    """One 612×792 page of dense Helvetica text, optionally an XObject
    image (pixel dims, drawn into ``image_rect`` pt), ``path_ops`` vector
    line operators, and the content stream split into ``stream_parts``
    array entries."""
    w = PdfWriter()
    page = w.add_blank_page(width=612, height=792)
    content = b''
    resources: dict[NameObject, object] = {}
    if image is not None:
        img_w, img_h = image
        img = StreamObject()
        img.set_data(zlib.compress(b'\xff\x00\x00' * (img_w * img_h)))
        for key, value in (('/Type', '/XObject'), ('/Subtype', '/Image'),
                           ('/Width', img_w), ('/Height', img_h),
                           ('/ColorSpace', '/DeviceRGB'), ('/BitsPerComponent', 8),
                           ('/Filter', '/FlateDecode')):
            img[NameObject(key)] = NameObject(value) if isinstance(value, str) \
                else NumberObject(value)
        resources[NameObject('/XObject')] = DictionaryObject(
            {NameObject('/Im1'): w._add_object(img)})
        x, y, rw, rh = image_rect
        content += f'q {x} {y} {rw} {rh} re W n /Im1 Do Q\n'.encode()
    for i in range(1, text_lines + 1):
        content += f'BT /F1 12 Tf 50 {792 - i * 30} Td (sample line {i:03d} of text) Tj ET\n'.encode()
    for i in range(path_ops):
        content += f'50 {742 - i % 600} m 550 {742 - i % 600} l S\n'.encode()
    resources[NameObject('/Font')] = DictionaryObject(
        {NameObject('/F1'): w._add_object(_FONT)})
    page[NameObject('/Resources')] = DictionaryObject(resources)
    if stream_parts == 1:
        stream = StreamObject()
        stream.set_data(content)
        page[NameObject('/Contents')] = w._add_object(stream)
    else:
        parts = []
        chunk = len(content) // stream_parts
        for i in range(stream_parts):
            stream = StreamObject()
            stream.set_data(content[i * chunk:(i + 1) * chunk])
            parts.append(w._add_object(stream))
        page[NameObject('/Contents')] = ArrayObject(parts)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def raster_pdf() -> bytes:
    """Sparse text + a small image — the classify-gate class."""
    return make_pdf(text_lines=3, image=(20, 20), image_rect=(200, 300, 60, 60))


def strip_pdf() -> bytes:
    """Dense text + a wide flat strip image — the fingerprint-gate class."""
    return make_pdf(image=(200, 60))
