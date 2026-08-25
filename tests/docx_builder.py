"""Synthetic docx builder for engine tests (lxml, zero network).

One body (paragraph + text box + table), one header part, and optional
inline-referenced media. Uses the standard ``w:``/``wp:``/``a:``/``wps:``
prefixes — engines match tags by prefix:localname, so lxml's
auto-generated ns-prefixes must not leak in."""

from __future__ import annotations

import io
import zipfile
from lxml import etree

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
WPS = 'http://schemas.microsoft.com/office/word/2010/wordprocessingShape'
PIC = 'http://schemas.openxmlformats.org/drawingml/2006/picture'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
CT = 'http://schemas.openxmlformats.org/package/2006/content-types'
NSMAP = {'w': W, 'wp': WP, 'a': A, 'wps': WPS, 'pic': PIC, 'r': R}


def _q(ns: str, name: str) -> str:
    return f'{{{ns}}}{name}'


def _w(name: str) -> str:
    return _q(W, name)


def _par(text: str):
    p = etree.Element(_w('p'))
    r = etree.SubElement(p, _w('r'))
    t = etree.SubElement(r, _w('t'))
    t.text = text
    return p


def _text_box_par(text: str):
    """A paragraph whose drawing holds a wordprocessingShape text box —
    the shape python-docx's typed API cannot see."""
    p = etree.Element(_w('p'))
    r = etree.SubElement(p, _w('r'))
    drawing = etree.SubElement(r, _w('drawing'))
    anchor = etree.SubElement(drawing, _q(WP, 'anchor'))
    graphic = etree.SubElement(anchor, _q(A, 'graphic'))
    gd = etree.SubElement(graphic, _q(A, 'graphicData'))
    gd.set(_q(R, 'uri'), WPS)
    wsp = etree.SubElement(gd, _q(WPS, 'wsp'))
    txbx = etree.SubElement(wsp, _q(WPS, 'txbx'))
    content = etree.SubElement(txbx, _w('txbxContent'))
    content.append(_par(text))
    return p


def _image_par(rid: str):
    """A paragraph whose drawing inline-references the media part — the
    reference chain an asset extractor needs (media present in the zip
    alone is not enough)."""
    p = etree.Element(_w('p'))
    r = etree.SubElement(p, _w('r'))
    drawing = etree.SubElement(r, _w('drawing'))
    inline = etree.SubElement(drawing, _q(WP, 'inline'))
    graphic = etree.SubElement(inline, _q(A, 'graphic'))
    gd = etree.SubElement(graphic, _q(A, 'graphicData'))
    gd.set(_q(R, 'uri'), PIC)
    pic = etree.SubElement(gd, _q(PIC, 'pic'))
    nv = etree.SubElement(pic, _q(PIC, 'nvPicPr'))
    cnv = etree.SubElement(nv, _q(PIC, 'cNvPr'))
    cnv.attrib.update({'id': '0', 'name': 'image1.png'})
    etree.SubElement(nv, _q(PIC, 'cNvPicPr'))
    blip_fill = etree.SubElement(pic, _q(PIC, 'blipFill'))
    blip = etree.SubElement(blip_fill, _q(A, 'blip'))
    blip.set(_q(R, 'embed'), rid)
    sp = etree.SubElement(pic, _q(PIC, 'spPr'))
    xfrm = etree.SubElement(sp, _q(A, 'xfrm'))
    off = etree.SubElement(xfrm, _q(A, 'off'))
    off.attrib.update({'x': '0', 'y': '0'})
    ext = etree.SubElement(xfrm, _q(A, 'ext'))
    ext.attrib.update({'cx': '100', 'cy': '100'})
    geom = etree.SubElement(sp, _q(A, 'prstGeom'))
    geom.set('prst', 'rect')
    etree.SubElement(geom, _q(A, 'avLst'))
    return p


def _table(rows: list[list[str]]):
    tbl = etree.Element(_w('tbl'))
    for row in rows:
        tr = etree.SubElement(tbl, _w('tr'))
        for cell_text in row:
            tc = etree.SubElement(tr, _w('tc'))
            tc.append(_par(cell_text))
    return tbl


def make_docx(*, media: bool = False) -> bytes:
    """A 1-page docx: body paragraph + text box + 2×2 table, one header
    line, optionally ``word/media/image1.png`` inline-referenced."""
    body = etree.Element(_w('body'))
    body.append(_par('body paragraph one'))
    body.append(_text_box_par('text box content'))
    body.append(_table([['cell A1', 'cell A2'], ['cell B1', 'cell B2']]))
    if media:
        body.append(_image_par('rIdM'))
    body.append(etree.Element(_w('sectPr')))
    document = etree.Element(_w('document'), nsmap=NSMAP)
    document.append(body)
    xml = etree.tostring(document, xml_declaration=True, encoding='UTF-8', standalone=True)

    header_root = etree.Element(_w('header'), nsmap={'w': W, 'r': R})
    header_root.append(_par('header line'))
    header_xml = etree.tostring(header_root, xml_declaration=True, encoding='UTF-8',
                                standalone=True)

    content_types = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="{CT}">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>
</Types>'''

    rels = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{REL}">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''

    image_rel = (f'<Relationship Id="rIdM" Type="http://schemas.openxmlformats.org/'
                 f'officeDocument/2006/relationships/image" Target="media/image1.png"/>\n')
    doc_rels = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{REL}">
<Relationship Id="rIdH" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/>
{image_rel if media else ''}</Relationships>'''

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('[Content_Types].xml', content_types)
        z.writestr('_rels/.rels', rels)
        z.writestr('word/document.xml', xml)
        z.writestr('word/_rels/document.xml.rels', doc_rels)
        z.writestr('word/header1.xml', header_xml)
        if media:
            z.writestr('word/media/image1.png', _PNG_1X1)
    return buf.getvalue()


# a real 1×1 RGB png (69 bytes): engines may reject truncated fixtures
_PNG_1X1 = (
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
    b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc8!\'\x07'
    b'\x00\x02\xb6\x01\x054\xa6u\xaa\x00\x00\x00\x00IEND\xaeB`\x82')
