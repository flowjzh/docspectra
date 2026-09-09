"""Markdown shaping shared by the providers: engines hand back markdown,
the extraction pipeline reads lines. The flattener turns the HTML table
blocks MinerU emits inside full.md into line text; the table-wall
predicate flags one pipe table (the pdf and office engines' own
serialization) swallowing the page. A provider whose engine grows
another such shape hosts its normalizer here — stdlib plus a chunking
constant only, importable from anywhere without an engine dependency."""

from __future__ import annotations

import re
from html.parser import HTMLParser

TABLE_SHARE_MIN = 0.7  # of all text chars; judgment call — measured wall
# 0.99, corpus real tables a minority share (parser.md §1)
WALL_ROW_CHARS = 500  # chars in one row; the engine's own paragraph-wall
# constant, measured wall row ~1,300 (parser.md §1)

_TABLE = re.compile(r'<table[^>]*>.*?</table>', re.DOTALL)


class TableFlattener(HTMLParser):
    """One ``<table>`` block → lines, rendered row by row as each row
    closes. A row within ``max_chars`` inlines as one line (cells
    space-joined — the label-value shape a resume card reads from); a
    longer row is a document section, and explodes, each block its own
    line. Image tags become the ``![](src)`` references the ImageAssets
    reconcile with. convert_charrefs folds the entities."""
    _BLOCK = {'p', 'li', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}

    def __init__(self, max_chars: int):
        super().__init__(convert_charrefs=True)
        self._max_chars = max_chars
        self.lines: list[str] = []
        self._row: list[list[str]] = []
        self._cell = ''
        self._blocks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ('td', 'th'):
            self._cell, self._blocks = '', []
        elif tag == 'br':
            self._flush()
        elif tag == 'img':
            self._cell += f'![]({dict(attrs).get("src", "")})'

    def handle_endtag(self, tag):
        if tag in ('td', 'th'):
            self._flush()
            self._row.append(self._blocks)
            self._blocks = []
        elif tag == 'tr':
            self._end_row()
        elif tag in self._BLOCK:
            self._flush()

    def handle_data(self, data):
        self._cell += data

    def _flush(self):
        if (text := self._cell.strip()):
            self._blocks.append(text)
        self._cell = ''

    def _end_row(self):
        self._flush()
        if self._blocks:  # a cell never closed keeps its blocks
            self._row.append(self._blocks)
        self._blocks = []
        if not self._row:
            return
        inline = '  '.join(' '.join(cell) for cell in self._row)
        if len(inline) <= self._max_chars:
            self.lines.append(inline)
        else:
            self.lines.extend(block for cell in self._row for block in cell)
        self._row = []

    def text(self) -> str:
        return '\n'.join(self.lines)


def flatten_tables(markdown: str) -> str:
    """Every ``<table>`` block of the markdown flattened to line text.
    The surrounding markdown passes through untouched — its blank lines
    included — until a table exists: from then on blank lines drop
    document-wide, layout truth being the remaining lines' order."""
    if '<table' not in markdown:
        return markdown
    # lazy: parser-only consumers must not pay the extractor init
    # (the xtremeparse package import) for a constant
    from xtremeparse.chunking import MAX_CHARS
    parts: list[str] = []

    def keep(text: str) -> None:
        parts.extend(line for line in text.split('\n') if line.strip())

    pos = 0
    for table in _TABLE.finditer(markdown):
        keep(markdown[pos:table.start()])
        flattener = TableFlattener(MAX_CHARS)
        flattener.feed(table.group(0))
        keep(flattener.text())
        pos = table.end()
    keep(markdown[pos:])
    return '\n'.join(parts)


def table_wall_hit(text: str) -> bool:
    """One markdown table swallowing the page's flowing prose — the
    layout-band false positive of geometry-based table detection in the
    pdf engines, and a real whole-document Word table in office ones
    (the legacy template class, serialized rows a section wide). Both
    conditions matter: a document that merely carries a real table
    stays, and a page of genuine data cells (high table share, short
    rows) stays."""
    lines = [line for line in text.splitlines() if line.strip()]
    rows = [line for line in lines
            if line.startswith('|') and line.strip('|- ')]
    if not rows:
        return False
    total = sum(map(len, lines))
    return (sum(map(len, rows)) / total >= TABLE_SHARE_MIN
            and max(map(len, rows)) > WALL_ROW_CHARS)
