"""The scan-level text gate: the table wall — one markdown table
swallowing a page of flowing prose, the layout-band false positive of
geometry-based table detection (parser.md §1). Engine-independent —
the mangled shapes are strings, so no local engine extra is needed."""

from docspectra.parser import Declined
from docspectra.providers.pdfscan import scan, table_wall_hit
from tests.pdf_builder import make_pdf

# max row 602 > WALL_ROW_CHARS, share 1.0 ≥ TABLE_SHARE_MIN — both
# thresholds explicit, in the gate-test style of sizing fixtures to them
WALL = '\n'.join(['|' + 'x' * 600 + '|'] + ['|short|row|'] * 5)

PROSE_PLUS_TABLE = ('Intro paragraph.\n\nMore prose here.\n\n'
                    '|Item|Qty|\n|---|---|\n|Widget|12|\n|Gadget|7|\n')

DATA_PAGE = '\n'.join(f'|Item {i}|{i * 3}|$ {i}.50|' for i in range(40))


def test_table_wall_declines():
    assert table_wall_hit(WALL)


def test_prose_with_real_table_stays():
    assert not table_wall_hit(PROSE_PLUS_TABLE)


def test_full_page_data_table_stays():
    assert not table_wall_hit(DATA_PAGE)


def test_scan_routes_table_wall_to_decline():
    declined = scan(make_pdf(), WALL)
    assert isinstance(declined, Declined)
    assert declined.reason == 'table_wall_gate'


def test_scan_keeps_data_page():
    result = scan(make_pdf(), DATA_PAGE)
    assert not isinstance(result, Declined)
