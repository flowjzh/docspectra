"""The scan-level routing: wall-shaped and layout-scrambled text handed
to ``scan`` decline so the chain escalates, while a data page stays a
result. The layout predicate's alignment seams — thin evidence and
ordered emission — are probed directly."""

from docspectra.parser import Declined
from docspectra.providers.pdfscan import layout_hit, scan
from tests.helpers import DATA_PAGE, WALL
from tests.pdf_builder import make_pdf, reader, sample_line


def _doc(*ranks: int) -> str:
    """Emission units with the blank-line separation ``_units`` splits
    on, drawn from the make_pdf() reference lines."""
    return '\n\n'.join(sample_line(i) for i in ranks)


# every consecutive SCRAMBLED pair travels > LAYOUT_LINE_JUMP ranks,
# ORDERED is the same lines in order
SCRAMBLED = _doc(1, 25, 2, 24, 3, 23, 4, 22, 5, 21)
ORDERED = _doc(*sorted((1, 25, 2, 24, 3, 23, 4, 22, 5, 21)))


def test_scan_routes_table_wall_to_decline():
    declined = scan(make_pdf(), WALL)
    assert isinstance(declined, Declined)
    assert declined.reason == 'table_wall_gate'


def test_scan_routes_layout_scramble_to_decline():
    declined = scan(make_pdf(), SCRAMBLED)
    assert isinstance(declined, Declined)
    assert declined.reason == 'layout_gate'


def test_scan_keeps_data_page():
    result = scan(make_pdf(), DATA_PAGE)
    assert not isinstance(result, Declined)


def test_layout_hit_keeps_ordered_emission():
    assert not layout_hit(reader(make_pdf()), ORDERED)


def test_layout_hit_passes_thin_alignment():
    # the unit-count bound decides before the layout-mode extraction —
    # a reader that got touched would fail this test
    assert not layout_hit(None, _doc(1, 25, 2))


def test_layout_hit_reads_the_shared_serializer_shapes():
    # headings, pipe table with separator row, prose — serializer
    # revisions fail here, not silently at the gate
    doc = (f'## {sample_line(25)}\n\n'
           f'| {sample_line(1)} |\n|---|\n| {sample_line(12)} |\n\n'
           + '\n\n'.join(f'## {sample_line(i)}' for i in (24, 2, 23, 3, 22, 4, 21, 5)))
    assert layout_hit(reader(make_pdf()), doc)
