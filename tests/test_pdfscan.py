"""The scan-level wall routing: a wall-shaped text handed to ``scan``
declines so the chain escalates, while a data page stays a result. The
predicate itself is tested in ``test_markdown.py`` (its module home)."""

from docspectra.parser import Declined
from docspectra.providers.pdfscan import scan
from tests.helpers import DATA_PAGE, WALL
from tests.pdf_builder import make_pdf


def test_scan_routes_table_wall_to_decline():
    declined = scan(make_pdf(), WALL)
    assert isinstance(declined, Declined)
    assert declined.reason == 'table_wall_gate'


def test_scan_keeps_data_page():
    result = scan(make_pdf(), DATA_PAGE)
    assert not isinstance(result, Declined)
