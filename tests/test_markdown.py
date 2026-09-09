"""The shared markdown shaping behind the providers: HTML table blocks
flatten to the line text the extraction pipeline reads, and the
table-wall predicate reads engine pipe rows as a decline gate."""

from docspectra.providers._markdown import flatten_tables, table_wall_hit
from tests.helpers import DATA_PAGE, PROSE_PLUS_TABLE, WALL


def test_short_rows_inline_cells_space_joined():
    md = ('<table><tr><td><p>name</p></td><td><p>value</p></td></tr>'
          '<tr><td><p>city</p></td><td><p>berlin</p></td></tr></table>')
    assert flatten_tables(md) == 'name  value\ncity  berlin'


def test_section_cell_explodes_into_lines():
    # a row past the chunk budget — a document section, not a
    # label-value row — renders its blocks one per line
    block = lambda i: f'line {i} ' + 'x' * 90
    md = '<table><tr><td colspan="2">' \
         + ''.join(f'<p>{block(i)}</p>' for i in range(3)) + '</td></tr></table>'
    assert flatten_tables(md) == f'{block(0)}\n{block(1)}\n{block(2)}'


def test_long_inline_row_still_explodes():
    # the inline ceiling is a chunk budget: a padded row past it renders
    # per block even though each cell holds a single paragraph
    pad = 'x' * 120
    md = f'<table><tr><td><p>{pad}</p></td><td><p>{pad}</p></td><td><p>{pad}</p></td></tr></table>'
    lines = flatten_tables(md).split('\n')
    assert lines == [pad, pad, pad]


def test_image_tag_becomes_markdown_reference():
    md = '<table><tr><td><img src="images/a.png"/></td><td><p>portrait</p></td></tr></table>'
    assert flatten_tables(md) == '![](images/a.png)  portrait'


def test_br_and_entities_inside_cells():
    md = '<table><tr><td><p>a<br>b</p></td><td><p>Q&amp;A</p></td></tr></table>'
    assert flatten_tables(md) == 'a b  Q&A'


def test_markdown_without_tables_passes_through():
    md = '# heading\n\nprose line\n\n- bullet\n'
    assert flatten_tables(md) == md


def test_prose_around_the_table_is_kept():
    md = 'before\n\n<table><tr><td><p>cell</p></td></tr></table>\n\nafter'
    assert flatten_tables(md) == 'before\ncell\nafter'


def test_table_wall_declines():
    assert table_wall_hit(WALL)


def test_prose_with_real_table_stays():
    assert not table_wall_hit(PROSE_PLUS_TABLE)


def test_full_page_data_table_stays():
    assert not table_wall_hit(DATA_PAGE)
