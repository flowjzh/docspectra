"""Shared test doubles and fixtures: owned by the protocol owner."""

import io
import zipfile

from xtremeparse.testing import FakeIssue, ScriptedRunner, agent_result

__all__ = ['DATA_PAGE', 'FakeIssue', 'PDF', 'PNG', 'ScriptedRunner',
           'WALL', 'agent_result']

PDF = b'%PDF-1.4\n\xff\xfename=resume.pdf'
PNG = b'\x89PNG\r\n\x1a\n' + b'x' * 32
OLE2 = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1' + b'\x00' * 64

# max row 602 > WALL_ROW_CHARS, share 1.0 ≥ TABLE_SHARE_MIN — both
# thresholds explicit, in the gate-test style of sizing fixtures to them
WALL = '\n'.join(['|' + 'x' * 600 + '|'] + ['|short|row|'] * 5)
DATA_PAGE = '\n'.join(f'|Item {i}|{i * 3}|$ {i}.50|' for i in range(40))
PROSE_PLUS_TABLE = ('Intro paragraph.\n\nMore prose here.\n\n'
                    '|Item|Qty|\n|---|---|\n|Widget|12|\n|Gadget|7|\n')


def zip_with(*entries: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for entry in entries:
            z.writestr(entry, '<x/>')
    return buf.getvalue()
