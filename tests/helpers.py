"""Shared test doubles and fixtures: owned by the protocol owner."""

import io
import zipfile

from xtremeparse.testing import FakeIssue, ScriptedRunner, agent_result

__all__ = ['FakeIssue', 'PDF', 'PNG', 'ScriptedRunner', 'agent_result']

PDF = b'%PDF-1.4\n\xff\xfename=resume.pdf'
PNG = b'\x89PNG\r\n\x1a\n' + b'x' * 32
OLE2 = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1' + b'\x00' * 64


def zip_with(*entries: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for entry in entries:
            z.writestr(entry, '<x/>')
    return buf.getvalue()
