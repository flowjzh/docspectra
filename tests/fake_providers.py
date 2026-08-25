"""In-memory fake engines for chain-scheduling tests (zero network).

Behavior is scripted per test via BEHAVIORS: 'accept' (default),
'decline', or 'raise'."""

from docspectra.parser import Declined, ParseResult, provider

BEHAVIORS: dict[str, str] = {}


def _run(name: str, data: bytes) -> ParseResult | Declined:
    mode = BEHAVIORS.get(name, 'accept')
    if mode == 'decline':
        return Declined(f'{name}-gate')
    if mode == 'raise':
        raise RuntimeError(f'{name} exploded')
    return ParseResult(text=f'{name}:{data.decode("latin1")}')


@provider('fake-a', ('pdf', 'docx'))
async def _fake_a(data: bytes, *, name: str | None = None):
    return _run('fake-a', data)


@provider('fake-b', ('pdf', 'doc', 'xlsx'))
async def _fake_b(data: bytes, *, name: str | None = None):
    return _run('fake-b', data)


@provider('fake_c', ('pdf', 'jpg', 'pptx'))
async def _fake_c(data: bytes, *, name: str | None = None):
    return _run('fake_c', data)
