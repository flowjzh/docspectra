"""txt/md passthrough — the one built-in provider that needs no extra."""

from __future__ import annotations

from docspectra.parser import Declined, ParseResult, provider
from docspectra.providers import traced


@provider('txt', ('txt', 'md', 'markdown'))
@traced
async def parse_txt(data: bytes, *, name: str | None = None) -> ParseResult | Declined:
    try:
        return ParseResult(text=data.decode('utf-8-sig'))
    except UnicodeDecodeError:
        return Declined('utf-8 decode failed')
