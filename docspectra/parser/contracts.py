"""Parser contracts: result shapes and control-flow values shared by core
and providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

FileInput = bytes | str | Path


@dataclass
class ImageAsset:
    """One embedded image carried faithfully, uninterpreted.

    ``name`` matches the ``![](...)`` reference in ParseResult.text;
    ``source`` is engine provenance ('pdf p2', 'word/media/image1.png').
    """

    name: str
    mime: str
    data: bytes
    source: str


# payload-budget policy for ParseResult.images — every provider skips
# assets over these limits and records the skip count in meta
MAX_ASSETS = 50
MAX_ASSET_BYTES = 5 * 1024 * 1024


@dataclass
class Attempt:
    """One provider's outcome on a file, recorded by core on success and
    failure alike — the eval suite reads routing through it."""

    provider: str
    outcome: Literal['success', 'declined', 'error']
    reason: str | None = None


@dataclass
class Declined:
    """Expected control flow: this engine cannot read *this* file, or
    read it into a known false-positive shape (a gate fired). Never an
    exception — the chain moves to the next provider."""

    reason: str


@dataclass
class ParseResult:
    """The outcome of a successful parse. ``format`` is constrained —
    extend the Literal, never free-form. ``meta`` is provider diagnostics
    (gate hits, page counts, task ids, timings). ``provider`` and
    ``attempts`` are stamped by core, not the provider."""

    text: str
    provider: str = ''  # engine that accepted the file (core-stamped)
    format: Literal['markdown'] = 'markdown'
    images: list[ImageAsset] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
