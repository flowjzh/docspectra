"""Provider registry, decline chains, and the async parse entry.

Core carries zero parsing dependencies: built-in providers are in-tree
modules under ``docspectra.providers``, try-imported at init — an absent
extra is skipped silently (an installed-but-broken provider is recorded
and surfaced in config errors instead). Engines declare themselves with
:func:`provider`; chains are ordered by ``DOCSPECTRA_PARSER_CONFIG`` (a
JSON env var mapping a document type to an ordered provider list),
falling back to the derived default chain — the canonical order of
installed providers that declared the type. Declines are return values,
hard errors fall through; chain exhaustion raises
:class:`ParseDeclinedError` with every attempt aggregated."""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Sequence

from docspectra.parser.contracts import Attempt, Declined, ParseResult
from docspectra.parser.sniff import sniff, validate_declared

# canonical order = the derived default chain: anydoc first for text-layer
# pdfs, then bce for scanned ones; mineru stays last — the three cloud
# extras share one dependency set (httpx + xtremeflow), so install state
# alone cannot rank them and this order is what keeps OCR on bce.
_BUILTIN_MODULES = ('txt', 'pdf_inspect', 'anydoc', 'bce', 'paddleocr', 'mineru')
_BUILTIN_NAMES = tuple(m.replace('_', '-') for m in _BUILTIN_MODULES)
_BUILTIN_RANK = {name: i for i, name in enumerate(_BUILTIN_NAMES)}
_CONFIG_ENV = 'DOCSPECTRA_PARSER_CONFIG'


@dataclass
class Provider:
    """One engine: the async parse entry plus the types it declares."""

    name: str
    types: tuple[str, ...]
    parse: Callable[..., Awaitable[ParseResult | Declined]]


class ParseDeclinedError(Exception):
    """Every provider in the chain declined or errored on this file.

    ``attempts`` aggregates every outcome. A hard error raised through
    the chain is chained as the cause.
    """

    def __init__(self, attempts: list[Attempt]):
        self.attempts = attempts
        detail = '; '.join(
            f"{a.provider}: {a.outcome}" + (f' ({a.reason})' if a.reason else '')
            for a in attempts
        )
        super().__init__(f'every provider declined: {detail}')


_providers: dict[str, Provider] = {}  # keyed by normalized (hyphen) engine name
_builtins_loaded = False
_config: dict | None = None  # effective chain config; resolved from env on first use
_import_errors: dict[str, str] = {}  # provider module → failure, when installed-but-broken


def _norm(name: str) -> str:
    return name.replace('_', '-')


def register(name: str, types: Sequence[str], fn) -> None:
    """Programmatic registration, same surface as the decorator. Names are
    normalized (hyphen↔underscore) and registration is last-write-wins."""
    _providers[_norm(name)] = Provider(_norm(name), tuple(types), fn)


def provider(name: str, types: Sequence[str]):
    """Declare an engine around an async ``parse(data, *, name=None)
    -> ParseResult | Declined``."""

    def deco(fn):
        register(name, types, fn)
        return fn

    return deco


def ensure_builtins() -> None:
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True
    for module in _BUILTIN_MODULES:
        try:
            importlib.import_module(f'docspectra.providers.{module}')
        except ImportError as e:
            if not isinstance(e, ModuleNotFoundError) \
                    or (e.name or '').startswith('docspectra'):
                _import_errors[_norm(module)] = f'{type(e).__name__}: {e}'


def configure(config: dict | None = None) -> None:
    """Pin chains to an explicit ``{type: [engine, ...]}`` mapping, or
    reload them from DOCSPECTRA_PARSER_CONFIG (None, validated here —
    the fail-fast hook a host calls at startup)."""
    global _config
    ensure_builtins()
    _config = _validate(config) if config is not None else _env_config()


async def parse(data: bytes, *, name: str | None = None) -> ParseResult:
    """Parse one file (bytes are the contract) into a ParseResult,
    walking the type's decline chain. Raises ValueError on unroutable
    input, on a declared binary extension whose content fails magic
    verification, or when no provider is installed; ParseDeclinedError
    when every provider declines or errors."""
    ensure_builtins()
    validate_declared(data, name)  # declared-binary gate, before any provider
    doc_type = sniff(data, name)
    if doc_type is None:
        raise ValueError(f'cannot determine document type of {name or "input"}')
    chain = _chain_for(doc_type)
    if not chain:
        raise ValueError(f'no provider installed for document type {doc_type!r}')
    attempts: list[Attempt] = []
    cause = None
    for provider_name in chain:
        p = _providers[provider_name]
        try:
            result = await p.parse(data, name=name)
        except Exception as e:
            attempts.append(Attempt(p.name, 'error', f'{type(e).__name__}: {e}'))
            cause = cause or e
            continue
        if isinstance(result, Declined):
            attempts.append(Attempt(p.name, 'declined', result.reason))
            continue
        result.provider = p.name
        result.attempts = [*attempts, Attempt(p.name, 'success')]
        return result
    raise ParseDeclinedError(attempts) from cause


async def parse_path(path: str | Path) -> ParseResult:
    """Convenience wrapper: read the file and parse it, name from the path."""
    path = Path(path)
    return await parse(path.read_bytes(), name=path.name)


def _chain_for(doc_type: str) -> list[str]:
    config = _effective_config()
    if doc_type in config:
        return list(config[doc_type])
    # derived default chain: canonical order of installed providers that
    # declared the type — extends automatically as built-ins are added.
    # Ranked by the canonical module order, not import order: a host that
    # imports a provider module directly must not reorder the chains
    # (host-registered providers keep their order after the built-ins).
    names = [p.name for p in _providers.values() if doc_type in p.types]
    return sorted(names, key=lambda n: _BUILTIN_RANK.get(n, len(_BUILTIN_NAMES)))


def _effective_config() -> dict:
    global _config
    if _config is None:
        _config = _env_config()
    return _config


def _env_config() -> dict:
    raw = os.environ.get(_CONFIG_ENV)
    if not raw:
        return {}
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f'{_CONFIG_ENV} is not valid JSON: {e}') from e
    if not isinstance(config, dict):
        raise ValueError(f'{_CONFIG_ENV} must be a JSON object of type → engine list')
    return _validate(config)


def _validate(config: dict) -> dict:
    validated = {}
    for doc_type, names in config.items():
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError(f'{_CONFIG_ENV}[{doc_type!r}] must be a list of engine names')
        normed = []
        for name in names:
            key = _norm(name)
            if key not in _providers:
                if key in _BUILTIN_NAMES:
                    detail = f' ({_import_errors[key]})' if key in _import_errors else ''
                    raise ValueError(
                        f'provider {name!r} is configured but not installed{detail} — '
                        f'install the matching extra (docspectra[{key}])')
                raise ValueError(
                    f'unknown parser provider {name!r}; known: {sorted(_providers)}')
            normed.append(key)
        validated[doc_type] = normed
    return validated
