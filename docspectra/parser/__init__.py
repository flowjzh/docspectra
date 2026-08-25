"""File → text. Provider engines as optional extras (``docspectra[pdf-inspect]``,
``docspectra[anydoc]``, ``docspectra[mineru]``), magic-byte type routing,
and file-level decline chains: ``DOCSPECTRA_PARSER_CONFIG`` orders each type's
chain and a provider whose gate fires on a specific file declines, letting the
next engine take it."""

from docspectra.parser.contracts import (Attempt, Declined, FileInput,
                                         ImageAsset, MAX_ASSETS, MAX_ASSET_BYTES,
                                         ParseResult)
from docspectra.parser.core import (
    ParseDeclinedError,
    Provider,
    configure,
    parse,
    parse_path,
    provider,
    register,
)
from docspectra.parser.sniff import sniff, validate_declared

__all__ = [
    'Attempt',
    'Declined',
    'FileInput',
    'ImageAsset',
    'MAX_ASSETS',
    'MAX_ASSET_BYTES',
    'ParseDeclinedError',
    'ParseResult',
    'Provider',
    'configure',
    'parse',
    'parse_path',
    'provider',
    'register',
    'sniff',
    'validate_declared',
]
