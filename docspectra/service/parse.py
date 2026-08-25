"""The parse service: file + JSON schema → JSON (parser → extractor)."""

from __future__ import annotations

from dataclasses import dataclass

from jsonschema import Draft202012Validator

from xtremeparse import Extractor, ExtractionResult

from docspectra.parser import FileInput, parse as parse_file, parse_path


class ParseService:
    """One file and one JSON Schema in, best-effort schema-shaped JSON
    out — validated and self-corrected against a generic JSON Schema
    validator (hosts with their own validator, e.g. docxcast, inject it
    through the extractor directly)."""

    def __init__(self, runner, *, scheduler=None):
        self._extractor = Extractor(runner, scheduler=scheduler)

    async def parse(self, data: FileInput, schema: dict, *,
                    name: str | None = None, validator=None) -> ExtractionResult:
        document = await parse_file(data, name=name) if isinstance(data, bytes) \
            else await parse_path(data)
        return await self._extractor.extract(document.text, schema,
                                             validator=validator or schema_validator(schema))


@dataclass
class GenericIssue:
    """JSON-Schema error shaped to the Issue protocol; array indices
    render bracketed (`jobs[0].company`) so correction routing matches."""

    path: str
    message: str
    code: str
    expected: object = None
    got: object = None


def schema_validator(schema: dict):
    """Validate data against a JSON Schema, returning error-level issues."""
    validator = Draft202012Validator(schema)

    def validate(data: dict) -> list:
        return [GenericIssue(_path(e), e.message, e.validator) for e in validator.iter_errors(data)]

    return validate


def _path(error) -> str:
    path = ''
    for part in error.absolute_path:
        if isinstance(part, int):
            path = f'{path}[{part}]'
        else:
            path = f'{path}.{part}' if path else str(part)
    return path or '$'
