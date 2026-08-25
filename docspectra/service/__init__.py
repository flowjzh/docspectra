"""The parse service: file + JSON schema → JSON (parser → xtremeparse extractor)."""

from docspectra.service.parse import GenericIssue, ParseService, schema_validator

__all__ = ['ParseService', 'schema_validator', 'GenericIssue']
