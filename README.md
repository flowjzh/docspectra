# DocSpectra

> **Any file in. Spectral JSON out. Fast.**

DocSpectra is a schema-driven document-parsing library: hand it a file in
any supported format and a JSON Schema, get back schema-conforming JSON —
extracted by extreme-concurrency LLM specialists that chunk, route, fan
out and self-correct (powered by
[xtremeparse](https://gitlab.com/flowjzh/xtremeparse)).

```
file + JSON schema ──► docspectra.parse ──► JSON
```

## Layout

- `docspectra/parser` — file → text (provider engines as optional extras:
  `pdf-inspect`, `anydoc`, `mineru`; magic-byte type routing;
  file-level decline chains — a provider whose gate fires on a specific
  file declines and the next engine takes it. Design record:
  `apps/cvspectra/docs/parser.md` in the cohirerx monorepo)
- `docspectra/agents` — the only PydanticAI boundary (AgentRunner adapter)
- `docspectra/service` — `parse`: file + JSON schema → JSON

DocSpectra is deliberately generic: JSON Schema in and out, no document
vendors, no HTTP server, no domain tools. Applications assemble it with
their own schema sources and renderers (e.g. the cvspectra deployment
bridges [docxcast](https://gitlab.com/flowjzh/docxcast) templates in the
cohirerx monorepo, where the governing design record and slice plan live).

## Status

Parser layer implemented and tested. See `apps/cvspectra/docs/` in the
cohirerx monorepo for the governing design record and slice plan.
