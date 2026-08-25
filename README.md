# DocSpectra

<img width="600" alt="DocSpectra" src="https://github.com/user-attachments/assets/1d858fb3-67c5-41f6-b0db-3fb6ab39fe4a" />

> **Any file in. Spectral JSON out. Fast.**

DocSpectra is a schema-driven document-parsing library: hand it a file in
any supported format and a JSON Schema, get back schema-conforming JSON —
extracted by extreme-concurrency LLM specialists that chunk, route, fan
out and self-correct (powered by
[xtremeparse](https://github.com/flowjzh/xtremeparse)).

```
file + JSON schema ──► docspectra.parse ──► JSON
```

## Layout

- `docspectra/parser` — file → text (provider engines as optional extras:
  `pdf-inspect`, `anydoc`, `mineru`; magic-byte type routing;
  file-level decline chains — a provider whose gate fires on a specific
  file declines and the next engine takes it)
- `docspectra/agents` — the only PydanticAI boundary (AgentRunner adapter)
- `docspectra/service` — `parse`: file + JSON schema → JSON

DocSpectra is deliberately generic: JSON Schema in and out, no document
vendors, no HTTP server, no domain tools. Applications assemble it with
their own schema sources and renderers (e.g. bridging
[DocXCast](https://github.com/flowjzh/docxcast) templates).

## Status

Parser layer implemented and tested.
