"""The traced provider decorator's span bookkeeping: outcome, decline
reason, input bytes, and the meta diagnostics — never text (spans are
an account ledger like the sidecar, PII stays out)."""

import pytest

from docspectra.parser import Declined, ParseResult
from docspectra.providers import traced


class FakeSpan:
    def __init__(self, name):
        self.name = name
        self.attributes = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def set_attribute(self, key, value):
        self.attributes[key] = value


class FakeTracer:
    def __init__(self):
        self.spans = []

    def start_as_current_span(self, name):
        span = FakeSpan(name)
        self.spans.append(span)
        return span


@pytest.fixture
def spans(monkeypatch):
    tracer = FakeTracer()
    monkeypatch.setattr('docspectra.providers.tracer', tracer)
    return tracer.spans


async def test_declined_records_outcome_and_reason(spans):
    @traced
    async def parse(data, *, name=None):
        return Declined('table_wall_gate')

    await parse(b'pdf')
    span = spans[0]
    assert span.name == 'test_traced.parse'
    assert span.attributes == {'parse.bytes': 3,
                               'parse.outcome': 'declined',
                               'parse.reason': 'table_wall_gate'}


async def test_success_records_sizes_never_text(spans):
    @traced
    async def parse(data, *, name=None):
        return ParseResult(text='secret resume text',
                           meta={'pages': 1, 'skipped_assets': 2,
                                 'cache_hit': True, 'model_version': 'pipeline',
                                 'batch_id': 'remote-pointer', 'noise': 'x'})

    await parse(b'pdf')
    assert spans[0].attributes == {
        'parse.bytes': 3,
        'parse.outcome': 'success',
        'parse.chars': 18,
        'parse.images': 0,
        'parse.pages': 1,
        'parse.skipped_assets': 2,
        'parse.cache_hit': True,
        'parse.model_version': 'pipeline',
        'parse.noise': 'x',
    }
