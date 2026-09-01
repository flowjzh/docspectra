"""The shared cloud plumbing — lane spec parsing and the result LRU."""

import pytest

from docspectra.parser import ParseResult
from docspectra.providers._cloud import ResultCache, lanes


def test_malformed_lane_spec(monkeypatch):
    monkeypatch.setenv('X_CONCURRENCY', 'submit:60')  # missing status lane
    with pytest.raises(ValueError, match='status'):
        lanes('X_CONCURRENCY', 'submit:60|status:600')


def test_facade_shares_payload_not_meta():
    cache = ResultCache()
    key = ResultCache.key(b'payload', 'm')
    cache.put(key, ParseResult(text='t', images=[], meta={'job_id': 'j1'}))
    hit = cache.facade(key, model='m', cache_hit=True)
    assert hit.text == 't' and hit.meta == {'model': 'm', 'cache_hit': True}
    assert cache.facade(ResultCache.key(b'other', 'm')) is None
