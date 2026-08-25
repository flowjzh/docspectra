"""Built-in provider engines. Core try-imports each module at init; a
module whose extra is absent fails to import and is skipped silently."""

from functools import wraps

from opentelemetry.trace import get_tracer

from docspectra.parser import Declined, ParseResult

# shared by the provider engines; spans are no-ops until a host
# registers a tracer provider
tracer = get_tracer('docspectra')


def traced(fn):
    """One span per parse call, named after the engine module — engine
    registrations and their span names can then never drift apart. The
    span carries the outcome bookkeeping (outcome, decline reason,
    input bytes, the meta diagnostics — never text: spans are an
    account ledger like the sidecar, PII stays out; batch_id stays off
    too, it points at remote state holding the file)."""
    span_name = f'{fn.__module__.rsplit(".", 1)[-1]}.parse'

    @wraps(fn)
    async def wrapper(*args, **kwargs):
        with tracer.start_as_current_span(span_name) as span:
            result = await fn(*args, **kwargs)
            if args and isinstance(args[0], (bytes, bytearray)):
                span.set_attribute('parse.bytes', len(args[0]))
            if isinstance(result, Declined):
                span.set_attribute('parse.outcome', 'declined')
                span.set_attribute('parse.reason', result.reason)
            else:
                span.set_attribute('parse.outcome', 'success')
                span.set_attribute('parse.chars', len(result.text))
                span.set_attribute('parse.images', len(result.images))
                for key, value in result.meta.items():
                    if key != 'batch_id':
                        span.set_attribute(f'parse.{key}', value)
            return result

    return wrapper
