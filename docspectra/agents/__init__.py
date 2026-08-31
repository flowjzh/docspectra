"""The only PydanticAI boundary: xtremeparse's AgentRunner adapted here.
Swapping agent frameworks means rewriting this module and nothing else."""

from __future__ import annotations

import json
import re
from typing import Sequence

import httpx
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models import Model
from pydantic_ai.output import StructuredDict

from xtremeparse.contracts import AgentResult, Issue, JSONSchema
from xtremeparse.scheduling import report_tokens


# qwen commercial/numeric models think by default and dashscope rejects
# tool_choice='required' (structured output) while thinking — same quirk
# argus works around; generalized to any qwen3.x numeric series
_THINKING_OFF = re.compile(
    r'^qwen-(plus|flash|turbo|max)(-.*)?$|'
    r'^qwen3(\.\d+)?-(omni-flash|vl-plus|flash|plus|max|\d+b(-a\d+b)?)(-.*)?$')


class PydanticAIRunner:
    """Adapts the AgentRunner protocol to PydanticAI.

    The shared ``content`` prefix rides its own system message (Agent
    ``instructions``) so provider prefix caching sees identical bytes
    across every call of one extraction; the unit card, scope and
    feedback trail in the user message — a history round already
    carries the card and scope, so the caller passes ``''`` for them
    and only the per-round feedback is fresh. Structured output goes
    through the provider's ``json_object`` channel by default: the
    schema rides the prompt (the shared payload embeds it) and the
    wire enforces only valid JSON — dashscope's ``json_schema``
    grammar under a non-thinking decode collapses long-material
    arrays into the cheapest valid output (empty arrays, stray
    fragments), which the adapter-side parse-retry avoids entirely.
    Fenced replies and a single-key wrapper the model may add are
    tolerated; only the top-level shape is checked — deep conformance
    is the injected validator's job. ``json_object=False`` restores
    the native JSON-schema grammar (``strict_output`` governs it).
    Sampling defaults to temperature 0 (extraction is
    deterministic-intent); ``model_settings`` overrides any field.
    Actual token usage is reported to the dispatching scheduler via
    xtremeparse's ``report_tokens``. ``tools`` is accepted (frozen
    protocol) but not implemented yet."""

    def __init__(self, model: Model, *, model_settings: dict | None = None,
                 strict_output: bool = True, json_object: bool = True):
        self.model = model
        # deterministic-intent extraction: sampling heat off by default
        self.model_settings = {'temperature': 0, **(model_settings or {})}
        # strict keeps the schema in a server-side grammar; without it some
        # providers inject the schema into the prompt behind the payload and
        # break the cacheable prefix — hosts whose strict grammar rejects
        # the schema shape turn it off here
        self.strict_output = strict_output
        self.json_object = json_object

    async def run(self, *, instructions: str, result_schema: JSONSchema, content: str,
                  scope: str = None, tools: list = None, history: list = None,
                  feedback: Sequence[Issue] = None) -> AgentResult:
        if tools:
            raise NotImplementedError('tool-calling is deferred; see design.md §5.1')
        free = result_schema.get('type') == 'string'  # free-text mode: lib parses
        if free:
            agent = Agent(self.model, output_type=str, instructions=content)
            return await self._run_once(agent, instructions, scope, feedback, history)
        if self.json_object:
            agent = Agent(self.model, output_type=str, instructions=content)
            return await self._run_json_object(agent, result_schema,
                                               instructions, scope, feedback, history)
        wrapped = result_schema.get('type') == 'array'
        agent = Agent(self.model,
                      output_type=NativeOutput(StructuredDict(_wrap(result_schema)
                                                               if wrapped else result_schema),
                                               strict=self.strict_output),
                      instructions=content)
        result = await self._raw_run(agent, _tail(instructions, scope, feedback), history)
        data = result.output.get('items') if wrapped else result.output
        return AgentResult(data=data, history=result.all_messages())

    async def _run_json_object(self, agent, result_schema, instructions, scope,
                               feedback, history):
        """One or two passes over the json_object channel: a reply that
        fails the adapter's shape check gets one repair round carrying
        the parse error; a still-unusable reply yields None data — the
        host validator's missing-field issue then routes the call
        through the normal correction loop."""
        settings = self._model_settings()
        settings['extra_body'] = {**(settings.get('extra_body') or {}),
                                  'response_format': {'type': 'json_object'}}
        prompt = _tail(instructions, scope, feedback) + _output_block(result_schema)
        messages = history
        for _ in range(2):
            result = await self._raw_run(agent, prompt, messages, settings)
            if (data := _relay(result.output, result_schema)) is not None:
                return AgentResult(data=data, history=result.all_messages())
            prompt = ('Your previous reply was not usable JSON of the requested '
                      'shape. Reply again with a single JSON value — an object '
                      'carrying every required key, or an array of such objects — '
                      'matching the JSON Schema in the context.')
            messages = result.all_messages()
        return AgentResult(data=None, history=result.all_messages())

    async def _run_once(self, agent, instructions, scope, feedback, history):
        result = await self._raw_run(agent, _tail(instructions, scope, feedback), history)
        return AgentResult(data=result.output, history=result.all_messages())

    async def _raw_run(self, agent, prompt, history, settings=None):
        result = await agent.run(prompt, message_history=history,
                                 model_settings=settings or self._model_settings())
        await report_tokens(result.usage.input_tokens + result.usage.output_tokens)
        return result

    def _model_settings(self):
        settings = dict(self.model_settings)
        if _THINKING_OFF.match(getattr(self.model, 'model_name', '')):
            settings['extra_body'] = {'enable_thinking': False}
        return settings


def enable_tracing() -> None:
    """Turn on the framework's native OTel span emission for every agent
    (process-wide). Every adapter in this boundary provides this hook so
    hosts enable tracing without naming the framework."""
    Agent.instrument_all(True)


def _json_body(body: dict) -> bytes:
    """SDK serialization style — ensure_ascii would double CJK bytes."""
    return json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()


def _stripped_headers(headers: httpx.Headers) -> httpx.Headers:
    """Headers for a rewritten body: the length/encoding headers are stale."""
    headers = headers.copy()
    headers.pop('content-length', None)
    headers.pop('content-encoding', None)
    return headers


def marked_request(request: httpx.Request) -> httpx.Request:
    """Rewrite a chat-completions request so its leading string-content
    system message becomes a cache-marked content block (dashscope's
    ``cache_control`` explicit-cache extension). The marker must close
    the shared payload at a message boundary — marking mid-message does
    not match. Non-matching shapes pass through untouched."""
    body = json.loads(request.read())
    messages = body.get('messages')
    if (messages and messages[0].get('role') == 'system'
            and isinstance(messages[0].get('content'), str)):
        messages[0]['content'] = [{'type': 'text', 'text': messages[0]['content'],
                                   'cache_control': {'type': 'ephemeral'}}]
        return httpx.Request(request.method, request.url,
                             headers=_stripped_headers(request.headers),
                             content=_json_body(body))
    return request


def _unthinking(request: httpx.Request) -> httpx.Request:
    """Drops ``enable_thinking: false`` into a chat-completions body."""
    body = json.loads(request.read())
    return httpx.Request(request.method, request.url,
                         headers=_stripped_headers(request.headers),
                         content=_json_body(body | {'enable_thinking': False}))


def _rename_cache_write(response: httpx.Response) -> httpx.Response:
    """Relabel dashscope's cache-write usage field to the OpenAI name
    pydantic-ai reads (``prompt_tokens_details.cache_write_tokens``);
    without the relabel the count never reaches the usage object or
    the span."""
    body = response.content
    if b'"cache_creation_input_tokens"' not in body:
        return response
    return httpx.Response(response.status_code,
                          headers=_stripped_headers(response.headers),
                          content=body.replace(b'"cache_creation_input_tokens"',
                                               b'"cache_write_tokens"'))


class _WireTransport(httpx.AsyncHTTPTransport):
    """Applies provider wire transforms — requests before they leave,
    chat-completion responses as they return; the openai client beneath
    is untouched."""

    def __init__(self, transforms: list):
        super().__init__()  # the base owns the connection pool
        self.transforms = transforms

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith('/chat/completions'):
            for transform in self.transforms:
                request = transform(request)
            response = await super().handle_async_request(request)
            if 'application/json' in response.headers.get('content-type', ''):
                await response.aread()  # a rewritable body must be read first
                return _rename_cache_write(response)
            return response
        return await super().handle_async_request(request)


def openai_model(base_url: str, api_key: str, model: str, *, explicit_cache: bool = False,
                 thinking_off: bool = False,
                 http_client: httpx.AsyncClient | None = None):
    """A pydantic-ai Model for any OpenAI-compatible endpoint — the
    provider mechanism lives inside the agents boundary; hosts pass
    only deployment config. Requires the ``docspectra[pydantic-ai]`` extra.
    ``explicit_cache`` marks the leading system message as a provider
    cache block (dashscope explicit cache): paired with a runner that
    puts the shared payload in Agent instructions, the call that runs
    first pays cache creation once and every call behind it hits; the
    wire transport relabels dashscope's non-standard cache-write usage
    field so the count also lands on the span. Providers that reject
    the extension must leave it off.
    ``thinking_off`` drops ``enable_thinking: false`` into every request
    — for consumers that build their own pydantic-ai agents outside the
    runner's model_settings (pydantic-evals LLMJudge: tools +
    tool_choice='required', which dashscope rejects while thinking).
    A host ``http_client`` (proxies, connection tuning) is used
    verbatim."""
    try:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
    except ImportError as e:
        raise ImportError('install the pydantic-ai extra: docspectra[pydantic-ai]') from e
    if transforms := ([marked_request] if explicit_cache else []) \
            + ([_unthinking] if thinking_off else []):
        http_client = httpx.AsyncClient(transport=_WireTransport(transforms))
    return OpenAIChatModel(model, provider=OpenAIProvider(
        base_url=base_url, api_key=api_key, http_client=http_client))


def _wrap(schema: JSONSchema) -> JSONSchema:
    return {'type': 'object', 'properties': {'items': schema},
            'required': ['items'], 'additionalProperties': False}


def _relay(text: str, schema: JSONSchema):
    """A json_object reply → the promised result shape, or None when the
    text is not usable JSON of that shape. Fenced code blocks and one
    single-key wrapper the model may add around the value are
    tolerated; only the top-level shape is checked."""
    if (obj := _load_json(text)) is None:
        return None
    obj = _unwrap(obj, schema)
    if (isinstance(obj, list) and schema.get('type') == 'object'
            and len(obj) == 1 and isinstance(obj[0], dict)):
        obj = obj[0]  # a per-item reply wrapped in a one-element array
    want = schema.get('type')
    if isinstance(obj, dict):
        return obj if want in (None, 'object') and \
            all(k in obj for k in schema.get('required') or ()) else None
    if isinstance(obj, list):
        return obj if want in (None, 'array') else None
    return obj if want == 'string' else None


def _load_json(text: str):
    s = text.strip()
    if s.startswith('```'):  # a fenced reply: take the block's body
        s = s.split('\n', 1)[1] if '\n' in s else s
        s = s.rsplit('```', 1)[0]
    try:
        return json.loads(s)
    except ValueError:
        return None


def _unwrap(obj, schema: JSONSchema):
    """Peel single-key wrappers the model may add — a dict that carries
    none of the schema's required keys and holds one dict/list value."""
    required = set(schema.get('required') or ())
    for _ in range(3):
        if (isinstance(obj, dict) and len(obj) == 1 and not (required & obj.keys())
                and isinstance((inner := next(iter(obj.values()))), (dict, list))):
            obj = inner
        else:
            return obj
    return obj


def _tail(instructions: str, scope, feedback) -> str:
    parts = [instructions] if instructions else []
    if scope is not None:
        parts.append(f'\n\n---\nAssigned material:\n{scope}')
    if feedback:
        lines = '\n'.join(f'- {i.path}: {i.message}' for i in feedback)
        parts.append(f'\n\n---\nValidation errors to fix:\n{lines}')
    return ''.join(parts)


def _output_block(schema: JSONSchema) -> str:
    """What the json_object channel cannot enforce on the wire: this
    call's expected shape. The payload's schema shows the whole
    document, so a per-item call there reads as an array — the
    directive pins the top level, and the partial schema itself is
    restated beside it so the model need not locate its subschema
    inside the shared payload (a full-document shape was measured to
    override the item expectation entirely)."""
    if schema.get('type') == 'array':
        head = ('Output: one JSON array — every item of this unit, '
                'each an object with all required keys.')
    else:
        head = ('Output: one JSON object — the assigned item alone, '
                'with all required keys.')
    return (f'\n\n---\n{head}\nOutput schema of this call:\n'
            + json.dumps(schema, ensure_ascii=False))
