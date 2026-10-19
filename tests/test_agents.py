"""Offline adapter tests: PydanticAI runner via FunctionModel."""

import json

import httpx
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from docspectra.agents import (
    PydanticAIRunner,
    _rename_cache_write,
    marked_request,
    openai_model,
)
from tests.helpers import FakeIssue

OBJ_SCHEMA = {'type': 'object', 'properties': {'name': {'type': 'string'}},
              'required': ['name'], 'additionalProperties': False}
ARR_SCHEMA = {'type': 'array', 'items': OBJ_SCHEMA}

def capturing_model(payload: str, seen: list):
    def generate(messages, info):
        seen.append(messages[-1].parts[0].content)
        return ModelResponse(parts=[TextPart(content=payload)])

    return FunctionModel(generate)


async def test_object_schema_round_trips():
    seen = []
    runner = PydanticAIRunner(capturing_model('{"name": "张三"}', seen))
    result = await runner.run(instructions='card', result_schema=OBJ_SCHEMA, content='全文')
    assert result.data == {'name': '张三'}
    assert result.history  # all_messages for multi-turn continuation


async def test_array_schema_is_wrapped_and_unwrapped():
    runner = PydanticAIRunner(capturing_model('{"items": [{"name": "a"}]}', []))
    result = await runner.run(instructions='card', result_schema=ARR_SCHEMA, content='全文',
                              scope='材料')
    assert result.data == [{'name': 'a'}]


def test_thinking_override():
    """Qwen numeric models default to thinking off for extraction; the
    judge opts back in with thinking=True (no enable_thinking key), and
    another model family keeps the provider default."""
    def fake(messages, info):
        return ModelResponse(parts=[TextPart(content='{}')])

    qwen = FunctionModel(fake, model_name='qwen3.7-max')
    off = PydanticAIRunner(qwen)._model_settings()
    assert off['extra_body'] == {'enable_thinking': False}
    on = PydanticAIRunner(qwen, thinking=True)._model_settings()
    assert 'extra_body' not in on
    forced = PydanticAIRunner(qwen, thinking=False)._model_settings()
    assert forced['extra_body'] == {'enable_thinking': False}
    other = PydanticAIRunner(FunctionModel(fake, model_name='gpt-x'))._model_settings()
    assert 'extra_body' not in other


async def test_private_matter_trails_in_the_user_message():
    seen = []
    runner = PydanticAIRunner(capturing_model('{"name": "x"}', seen))
    await runner.run(instructions='THE CARD', result_schema=OBJ_SCHEMA,
                     content='SHARED PREFIX', scope='SCOPE TEXT',
                     feedback=[FakeIssue('name', 'required')])
    text = seen[-1]
    assert 'SHARED PREFIX' not in text  # payload rides its own system message
    assert text.startswith('THE CARD')
    assert text.index('SCOPE TEXT') > text.index('THE CARD')
    assert '- name: required' in text and text.index('- name: required') > text.index('SCOPE TEXT')


def usage_model(payload: str, usage: RequestUsage):
    return FunctionModel(lambda messages, info: ModelResponse(
        parts=[TextPart(content=payload)], usage=usage))


async def test_usage_reported_through_the_scheduling_boundary(monkeypatch):
    reported = []

    async def fake_report(actual):
        reported.append(actual)

    monkeypatch.setattr('docspectra.agents.report_tokens', fake_report)
    runner = PydanticAIRunner(usage_model('{"name": "张三"}',
                                          RequestUsage(input_tokens=11, output_tokens=7)))
    result = await runner.run(instructions='card', result_schema=OBJ_SCHEMA, content='全文')
    assert reported == [18]  # input + output of the one call
    assert result.data == {'name': '张三'}


async def test_wire_layout_and_json_object_channel():
    """The wire contract cache reuse depends on: payload alone in the
    leading system message, private matter in the user message, the
    structured channel as bare json_object — no schema on the wire
    (json_schema's grammar collapses long-material arrays under a
    non-thinking decode; the schema rides the prompt instead)."""
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.read().decode()))
        return httpx.Response(200, json={
            'id': 'x', 'object': 'chat.completion', 'created': 0, 'model': 'm',
            'choices': [{'index': 0, 'finish_reason': 'stop',
                         'message': {'role': 'assistant', 'content': '{"name": "x"}'}}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})

    model = openai_model('http://stub/v1', 'x', 'm',
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    runner = PydanticAIRunner(model)
    await runner.run(instructions='THE CARD', result_schema=OBJ_SCHEMA,
                     content='SHARED PAYLOAD', scope='SCOPE')
    body = bodies[0]
    user = body['messages'][1]['content']
    assert [(m['role']) for m in body['messages']] == ['system', 'user']
    assert body['messages'][0]['content'] == 'SHARED PAYLOAD'
    assert user.startswith('THE CARD')
    assert 'Assigned material:\nSCOPE' in user
    # the per-call output block: top-level directive + the partial
    # schema restated beside the payload's full-document schema
    assert 'the assigned item alone' in user
    assert json.dumps(OBJ_SCHEMA, ensure_ascii=False) in user
    assert body.get('tools') is None
    assert body['response_format'] == {'type': 'json_object'}

    legacy = PydanticAIRunner(model, json_object=False)
    await legacy.run(instructions='THE CARD', result_schema=ARR_SCHEMA,
                     content='SHARED PAYLOAD', scope='SCOPE')
    wire = bodies[1]['response_format']
    assert wire['json_schema']['strict'] is True
    assert wire['json_schema']['schema']['properties']['items'] == ARR_SCHEMA


async def test_json_object_reply_tolerates_fences_and_wrappers():
    runner = PydanticAIRunner(capturing_model(
        '```json\n{"work_experiences": [{"name": "a"}]}\n```', []))
    result = await runner.run(instructions='card', result_schema=ARR_SCHEMA, content='全文')
    assert result.data == [{'name': 'a'}]


async def test_json_object_reply_wraps_a_per_item_answer_in_an_array():
    # a per-item call (object schema) whose reply came wrapped in a
    # one-element array under the unit's name
    runner = PydanticAIRunner(capturing_model('{"work_experiences": [{"name": "a"}]}', []))
    result = await runner.run(instructions='card', result_schema=OBJ_SCHEMA, content='全文')
    assert result.data == {'name': 'a'}


async def test_unusable_reply_gets_one_repair_round():
    seen = []

    def generate(messages, info):
        seen.append(messages[-1].parts[0].content)
        return ModelResponse(parts=[TextPart(
            content='not json' if len(seen) == 1 else '{"name": "ok"}')])

    runner = PydanticAIRunner(FunctionModel(generate))
    result = await runner.run(instructions='card', result_schema=OBJ_SCHEMA, content='全文')
    assert result.data == {'name': 'ok'}
    assert 'not usable JSON' in seen[1]


async def test_two_bad_replies_yield_none_data():
    runner = PydanticAIRunner(capturing_model('still not json', []))
    result = await runner.run(instructions='card', result_schema=OBJ_SCHEMA, content='全文')
    assert result.data is None


def test_marked_request_marks_leading_system_payload():
    body = {'model': 'm', 'messages': [
        {'role': 'system', 'content': 'SHARED PAYLOAD'},
        {'role': 'user', 'content': 'THE CARD'}]}
    assert marked_request(body) is True
    assert body['messages'][0]['content'] == [
        {'type': 'text', 'text': 'SHARED PAYLOAD',
         'cache_control': {'type': 'ephemeral'}}]


def test_marked_request_marks_the_shared_prefix_user_message_too():
    body = {'model': 'm', 'messages': [
        {'role': 'system', 'content': 'SHARED PAYLOAD'},
        {'role': 'user', 'content': 'SEGMENTS'},
        {'role': 'assistant', 'content': '0-9 a.0'},
        {'role': 'user', 'content': 'repair feedback'}]}
    assert marked_request(body) is True
    messages = body['messages']
    assert [m['content'][0]['text'] for m in messages[:2]] == ['SHARED PAYLOAD', 'SEGMENTS']
    assert all(m['content'][0]['cache_control'] == {'type': 'ephemeral'} for m in messages[:2])
    assert messages[2]['content'] == '0-9 a.0'
    assert messages[3]['content'] == 'repair feedback'


def test_marked_request_passes_other_shapes_through():
    for messages in ([{'role': 'user', 'content': 'hi'}],
                     [{'role': 'system', 'content': [{'type': 'text', 'text': 'already blocks'}]}]):
        body = {'model': 'm', 'messages': messages}
        assert marked_request(body) is False


def test_rename_cache_write_relabels_dashscope_usage_field():
    response = _rename_cache_write(httpx.Response(200, json={'usage': {
        'prompt_tokens': 5,
        'prompt_tokens_details': {'cache_creation_input_tokens': 5,
                                  'cached_tokens': 0}}}))
    details = json.loads(response.content.decode())['usage']['prompt_tokens_details']
    assert details == {'cache_write_tokens': 5, 'cached_tokens': 0}
    assert response.headers['content-length'] == str(len(response.content))


def test_rename_cache_write_passes_plain_usage_through():
    response = _rename_cache_write(httpx.Response(200, json={'usage': {'prompt_tokens': 5}}))
    assert json.loads(response.content.decode())['usage'] == {'prompt_tokens': 5}
