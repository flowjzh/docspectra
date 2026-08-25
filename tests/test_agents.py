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


async def test_wire_layout_and_native_output():
    """The wire contract cache reuse depends on: payload alone in the
    leading system message, private matter in the user message, schema
    as response_format (tools would displace the prefix)."""
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
    await runner.run(instructions='THE CARD', result_schema=ARR_SCHEMA,
                     content='SHARED PAYLOAD', scope='SCOPE')
    body = bodies[0]
    assert [(m['role'], m['content']) for m in body['messages']] == [
        ('system', 'SHARED PAYLOAD'), ('user', 'THE CARD\n\n---\nAssigned material:\nSCOPE')]
    assert body.get('tools') is None
    assert body['response_format']['json_schema']['strict'] is True  # schema stays out of the prompt
    schema = body['response_format']['json_schema']['schema']
    assert schema['properties']['items'] == ARR_SCHEMA  # array wrapped for native output


def test_marked_request_marks_leading_system_payload():
    body = {'model': 'm', 'messages': [
        {'role': 'system', 'content': 'SHARED PAYLOAD'},
        {'role': 'user', 'content': 'THE CARD'}]}
    request = marked_request(_request(body))
    marked = json.loads(request.read().decode())['messages'][0]
    assert marked['content'] == [{'type': 'text', 'text': 'SHARED PAYLOAD',
                                  'cache_control': {'type': 'ephemeral'}}]
    assert request.headers['content-length'] == str(len(request.content))


def test_marked_request_passes_other_shapes_through():
    for messages in ([{'role': 'user', 'content': 'hi'}],
                     [{'role': 'system', 'content': [{'type': 'text', 'text': 'already blocks'}]}]):
        request = _request({'model': 'm', 'messages': messages})
        assert marked_request(request) is request


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


def _request(body):
    return httpx.Request('POST', 'http://x/v1/chat/completions',
                         json=body, headers={'authorization': 'Bearer k'})
