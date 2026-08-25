"""Boundary probes for the parse service wiring."""

from xtremeparse import TaskScheduler
from xtremeparse.contracts import AgentResult

from docspectra.service.parse import GenericIssue, ParseService, schema_validator
from tests.helpers import ScriptedRunner, agent_result

SCHEMA = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string', 'description': '姓名'},
        'jobs': {'type': 'array', 'description': '经历', 'items': {
            'type': 'object', 'properties': {'company': {'type': 'string'}},
            'required': ['company']}},
    },
    'required': ['name'],
}


async def test_file_to_json_round_trip(tmp_path):
    file = tmp_path / 'resume.txt'
    file.write_text('姓名：张三', encoding='utf-8')
    runner = ScriptedRunner(
        agent_result('0 b\na: 0'),
        agent_result('0 b\na: 0'),  # zero-verification round
        agent_result({'name': '张三'}),
    )
    result = await ParseService(runner).parse(file, SCHEMA)
    assert result.data == {'name': '张三'} and result.issues == []
    assert runner.calls[0]['content'].startswith('姓名：张三')


async def test_validator_reports_bracketed_array_paths():
    issues = schema_validator(SCHEMA)({'name': 1, 'jobs': [{'company': 'ok'}, {}]})
    by_path = {i.path: i for i in issues}
    assert sorted(by_path) == ['jobs[1]', 'name']  # required-error path sits on the item
    assert 'company' in by_path['jobs[1]'].message
    assert all(isinstance(i, GenericIssue) for i in issues)


async def test_validator_clean_data_yields_no_issues():
    assert schema_validator(SCHEMA)({'name': '张三'}) == []


def test_empty_path_renders_root():
    issue = schema_validator({'type': 'object'})('not-a-dict')[0]
    assert issue.path == '$'


def test_scheduler_is_forwarded_to_the_extractor():
    scheduler = TaskScheduler(4)
    service = ParseService(ScriptedRunner(), scheduler=scheduler)
    assert service._extractor.scheduler is scheduler
