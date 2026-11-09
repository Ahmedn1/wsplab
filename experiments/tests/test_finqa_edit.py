"""Protect the frozen prompt/scoring and report-based edit fixture boundaries."""
import copy

import pytest
from finqa_edit import FinqaGrader, convert, first_per_report
from finqa_screen import prompt


def row(report, suffix='1', answer='12.5%'):
    return {'id': f'{report}/page_1.pdf-{suffix}', 'pre_text': ['Source'], 'post_text': [],
            'table_ori': [['Year', 'Value'], ['2020', '25']],
            'qa': {'question': 'How much?', 'answer': answer, 'program': 'hidden program',
                   'gold_inds': {'text_0': 'hidden annotation'}}}


def test_edit_fixture_preserves_prompt_and_omits_hidden_fields():
    item = row('A/2020')
    converted = convert(item)
    assert converted['problem'] == prompt(item)
    assert set(converted) == {'id', 'problem', 'answer', 'source'}
    changed = copy.deepcopy(item)
    changed['qa'].update(program='different', gold_inds={}, answer='999')
    assert convert(changed)['problem'] == converted['problem']


def test_report_selection_stays_in_partition_and_source_order():
    rows = [row('EVAL/2020'), row('A/2020', answer=''), row('B/2020'), row('A/2020', '2')]
    chosen = first_per_report(rows, ['A/2020', 'B/2020'], numeric_only=True)
    assert [r['id'] for r in chosen] == ['A/2020/page_1.pdf-2', 'B/2020/page_1.pdf-1']
    assert first_per_report(rows, ['A/2020'])[0]['qa']['answer'] == ''
    with pytest.raises(ValueError, match='Missing'):
        first_per_report(rows, ['MISSING/2020'])
    with pytest.raises(ValueError, match='Duplicate'):
        first_per_report(rows, ['A/2020', 'A/2020'])


def test_edit_grader_keeps_units_and_requires_finished_thinking():
    grader = FinqaGrader()
    target = {'answer': '12.5%'}
    assert grader.grade('Calculation</think>\\boxed{12.5\\%}', target).correct
    assert not grader.grade('Calculation</think>\\boxed{0.125}', target).correct
    assert not grader.grade('<think>\\boxed{12.5}', target).correct
    assert not grader.grade('Calculation</think>\\boxed{12.5} or \\boxed{13}', target).correct
    with pytest.raises(ValueError, match='Unsupported'):
        convert(row('A/2020', answer=''))
