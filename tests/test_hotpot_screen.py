"""Protect the benchmark from gold leakage and accidental reasoning credit."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('hotpot_screen', Path(__file__).parents[1] / 'scripts/live/hotpot_screen.py')
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def example():
    return {'id': 'example', 'question': 'Where was the author of Blue Book born?',
            'answer': 'Secret City', 'context': {'title': ['Blue Book', 'A Writer', 'Other', 'Novel', 'Town'],
            'sentences': [['Blue Book was written by A Writer.'], ['A Writer was born in Secret City.'],
                          ['Something unrelated.'], ['Another novel by another author.'], ['A town far away.']]},
            'supporting_facts': {'title': ['Blue Book', 'A Writer'], 'sent_id': [0, 0]}}


@pytest.mark.parametrize('mode', ['full', 'bm25'])
def test_deployable_prompts_ignore_answer_and_support_labels(mode):
    row = example()
    changed = copy.deepcopy(row)
    changed['answer'] = 'A different gold answer'
    changed['supporting_facts'] = {'title': ['Other'], 'sent_id': [999]}
    assert screen.prompt(row, mode) == screen.prompt(changed, mode)
    assert screen.prompt(row, mode).endswith('Question: ' + row['question'])


def test_oracle_keeps_whole_paragraphs_not_just_answer_sentences():
    row = example()
    row['context']['sentences'][1].append('An additional sentence not marked as supporting.')
    result = screen.paragraphs(row, 'oracle')
    assert len(result) == 2
    assert result[1][1][-1].startswith('An additional sentence')
    assert len(screen.paragraphs(row, 'full')) == 5
    assert len(screen.paragraphs(row, 'bm25')) == 4


def test_unfinished_thinking_cannot_receive_answer_credit():
    assert screen.extract('<answer>Correct but only in thought</answer>', True) == ('', False, True)
    text = '<answer>Wrong</answer></think><answer>Right</answer>'
    assert screen.extract(text, True) == ('Right', True, False)


def test_multiple_answer_tags_are_not_cherry_picked():
    text = '<answer>Wrong</answer><answer>Right</answer>'
    assert screen.extract(text, False) == (text, False, False)


def test_resume_rejects_duplicate_and_foreign_items(tmp_path):
    path = tmp_path / 'responses.jsonl'
    for rows in ([{'id': 'a'}, {'id': 'a'}], [{'id': 'foreign'}]):
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        with pytest.raises(ValueError):
            screen.read_records(path, ['a'])
