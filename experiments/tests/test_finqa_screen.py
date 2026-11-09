"""Check numerical units and separation of benchmark inputs from gold annotations."""
import copy

import finqa_screen as screen


def test_percent_units_are_not_silently_rescaled():
    assert screen.grade(r'\boxed{12.524%}', False, '12.52%')['correct']
    assert screen.grade(r'\boxed{12.524\%}', False, '12.52%')['correct']
    assert not screen.grade(r'\boxed{0.12524\%}', False, '12.52%')['correct']
    assert not screen.grade(r'\boxed{0.12524}', False, '12.52%')['correct']
    assert screen.grade(r'\boxed{38.9%}', False, '39%')['correct']
    assert not screen.grade(r'\boxed{38%}', False, '39%')['correct']


def test_human_display_precision_and_exact_integer_answers():
    assert screen.grade(r'\boxed{18.57143}', False, '18.6')['correct']
    assert not screen.grade(r'\boxed{18.4}', False, '18.6')['correct']
    assert not screen.grade(r'\boxed{1739.4}', False, '1739')['correct']
    assert screen.grade(r'\boxed{40,444,920}', False, '$ 40444920\\n')['correct']


def test_unfinished_thinking_and_multiple_numbers_cannot_match():
    assert not screen.grade(r'\boxed{12.52%}', True, '12.52%')['correct']
    assert screen.grade(r'\boxed{99}</think>\boxed{12.52%}', True, '12.52%')['correct']
    assert not screen.grade('12.52 or 99', False, '12.52%')['parsed']
    assert not screen.grade(r'\boxed{12.52}\boxed{99}', False, '12.52%')['parsed']


def test_prompt_ignores_all_gold_and_retriever_fields():
    row={'pre_text':['Original passage.'],'post_text':['End.'],'table_ori':[['Year','2016','2015'],['Profit','20','10']],
         'table':[['Corrupted headers']], 'qa':{'question':'What was profit in 2016?','answer':'20','program':'secret',
             'gold_inds':{'text_0':'secret'},'model_input':['secret']},'model_input':['secret']}
    original=screen.prompt(row)
    altered=copy.deepcopy(row)
    altered['qa']={'question':row['qa']['question'],'answer':'other','program':'other',
                   'gold_inds':{'text_0':'other'},'model_input':['other']}
    altered['table']=[['Different normalized table']]
    altered['model_input']=['other']
    assert screen.prompt(altered)==original
    assert 'Year | 2016 | 2015' in original
    assert 'secret' not in original
