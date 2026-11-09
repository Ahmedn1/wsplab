"""Guard prompt controls against annotation leakage and incorrect source mapping."""
import copy

import pytest
import finqa_controls as controls
from finqa_screen import prompt


def row():
    return {'id':'sample','pre_text':['Context.', 'Relevant source.'],'post_text':['Unrelated source.'],
            'table_ori':[['','2020','2019'],['Measure','Value','Value'],
                         ['Revenue<sup>a</sup>','30','20'],['Cost','10','8']],
            'table':[['Measure','2020 Value','2019 Value'],['revenue (a)','30','20'],['cost','10','8']],
            'qa':{'question':'What was revenue in 2020?','answer':'30','program':'secret',
                  'gold_inds':{'table_1':'Never put this generated annotation in a prompt.', 'text_1':'Relevant source.'}}}


def test_full_control_changes_only_illustrative_sentence():
    item=row()
    expected=prompt(item).replace(controls.OLD, controls.NEW)
    assert controls.make_prompt(item, controls.CONDITIONS['clean_thinking']) == expected
    changed=copy.deepcopy(item)
    changed['qa']['answer']='different'
    changed['qa']['program']='different'
    changed['qa']['gold_inds']={}
    assert controls.make_prompt(changed, controls.CONDITIONS['clean_thinking']) == expected


def test_evidence_preserves_headers_and_original_cells_without_gold_rendering():
    item=row()
    selected,mapping=controls.select_evidence(item)
    assert selected['table_ori'] == item['table_ori'][:3]
    assert selected['pre_text'] == ['Relevant source.']
    assert selected['post_text'] == []
    result=controls.make_prompt(item,controls.CONDITIONS['evidence_thinking'])
    assert 'Revenue<sup>a</sup> | 30 | 20' in result
    assert 'secret' not in result
    assert 'Never put' not in result
    assert mapping['original_table_indices']==[0,1,2]
    assert item['table_ori'][-1][0]=='Cost'  # Input not mutated.


@pytest.mark.parametrize('kind',['text','table'])
def test_misaligned_annotations_abort_instead_of_guessing(kind):
    item=row()
    if kind=='text':item['qa']['gold_inds']['text_1']='Wrong source text'
    else:item['table'][1][0]='Wrong row'
    with pytest.raises(ValueError):controls.select_evidence(item)


def test_paired_report_requires_all_matched_items():
    left=[{'id':'a','correct':False},{'id':'b','correct':True}]
    right=[{'id':'a','correct':True}]
    assert controls.paired(left,right,['a','b'])=={'status':'incomplete'}
    right.append({'id':'b','correct':True})
    result=controls.paired(left,right,['a','b'])
    assert result['difference']==.5 and result['wrong_to_right']==1 and result['right_to_wrong']==0
