"""Correct a presentation-only parser bug without changing any generated answer."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

if __package__:
    from .finqa_screen import grade
    from .hotpot_screen import digest, write_json
else:
    from finqa_screen import grade
    from hotpot_screen import digest, write_json


def rescore(root):
    plan=json.loads((root/'plan.json').read_text())
    status=json.loads((root/'run/status.json').read_text())['status']
    result={'status':status,'at':time.time(),'paid_api_calls':0,
            'correction':'Normalize LaTeX escaped percent/currency and thin-space formatting; no unit rescaling or answer changes.',
            'metric':'custom direct numeric answer accuracy, not official FinQA execution accuracy',
            'scorer_sha256':digest(Path(__file__).with_name('finqa_screen.py')),
            'conditions':{},'source_sha256':{}}
    for name in ('nonthinking','thinking'):
        path=root/'run'/f'{name}.jsonl'
        raw=[json.loads(line) for line in path.read_bytes().split(b'\n')[:-1]] if path.exists() else []
        ids=[r['id'] for r in raw]
        if len(set(ids))!=len(ids) or not set(ids)<=set(plan['pilot_ids']):
            raise ValueError('Foreign or duplicated responses')
        corrected=[]
        for row in raw:
            # Older unfinished-thinking records do not repeat their human gold.
            gold=row.get('gold')
            if gold is None:
                pilot=[json.loads(s) for s in (root/'pilot.jsonl').read_text().splitlines()]
                gold=next(r['qa']['answer'] for r in pilot if r['id']==row['id'])
            new=grade(row['response'],name=='thinking',gold)
            corrected.append({**row,'original_correct':row['correct'],'original_parsed':row['parsed'],**new})
        output=root/'run'/f'{name}-rescored.jsonl'
        tmp=output.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in corrected));tmp.replace(output)
        result['conditions'][name]={'completed':len(corrected),'correct':sum(r['correct'] for r in corrected),
            'accuracy':statistics.mean(r['correct'] for r in corrected) if corrected else None,
            'unparseable':sum(not r['parsed'] for r in corrected),'truncated':sum(r['truncated'] for r in corrected),
            'original_correct':sum(r['original_correct'] for r in corrected),
            'changed_correctness_ids':[r['id'] for r in corrected if r['correct']!=r['original_correct']]}
        if path.exists():result['source_sha256'][name]=digest(path)
    values=list(result['conditions'].values())
    if all(v['completed']==32 for v in values):
        passed=max(v['accuracy'] for v in values)<=.4 and all(v['truncated']/32<=.1 and v['unparseable']/32<=.1 for v in values)
        result['screening_rule']='passes_manual_review_required' if passed else 'not_qualified_by_frozen_rule'
    else:result['screening_rule']='incomplete'
    write_json(root/'run/rescored-summary.json',result)
    destination=Path('experiments/08_finqa/natural-finqa-rescored-results')
    write_json(destination.with_suffix('.json'),result)
    lines=['# FinQA pilot: corrected numeric parsing','',f'Status: **{status}**. Rule: **{result["screening_rule"]}**.','',
        'Identical generated responses, with LaTeX presentation escapes normalized. Original responses and scores are preserved. No new inference, unit rescaling, changed labels, or paid calls.','',
        'This is custom direct-answer accuracy, not official FinQA program/execution accuracy.','',
        '| Condition | Completed | Correct | Accuracy | Unparseable | Truncated |','|---|---:|---:|---:|---:|---:|']
    for name,v in result['conditions'].items():
        acc=f'{100*v["accuracy"]:.1f}%' if v['completed'] else 'pending'
        lines.append(f'| {name} | {v["completed"]}/32 | {v["correct"]} | {acc} | {v["unparseable"]} | {v["truncated"]} |')
    lines+=['','See [the frozen plan and limitations](natural-finqa-plan.md). Incomplete conditions are progress only.','']
    destination.with_suffix('.md').write_text('\n'.join(lines))
    if status in {'completed','failed','stopped_time_limit'}:
        raw_report=Path('experiments/08_finqa/natural-finqa-results.md')
        note='**Parser correction:** use [the rescored results](natural-finqa-rescored-results.md). The table below preserves original scores before LaTeX escape normalization.\n\n'
        if raw_report.exists() and not raw_report.read_text().startswith('**Parser correction:**'):
            raw_report.write_text(note+raw_report.read_text())
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.wsplab/natural-finqa'))
    parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    deadline=time.monotonic()+3.5*3600
    while True:
        result=rescore(args.root)
        if not args.watch or result['status'] in {'completed','failed','stopped_time_limit'} or time.monotonic()>deadline:
            break
        time.sleep(15)
