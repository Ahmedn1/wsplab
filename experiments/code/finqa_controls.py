"""Frozen local FinQA prompt and annotated-evidence diagnostics; no weight edits."""
from __future__ import annotations

import argparse
import copy
import html
import json
import math
import os
import re
import statistics
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

if __package__:
    from .finqa_screen import CONTEXT, SEED, grade, verify
    from .finqa_screen import prompt as original_prompt
    from .hotpot_screen import digest, read_records, write_json
else:
    from finqa_screen import CONTEXT, SEED, grade, verify
    from finqa_screen import prompt as original_prompt
    from hotpot_screen import digest, read_records, write_json

CONDITIONS = {
    'clean_nonthinking': {'thinking': False, 'evidence': False},
    'clean_thinking': {'thinking': True, 'evidence': False},
    'evidence_thinking': {'thinking': True, 'evidence': True},
}
OLD = 'For a percentage, give percentage units (for example 12.5%, not 0.125).'
NEW = 'For a percentage, give percentage units rather than a fraction.'


def canonical(text):
    text = html.unescape(re.sub(r'<[^>]+>', '', text))
    return re.sub(r'[^a-z0-9]', '', text.lower())


def select_evidence(row):
    """Map annotation IDs to unchanged source rows; never feed gold rendering/program."""
    table, normalized = row['table_ori'], row['table']
    offset = len(table) - len(normalized)
    if offset < 0:
        raise ValueError('Cannot align original and normalized tables')
    # Conservative prefix retention preserves multirow headers. If a prefix
    # contains a data row as well, retain it rather than guessing it is dispensable.
    indices = set(range(offset + 1))
    text_ids = set()
    source_text = row['pre_text'] + row['post_text']
    for key, annotation in row['qa']['gold_inds'].items():
        kind, number = key.split('_')
        index = int(number)
        if kind == 'table':
            actual = index + offset
            if not 0 <= index < len(normalized) or not 0 <= actual < len(table):
                raise ValueError('Annotated table index out of range')
            if canonical(normalized[index][0]) != canonical(table[actual][0]):
                raise ValueError('Original/normalized row-label mismatch')
            indices.add(actual)
        elif kind == 'text':
            if not 0 <= index < len(source_text) or canonical(annotation) != canonical(source_text[index]):
                raise ValueError('Annotated text does not match source')
            text_ids.add(index)
        else:
            raise ValueError('Unknown annotation type')
    result = copy.deepcopy(row)
    result['table_ori'] = [table[i] for i in sorted(indices)]
    result['pre_text'] = [t for i, t in enumerate(row['pre_text']) if i in text_ids]
    result['post_text'] = [t for i, t in enumerate(row['post_text'], start=len(row['pre_text'])) if i in text_ids]
    mapping = {'original_table_indices': sorted(indices), 'source_text_indices': sorted(text_ids),
               'header_offset': offset, 'unchanged_source_content': True}
    return result, mapping


def make_prompt(row, condition):
    selected = select_evidence(row)[0] if condition['evidence'] else row
    text = original_prompt(selected)
    if text.count(OLD) != 1:
        raise ValueError('Original prompt changed')
    return text.replace(OLD, NEW)


def prepare(root):
    baseline = verify(root)
    folder = root/'controls'
    folder.mkdir(exist_ok=True)
    audit = json.loads((root/'audit.json').read_text())
    rows = [json.loads(s) for s in (root/'pilot.jsonl').read_text().splitlines()]
    mappings = {r['id']: select_evidence(r)[1] for r in rows}
    manifest = {'baseline_plan_sha256': digest(root/'plan.json'), 'audit_sha256': digest(root/'audit.json'),
                'baseline_responses_sha256': {n: digest(root/'run'/f'{n}-rescored.jsonl') for n in ('nonthinking','thinking')},
                'conditions': CONDITIONS, 'seed': SEED, 'context': CONTEXT,
                'pilot_ids': baseline['pilot_ids'], 'clean_ids': audit['clean_ids'],
                'sensitive_ids': audit['sensitive_ids'], 'mappings': mappings,
                'paid_api_calls': 0, 'external_evaluation_used': False,
                'interpretation': 'All-32 original-label scores primary; pre-flagged 24-item sensitivity secondary, post hoc and not blinded. Evidence condition uses gold labels and is diagnostic only.'}
    target = folder/'plan.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('Frozen diagnostic inputs changed')
    write_json(target, manifest)
    return baseline, manifest, rows


def metrics(records, ids):
    selected = [r for r in records if r['id'] in ids]
    return {'n': len(selected), 'correct': sum(r['correct'] for r in selected),
            'accuracy': statistics.mean(r['correct'] for r in selected) if selected else None,
            'unparseable': sum(not r['parsed'] for r in selected),
            'truncated': sum(r['truncated'] for r in selected)}


def paired(left, right, ids):
    a = {r['id']: r['correct'] for r in left}
    b = {r['id']: r['correct'] for r in right}
    if not set(ids) <= set(a) & set(b):
        return {'status': 'incomplete'}
    changes = [int(b[i])-int(a[i]) for i in ids]
    return {'n': len(ids), 'difference': statistics.mean(changes),
            'standard_error': statistics.stdev(changes)/math.sqrt(len(ids)),
            'wrong_to_right': changes.count(1), 'right_to_wrong': changes.count(-1),
            'exploratory_not_multiplicity_adjusted': True}


def report(root, status):
    folder = root/'controls'
    plan = json.loads((folder/'plan.json').read_text())
    raw = {f'original_{n}': read_records(root/'run'/f'{n}-rescored.jsonl', plan['pilot_ids']) for n in ('nonthinking','thinking')}
    raw.update({name: read_records(folder/f'{name}.jsonl', plan['pilot_ids']) for name in CONDITIONS})
    result = {'status': status, 'at': time.time(), 'paid_api_calls': 0, 'external_evaluation_used': False,
              'metric': 'custom direct numeric answer accuracy; not official FinQA program execution',
              'conditions': {name: {'all_32': metrics(records, plan['pilot_ids']),
                                    'sensitivity_24': metrics(records, plan['clean_ids'])} for name, records in raw.items()},
              'paired': {}}
    for left, right in [('original_nonthinking','clean_nonthinking'),('original_thinking','clean_thinking'),('clean_thinking','evidence_thinking')]:
        result['paired'][right+'-minus-'+left] = {
            'all_32': paired(raw[left],raw[right],plan['pilot_ids']),
            'sensitivity_24': paired(raw[left],raw[right],plan['clean_ids'])}
    write_json(folder/'summary.json',result)
    dest = Path('experiments/08_finqa/natural-finqa-controls-results')
    write_json(dest.with_suffix('.json'),result)
    lines = ['# FinQA prompt and evidence controls','',f'Status: **{status}**. No paid calls or weight edits.','',
             'Original-label numeric accuracy is primary. The 24-item audit sensitivity is post hoc, not a new benchmark. Gold evidence is diagnostic only.','',
             '| Condition | Completed | All 32 correct | All 32 accuracy | Sensitivity correct | Unparseable | Truncated |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, values in result['conditions'].items():
        a,b=values['all_32'],values['sensitivity_24']
        accuracy=f'{100*a["accuracy"]:.1f}%' if a['n'] else 'pending'
        lines.append(f'| {name} | {a["n"]}/32 | {a["correct"]} | {accuracy} | {b["correct"]}/{b["n"]} | {a["unparseable"]} | {a["truncated"]} |')
    lines += ['', 'Partial counts are progress only. See [the plan](natural-finqa-controls-plan.md) and [the audit](natural-finqa-audit.json).','']
    dest.with_suffix('.md').write_text('\n'.join(lines))
    return result


def run(root):
    os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    baseline, plan, rows = prepare(root)
    folder = root/'controls'
    source = Path(__file__)
    dependencies = [source, source.with_name('finqa_screen.py'), source.with_name('hotpot_screen.py')]
    metadata = {'source_hashes': {p.name:digest(p) for p in dependencies}, 'plan_sha256':digest(folder/'plan.json'),
                'versions': {k:version(k) for k in ('torch','transformers','vllm')},
                'max_model_len':CONTEXT, 'max_output_tokens':'context minus prompt', 'seed':SEED,
                'presence_penalty':0, 'sampler':'vllm built-in', 'gpu_memory_utilization':.7,
                'max_num_seqs':2, 'max_num_batched_tokens':2048, 'enforce_eager':True, 'run_hours':3}
    if (folder/'execution.json').exists() and json.loads((folder/'execution.json').read_text()) != metadata:
        raise ValueError('Execution code/settings changed')
    write_json(folder/'execution.json', metadata)
    for p in dependencies: (folder/p.name).write_bytes(p.read_bytes())
    write_json(folder/'process.json', {'pid':os.getpid(),'at':time.time()})
    write_json(folder/'status.json', {'status':'loading_model','at':time.time()})
    report(root,'loading_model')
    free = subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    if int(free.strip()) < 14000: raise RuntimeError('GPU not free')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(baseline['model_path'],local_files_only=True)
    prompts = {n:[tokenizer.apply_chat_template([{'role':'user','content':make_prompt(r,c)}], tokenize=False,
                  add_generation_prompt=True, enable_thinking=c['thinking']) for r in rows] for n,c in CONDITIONS.items()}
    if any(len(tokenizer(p,add_special_tokens=False).input_ids)>CONTEXT-8192 for texts in prompts.values() for p in texts):
        raise ValueError('Insufficient output room; no truncation or sample replacement allowed')
    model = LLM(model=baseline['model_path'],dtype='bfloat16',gpu_memory_utilization=.7,max_model_len=CONTEXT,
                enforce_eager=True,seed=0,max_num_seqs=2,max_num_batched_tokens=2048)
    started=time.monotonic()
    for name,condition in CONDITIONS.items():
        path=folder/f'{name}.jsonl'; records=read_records(path,plan['pilot_ids'])
        completed={r['id'] for r in records}
        todo=[(i,r) for i,r in enumerate(rows) if r['id'] not in completed]
        for start in range(0,len(todo),2):
            if time.monotonic()-started>10800:
                write_json(folder/'status.json',{'status':'stopped_time_limit','at':time.time()});report(root,'stopped_time_limit');return
            batch=todo[start:start+2];texts=[prompts[name][i] for i,_ in batch]
            sizes=[len(tokenizer(t,add_special_tokens=False).input_ids) for t in texts]
            thinking=condition['thinking']
            params=[SamplingParams(temperature=.6 if thinking else .7,top_p=.95 if thinking else .8,top_k=20,
                    presence_penalty=0,max_tokens=CONTEXT-size,seed=SEED+i) for (i,_),size in zip(batch,sizes)]
            write_json(folder/'status.json',{'status':'running','condition':name,'completed':len(records),'expected':32,'at':time.time(),'paid_api_calls':0})
            before=time.monotonic();outputs=model.generate(texts,params,use_tqdm=False)
            for (_,row),size,output in zip(batch,sizes,outputs):
                response=output.outputs[0]
                record={'id':row['id'],'condition':name,'question':row['qa']['question'],'response':response.text,
                        'gold':row['qa']['answer'],'truncated':response.finish_reason=='length',
                        'output_tokens':len(response.token_ids),'prompt_tokens':size,'batch_seconds':time.monotonic()-before,
                        'at':time.time(),**grade(response.text,thinking,row['qa']['answer'])}
                with path.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                records.append(record)
                print(json.dumps({k:record[k] for k in ('id','condition','correct','output_tokens','truncated')}),flush=True)
            report(root,'running')
    prepare(root)
    final=report(root,'completed');final['fixture_hashes_verified_after_inference']=True
    write_json(folder/'summary.json',final);write_json(Path('experiments/08_finqa/natural-finqa-controls-results.json'),final)
    write_json(folder/'status.json',{'status':'completed','at':time.time(),'paid_api_calls':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.wsplab/natural-finqa'));parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        _,plan,_=prepare(args.root.resolve());print(json.dumps({'pilot':len(plan['pilot_ids']),'sensitivity':len(plan['clean_ids'])}))
    else:
        try:run(args.root.resolve())
        except Exception as exc:
            folder=args.root/'controls';folder.mkdir(exist_ok=True)
            write_json(folder/'status.json',{'status':'failed','at':time.time(),'error':str(exc)})
            raise
