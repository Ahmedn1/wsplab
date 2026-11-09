"""Unpaid FinQA direct-numeric-answer screen, not official program execution accuracy."""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import time
from decimal import Decimal, InvalidOperation
from importlib.metadata import version
from pathlib import Path

if __package__:
    from .hotpot_screen import digest, read_records, write_json
else:
    from hotpot_screen import digest, read_records, write_json

SEED = 20261108
CONTEXT = 32768


def numeric(text):
    # LaTeX presentation escapes are not numerical errors. Keep units/scale unchanged.
    text = text.replace(r'\%', '%').replace(r'\$', '$').replace(r'\,', '')
    text = text.strip().replace('\\n', '').replace(',', '').replace('$', '').strip()
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*%?', text):
        return None
    try:
        return Decimal(text.rstrip('%').strip())
    except InvalidOperation:
        return None


def grade(text, thinking, gold):
    if thinking and '</think>' not in text:
        return {'correct': False, 'parsed': False, 'unfinished_thinking': True, 'prediction': ''}
    final = text.rsplit('</think>', 1)[-1].strip()
    boxes = re.findall(r'\\boxed\{([^{}]*)\}', final)
    if len(boxes) == 1:
        predicted = boxes[0].strip()
    elif not boxes:
        predicted = re.sub(r'^(?:answer|final answer)\s*:\s*', '', final, flags=re.IGNORECASE).strip()
    else:
        predicted = ''
    value, target = numeric(predicted), numeric(gold)
    if target is None:
        raise ValueError('Unsupported gold: do not silently drop an example')
    # Honor precision of the published human answer; integers are exact except
    # whole-percent answers, which are rounded to whole percentage points.
    places = max(0, -target.as_tuple().exponent)
    tolerance = Decimal('0.5') * Decimal(10) ** (-places) if places or '%' in gold else Decimal('0.000001')
    correct = value is not None and abs(value-target) <= tolerance + Decimal('0.00000001')
    return {'correct': correct, 'parsed': value is not None, 'unfinished_thinking': False,
            'prediction': predicted, 'gold': gold, 'tolerance': str(tolerance)}


def prompt(row):
    # Use the original table, which preserves multi-row headers lost in the
    # dataset's normalized table. Do not read qa.program/gold_inds/model_input.
    table = '\n'.join(' | '.join(str(c) for c in r) for r in row['table_ori'])
    context = '\n'.join(row['pre_text']) + '\n\nTable:\n' + table + '\n\n' + '\n'.join(row['post_text'])
    return ('Read this financial-report excerpt and answer the question. Reason through the calculation. '
            'Put only the final number in \\boxed{...}, using the units requested in the question. '
            'For a percentage, give percentage units (for example 12.5%, not 0.125).\n\n'
            + context + '\n\nQuestion: ' + row['qa']['question'])


def verify(root):
    plan = json.loads((root/'plan.json').read_text())
    for name, value in plan['files'].items():
        if digest(root/name) != value:
            raise ValueError(f'Input changed: {name}')
    for name, value in plan['model_files'].items():
        if digest(Path(plan['model_path'])/name) != value:
            raise ValueError(f'Model changed: {name}')
    return plan


def prepare(root):
    if (root/'plan.json').exists():
        return verify(root)
    preparation = json.loads((root/'preparation.json').read_text())
    assert digest(root/'pilot.jsonl') == preparation['pilot_sha256']
    pilot = [json.loads(s) for s in (root/'pilot.jsonl').read_text().splitlines()]
    assert len(pilot) == 32 and all(numeric(r['qa']['answer']) is not None for r in pilot)
    old = json.loads(Path('.wsplab/natural-mgsm/plan.json').read_text())
    sources = json.loads((root/'data/sources.json').read_text())
    for name, source in sources.items():
        assert digest(root/'data'/name) == source['sha256']
    files = ['preparation.json','pilot.jsonl','data/sources.json',*(f'data/{name}' for name in sources)]
    plan = {**preparation, 'status':'frozen_before_inference', 'model':old['model'],
            'model_revision':old['model_revision'], 'model_path':str(Path('.wsplab/natural-mgsm/model').resolve()),
            'model_files':{k.removeprefix('model/'):v for k,v in old['files'].items() if k.startswith('model/')},
            'files':{name:digest(root/name) for name in files},
            'metric':'custom direct numeric answer matching published qa.answer precision; NOT official program accuracy',
            'rule':'Both conditions completed; best accuracy <= .40, each truncation <= .10 and unparseable <= .10. Manual error review required.',
            'conditions':['nonthinking','thinking'],'paid_api_calls':0,
            'gold_caveat':'Human display answer and execution answer can disagree; preserve both and review disagreements. No labels corrected or examples removed.'}
    write_json(root/'plan.json',plan)
    return verify(root)


def report(root,status):
    ids=json.loads((root/'plan.json').read_text())['pilot_ids']
    data={'status':status,'paid_api_calls':0,'metric':'direct_answer_accuracy_not_official_execution_accuracy', 'conditions':{}}
    for name in ('nonthinking','thinking'):
        rows=read_records(root/'run'/f'{name}.jsonl',ids)
        data['conditions'][name]={'completed':len(rows),'correct':sum(r['correct'] for r in rows),
             'accuracy':statistics.mean(r['correct'] for r in rows) if rows else None,
             'truncated':sum(r['truncated'] for r in rows),'unparseable':sum(not r['parsed'] for r in rows)}
    values=list(data['conditions'].values())
    if all(v['completed']==32 for v in values):
        passed=max(v['accuracy'] for v in values)<=.4 and all(v['truncated']/32<=.1 and v['unparseable']/32<=.1 for v in values)
        data['screening_rule']='passes_manual_review_required' if passed else 'not_qualified_by_frozen_rule'
    else:
        data['screening_rule']='incomplete'
    write_json(root/'run/summary.json',data)
    write_json(Path('experiments/08_finqa/natural-finqa-results.json'),data)
    lines=['# FinQA direct-answer pilot','',f'Status: **{status}**. Rule: **{data["screening_rule"]}**.','',
           'Custom numeric-answer metric, not official FinQA program/execution accuracy. No paid calls, edits or council run.','',
           '| Condition | Completed | Correct | Accuracy | Truncated | Unparseable |','|---|---:|---:|---:|---:|---:|']
    for name,v in data['conditions'].items():
        acc=f'{100*v["accuracy"]:.1f}%' if v['completed'] else 'pending'
        lines.append(f'| {name} | {v["completed"]}/32 | {v["correct"]} | {acc} | {v["truncated"]} | {v["unparseable"]} |')
    lines+=['','Incomplete conditions are progress only. See [the plan](natural-finqa-plan.md).','']
    Path('experiments/08_finqa/natural-finqa-results.md').write_text('\n'.join(lines))
    return data


def run(root):
    os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    plan=verify(root)
    out=root/'run';out.mkdir(exist_ok=True)
    metadata={'plan_sha256':digest(root/'plan.json'),'script_sha256':digest(__file__),
              'helper_sha256':digest(Path(__file__).with_name('hotpot_screen.py')),
              'versions':{k:version(k) for k in ('torch','transformers','vllm')},
              'seed':SEED,'context':CONTEXT,'max_output':'remaining context','presence_penalty':0,
              'sampler':'vllm built-in','active_limit_hours':3,'paid_api_calls':0}
    if (out/'execution.json').exists() and json.loads((out/'execution.json').read_text())!=metadata:
        raise ValueError('Execution changed; preserve previous run')
    write_json(out/'execution.json',metadata)
    for source in (Path(__file__),Path(__file__).with_name('hotpot_screen.py')):
        (out/source.name).write_bytes(source.read_bytes())
    write_json(out/'process.json',{'pid':os.getpid(),'at':time.time()})
    write_json(out/'status.json',{'status':'loading_model','at':time.time()});report(root,'loading_model')
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    if int(free.strip())<14000: raise RuntimeError('GPU not free')
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(plan['model_path'],local_files_only=True)
    rows=[json.loads(s) for s in (root/'pilot.jsonl').read_text().splitlines()]
    prompts={name:[tokenizer.apply_chat_template([{'role':'user','content':prompt(r)}],tokenize=False,
                  add_generation_prompt=True,enable_thinking=name=='thinking') for r in rows]
             for name in ('nonthinking','thinking')}
    if any(len(tokenizer(p,add_special_tokens=False).input_ids)>CONTEXT-8192 for texts in prompts.values() for p in texts):
        raise ValueError('Prompt too long; do not truncate or drop examples')
    model=LLM(model=plan['model_path'],dtype='bfloat16',gpu_memory_utilization=.7,max_model_len=CONTEXT,
              enforce_eager=True,seed=0,max_num_seqs=2,max_num_batched_tokens=2048)
    start=time.monotonic()
    for name in ('nonthinking','thinking'):
        path=out/f'{name}.jsonl';records=read_records(path,plan['pilot_ids'])
        done={r['id'] for r in records};todo=[(i,r) for i,r in enumerate(rows) if r['id'] not in done]
        thinking=name=='thinking'
        for k in range(0,len(todo),2):
            if time.monotonic()-start>10800:
                report(root,'stopped_time_limit');write_json(out/'status.json',{'status':'stopped_time_limit','at':time.time()});return
            batch=todo[k:k+2];texts=[prompts[name][i] for i,_ in batch]
            lengths=[len(tokenizer(t,add_special_tokens=False).input_ids) for t in texts]
            params=[SamplingParams(temperature=.6 if thinking else .7,top_p=.95 if thinking else .8,
                    top_k=20,presence_penalty=0,max_tokens=CONTEXT-size,seed=SEED+i) for (i,_),size in zip(batch,lengths)]
            write_json(out/'status.json',{'status':'running','condition':name,'completed':len(records),'expected':32,'at':time.time(),'paid_api_calls':0})
            before=time.monotonic();outputs=model.generate(texts,params,use_tqdm=False)
            for (_,row),length,output in zip(batch,lengths,outputs):
                response=output.outputs[0]
                record={'id':row['id'],'condition':name,'question':row['qa']['question'],
                        'response':response.text,'gold_execution_answer':row['qa']['exe_ans'],
                        'truncated':response.finish_reason=='length','output_tokens':len(response.token_ids),
                        'prompt_tokens':length,'batch_seconds':time.monotonic()-before,'at':time.time(),
                        **grade(response.text,thinking,row['qa']['answer'])}
                with path.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                records.append(record)
                print(json.dumps({k:record[k] for k in ('id','condition','correct','output_tokens','truncated')}),flush=True)
            report(root,'running')
    verify(root);final=report(root,'completed');final['fixture_hashes_verified_after_inference']=True
    write_json(out/'summary.json',final);write_json(Path('experiments/08_finqa/natural-finqa-results.json'),final)
    write_json(out/'status.json',{'status':'completed','at':time.time(),'paid_api_calls':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.wsplab/natural-finqa'));parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        print(json.dumps({'pilot':len(prepare(args.root.resolve())['pilot_ids'])}))
    else:
        try:run(args.root.resolve())
        except Exception as exc:
            out=args.root/'run';out.mkdir(parents=True,exist_ok=True)
            write_json(out/'status.json',{'status':'failed','error':str(exc),'at':time.time()})
            raise
