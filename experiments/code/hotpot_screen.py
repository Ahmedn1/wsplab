"""Local, frozen HotpotQA task screen; no editors, training, or paid API calls."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import statistics
import string
import subprocess
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

SEED = 20261107
CONTEXT = 32768
DATA_REVISION = '1908d6afbbead072334abe2965f91bd2709910ab'
CONDITIONS = {
    'full_nonthinking': ('full', False),
    'oracle_nonthinking': ('oracle', False),
    'bm25_nonthinking': ('bm25', False),
    'full_thinking': ('full', True),
    'oracle_thinking': ('oracle', True),
    'bm25_thinking': ('bm25', True),
}
# Fixed lexical control, not tuned on pilot answers. No answer/support labels used.
STOP = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'of', 'in', 'on', 'at', 'to', 'for', 'from', 'with', 'and', 'or', 'by',
        'as', 'that', 'this', 'which', 'who', 'what', 'when', 'where', 'how',
        'did', 'does', 'do', 'has', 'have', 'had'}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def verify(root):
    plan = json.loads((root / 'plan.json').read_text())
    for path, expected in plan['files'].items():
        if digest(root / path) != expected:
            raise ValueError(f'Frozen input changed: {path}')
    for path, expected in plan['model_files'].items():
        if digest(Path(plan['model_path']) / path) != expected:
            raise ValueError(f'Model input changed: {path}')
    return plan


def prepare(root, model_fixture):
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'plan.json').exists():
        return verify(root)
    import pyarrow.parquet as pq
    sources = json.loads((root / 'data/sources.json').read_text())
    for name, info in sources.items():
        if digest(root / 'data' / name) != info['sha256']:
            raise ValueError('Download digest mismatch')
    rows = pq.read_table(root / 'data/validation.parquet').to_pylist()
    assert len(rows) == 7405 and len({r['id'] for r in rows}) == len(rows)
    # Keep duplicate question strings together, without viewing answers or outcomes.
    groups = {}
    for row in rows:
        groups.setdefault(' '.join(row['question'].lower().split()), []).append(row)
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    boundaries = {'pilot': (0, 32), 'practice': (32, 160), 'selection': (160, 288),
                  'evaluation': (288, 1288), 'unused': (1288, len(keys))}
    splits = {name: [r['id'] for k in keys[a:b] for r in groups[k]]
              for name, (a, b) in boundaries.items()}
    # One representative per duplicate question in the pilot. Others remain in its partition.
    pilot = [groups[k][0] for k in keys[:32]]
    (root / 'pilot.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in pilot))
    old = json.loads((model_fixture / 'plan.json').read_text())
    model_files = {k.removeprefix('model/'): v for k, v in old['files'].items() if k.startswith('model/')}
    files = [root / 'pilot.jsonl', root / 'data/sources.json',
             *(root / 'data' / name for name in sorted(sources))]
    plan = {'created_at': time.time(), 'dataset': 'hotpotqa/hotpot_qa', 'configuration': 'distractor',
            'dataset_revision': DATA_REVISION, 'source_split': 'validation', 'seed': SEED,
            'model': old['model'], 'model_revision': old['model_revision'],
            'model_path': str((model_fixture / 'model').resolve()), 'model_files': model_files,
            'files': {str(p.relative_to(root)): digest(p) for p in files}, 'split_ids': splits,
            'pilot_ids': [r['id'] for r in pilot], 'conditions': CONDITIONS, 'paid_api_calls': 0,
            'external_evaluation_used': False,
            'limitations': ['Public dataset may be in pretraining.', 'Document titles may overlap partitions; only duplicate questions are grouped.',
                           'Oracle paragraphs use gold support labels and are diagnostic only.', '32 pilot questions are exploratory task selection, not a council result.'],
            'qualification_rule': 'In at least one matched thinking mode: full EM <= .50, oracle EM >= .65 and oracle-full EM >= .20; '
                                  'best full/BM25 EM across modes <= .60; all conditions truncation <= .10 and answer-marker missing <= .10. '
                                  'Review failures and paired uncertainty before spending. This does not establish editability.'}
    write_json(root / 'plan.json', plan)
    verify(root)
    return plan


def lexical_tokens(text):
    return [x for x in re.findall(r'\w+', text.lower()) if x not in STOP]


def paragraphs(row, setting):
    docs = list(zip(row['context']['title'], row['context']['sentences']))
    if setting == 'full':
        return docs
    if setting == 'oracle':
        support = set(row['supporting_facts']['title'])
        result = [(title, sentences) for title, sentences in docs if title in support]
        if {title for title, _ in result} != support:
            raise ValueError('Missing oracle supporting paragraph')
        return result
    if setting != 'bm25':
        raise ValueError(setting)
    terms = [lexical_tokens(title + ' ' + ' '.join(sents)) for title, sents in docs]
    query = set(lexical_tokens(row['question']))
    average = statistics.mean(map(len, terms)) or 1
    df = Counter(t for doc in terms for t in set(doc))
    scores = []
    for i, doc in enumerate(terms):
        counts = Counter(doc)
        score = sum(math.log(1 + (len(docs) - df[t] + .5) / (df[t] + .5)) *
                    counts[t] * 2.5 / (counts[t] + 1.5 * (.25 + .75 * len(doc) / average))
                    for t in query if counts[t])
        scores.append((score, i))
    selected = {i for _, i in sorted(scores, key=lambda pair: (-pair[0], pair[1]))[:4]}
    return [doc for i, doc in enumerate(docs) if i in selected]


def prompt(row, setting):
    context = '\n\n'.join(f'Title: {title}\n' + ''.join(sentences) for title, sentences in paragraphs(row, setting))
    return ('Use the passages to answer the question. Give the short answer inside '
            '<answer>...</answer>. Do not put an explanation inside the answer tags.\n\n'
            + context + '\n\nQuestion: ' + row['question'])


def extract(text, thinking):
    if thinking and '</think>' not in text:
        return '', False, True
    final = text.rsplit('</think>', 1)[-1].strip()
    matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', final, flags=re.DOTALL)
    if len(matches) == 1 and final.count('<answer>') == final.count('</answer>') == 1:
        return matches[0].strip(), True, False
    # Never search reasoning for a lucky match. Plain final text is scored as a whole.
    return final, False, False


def scorer(root):
    path = root / 'data/hotpot_evaluate_v1.py'
    # Execute only the three reviewed, unchanged answer-scoring functions.
    # The original CLI imports ujson/pickle, neither needed for these functions.
    names = {'normalize_answer', 'f1_score', 'exact_match_score'}
    tree = ast.parse(path.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in definitions} != names:
        raise ValueError('Official answer scorer is incomplete')
    namespace = {'re': re, 'string': string, 'Counter': Counter}
    # Source is pinned, hash-verified and manually reviewed; only scoring definitions are loaded.
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)  # noqa: S102
    return SimpleNamespace(**{name: namespace[name] for name in names})


def read_records(path, ids):
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    seen = [r['id'] for r in rows]
    if len(seen) != len(set(seen)) or not set(seen) <= set(ids):
        raise ValueError('Duplicate or foreign responses')
    return rows


def summarize(rows):
    return {'responses': len(rows), 'em': statistics.mean(r['em'] for r in rows) if rows else None,
            'f1': statistics.mean(r['f1'] for r in rows) if rows else None,
            'truncated': sum(r['truncated'] for r in rows),
            'missing_answer_marker': sum(not r['answer_marker'] for r in rows),
            'unfinished_thinking': sum(r['unfinished_thinking'] for r in rows),
            'mean_output_tokens': statistics.mean(r['output_tokens'] for r in rows) if rows else None}


def report(root, state):
    results = {name: read_records(root / 'run' / f'{name}.jsonl', json.loads((root / 'plan.json').read_text())['pilot_ids'])
               for name in CONDITIONS}
    data = {'status': state, 'at': time.time(), 'paid_api_calls': 0, 'is_council_result': False,
            'external_evaluation_used': False, 'conditions': {k: summarize(v) for k, v in results.items()}}
    paired = {}
    for mode in ('nonthinking', 'thinking'):
        full, oracle = results['full_' + mode], results['oracle_' + mode]
        if len(full) == len(oracle) == 32:
            original = {r['id']: r['em'] for r in full}
            diffs = [r['em'] - original[r['id']] for r in oracle]
            paired[mode] = {'oracle_minus_full_em': statistics.mean(diffs),
                            'standard_error': statistics.stdev(diffs) / math.sqrt(32),
                            'full_wrong_oracle_right': diffs.count(1), 'full_right_oracle_wrong': diffs.count(-1)}
    data['paired_oracle_diagnostic'] = paired
    if all(len(rows) == 32 for rows in results.values()):
        values = data['conditions']
        gap = any(values['full_' + m]['em'] <= .5 and values['oracle_' + m]['em'] >= .65
                  and paired[m]['oracle_minus_full_em'] >= .2 for m in paired)
        hard = max(values[k]['em'] for k in values if not k.startswith('oracle')) <= .6
        clean = all(v['truncated'] / 32 <= .1 and v['missing_answer_marker'] / 32 <= .1 for v in values.values())
        data['screening_rule'] = 'passes_manual_review_required' if gap and hard and clean else 'not_qualified_by_frozen_rule'
    else:
        data['screening_rule'] = 'incomplete'
    write_json(root / 'run/summary.json', data)
    dest = Path('experiments/07_hotpotqa/natural-hotpotqa-results')
    write_json(dest.with_suffix('.json'), data)
    lines = ['# HotpotQA local pilot', '', f'Status: **{state}**. Screening rule: **{data["screening_rule"]}**.', '',
             'No paid API calls, weight edits, council run or external evaluation. Oracle uses gold supporting paragraphs and is diagnostic only.', '',
             '| Condition | Completed / 32 | Answer EM | Answer F1 | Truncated | Missing answer tags |',
             '|---|---:|---:|---:|---:|---:|']
    for name, v in data['conditions'].items():
        metrics = f'{100*v["em"]:.1f}% | {100*v["f1"]:.1f}%' if v['responses'] else 'pending | pending'
        lines.append(f'| {name} | {v["responses"]}/32 | {metrics} | {v["truncated"]} | {v["missing_answer_marker"]} |')
    lines += ['', 'Incomplete conditions are progress only. See [the frozen plan](natural-hotpotqa-plan.md).', '']
    dest.with_suffix('.md').write_text('\n'.join(lines))
    return data


def run(root):
    # Match the repository LM runner: this machine has no CUDA compiler for
    # FlashInfer's JIT sampler. Use vLLM's built-in sampler for every condition.
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    plan = verify(root)
    out = root / 'run'
    out.mkdir(exist_ok=True)
    meta = {'script_sha256': digest(__file__), 'plan_sha256': digest(root / 'plan.json'),
            'versions': {k: version(k) for k in ('torch', 'transformers', 'vllm')},
            'context': CONTEXT, 'output_limit': 'context minus prompt; no artificial short cap',
            'seed': SEED, 'presence_penalty': 0, 'time_limit_hours': 3, 'paid_api_calls': 0,
            'VLLM_USE_FLASHINFER_SAMPLER': '0',
            'engine': {'dtype': 'bfloat16', 'max_num_seqs': 2, 'gpu_memory_utilization': .7}}
    if (out / 'execution.json').exists() and json.loads((out / 'execution.json').read_text()) != meta:
        raise ValueError('Execution configuration changed')
    write_json(out / 'execution.json', meta)
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write_json(out / 'process.json', {'pid': os.getpid(), 'at': time.time()})
    write_json(out / 'status.json', {'status': 'loading_model', 'at': time.time()})
    report(root, 'loading_model')
    free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
    if int(free.strip()) < 14000:
        raise RuntimeError('GPU not free; refusing to interfere with another run')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(plan['model_path'], local_files_only=True)
    pilot = [json.loads(line) for line in (root / 'pilot.jsonl').read_text().splitlines()]
    # Check every condition before inference; never truncate passages or silently drop items.
    prompts = {}
    for name, (setting, thinking) in CONDITIONS.items():
        prompts[name] = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt(r, setting)}],
                         tokenize=False, add_generation_prompt=True, enable_thinking=thinking) for r in pilot]
        if any(len(tokenizer(p, add_special_tokens=False).input_ids) > CONTEXT - 8192 for p in prompts[name]):
            raise ValueError('Insufficient output room in context; revise plan before inference')
    model = LLM(model=plan['model_path'], dtype='bfloat16', gpu_memory_utilization=.7,
                max_model_len=CONTEXT, enforce_eager=True, seed=0, max_num_seqs=2, max_num_batched_tokens=2048)
    official = scorer(root)
    start = time.monotonic()
    for name, (setting, thinking) in CONDITIONS.items():
        path = out / f'{name}.jsonl'
        records = read_records(path, plan['pilot_ids'])
        completed = {r['id'] for r in records}
        todo = [(i, r) for i, r in enumerate(pilot) if r['id'] not in completed]
        for index in range(0, len(todo), 2):
            if time.monotonic() - start > 3 * 3600:
                report(root, 'stopped_time_limit')
                write_json(out / 'status.json', {'status': 'stopped_time_limit', 'at': time.time()})
                return
            batch = todo[index:index+2]
            texts = [prompts[name][i] for i, _ in batch]
            sizes = [len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts]
            params = [SamplingParams(temperature=.6 if thinking else .7, top_p=.95 if thinking else .8,
                       top_k=20, presence_penalty=0, max_tokens=CONTEXT-size, seed=SEED+i)
                      for (i, _), size in zip(batch, sizes)]
            write_json(out / 'status.json', {'status': 'running', 'condition': name, 'completed': len(records),
                       'expected': 32, 'at': time.time(), 'paid_api_calls': 0})
            before = time.monotonic()
            outputs = model.generate(texts, params, use_tqdm=False)
            for (_, row), size, output in zip(batch, sizes, outputs):
                result = output.outputs[0]
                answer, marker, unfinished = extract(result.text, thinking)
                f1 = official.f1_score(answer, row['answer'])[0]
                em = int(official.exact_match_score(answer, row['answer']))
                selected = {title for title, _ in paragraphs(row, setting)}
                support = set(row['supporting_facts']['title'])
                record = {'id': row['id'], 'type': row['type'], 'condition': name, 'question': row['question'],
                          'gold': row['answer'], 'response': result.text, 'prediction': answer,
                          'em': em, 'f1': f1, 'answer_marker': marker, 'unfinished_thinking': unfinished,
                          'truncated': result.finish_reason == 'length', 'output_tokens': len(result.token_ids),
                          'prompt_tokens': size, 'batch_seconds': time.monotonic()-before,
                          'all_support_paragraphs_retained': support <= selected,
                          'at': time.time()}
                with path.open('a') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                records.append(record)
                print(json.dumps({k: record[k] for k in ('id','condition','em','f1','output_tokens','truncated')}), flush=True)
            report(root, 'running')
    verify(root)
    final = report(root, 'completed')
    final['fixture_hashes_verified_after_inference'] = True
    write_json(out / 'summary.json', final)
    write_json(Path('experiments/07_hotpotqa/natural-hotpotqa-results.json'), final)
    write_json(out / 'status.json', {'status': 'completed', 'at': time.time(), 'paid_api_calls': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.wsplab/natural-hotpotqa'))
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        plan = prepare(args.root.resolve(), Path('.wsplab/natural-mgsm').resolve())
        print(json.dumps({'pilot': len(plan['pilot_ids']), 'split_sizes': {k:len(v) for k,v in plan['split_ids'].items()}}))
    else:
        try:
            run(args.root.resolve())
        except Exception as exc:
            out = args.root / 'run'
            out.mkdir(parents=True, exist_ok=True)
            write_json(out / 'status.json', {'status':'failed','error':str(exc),'at':time.time()})
            raise
