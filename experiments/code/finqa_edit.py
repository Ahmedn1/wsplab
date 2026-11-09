"""Local, interactive FinQA weight-edit pilot using the normal wsplab session tools.

Only practice records are exported to the editor. Selection is scored privately;
the reserved external partition is neither materialized nor evaluated here.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from wsplab.eval.graders import Grade
from wsplab.runners.base import Generation
from wsplab.runners.lm import LM

if __package__:
    from .finqa_screen import CONTEXT, grade, numeric, prompt, verify
    from .hotpot_screen import digest, write_json
else:
    from finqa_screen import CONTEXT, grade, numeric, prompt, verify
    from hotpot_screen import digest, write_json

SAMPLING = {'temperature': .6, 'top_p': .95, 'top_k': 20,
            'presence_penalty': 0, 'max_tokens': CONTEXT}
NOTE = (
    'Improve genuine financial-report numerical reasoning by direct weight edits, without training. '
    'The model is an unmodified public Qwen3-0.6B, not a deliberately damaged checkpoint. '
    'The prompt, full source context, thinking mode, sampler and scorer are frozen. '
    'Do not change prompts or use annotated evidence. Do not run or simulate the model outside '
    'the official forward tool, including on saved activations. No gradients, optimizers, or fitting '
    'weights to a loss. Weight statistics and analysis of recorded activations are allowed. '
    'Use the first 16 practice records as DEV and the next 16 as HOLDOUT. Do not inspect or use '
    'the remaining 32 practice records in this bounded pilot. Freeze the edit before one HOLDOUT '
    'comparison; do not retune afterward. Test at most two nonzero candidates, then retain the '
    'best DEV candidate only if its gain exceeds one paired standard error; otherwise restore '
    'original weights. A final no-change result is acceptable. '
    'Published references can be ambiguous or wrong: retain the frozen scorer and do not silently '
    'change labels. The 32 selection questions are private final evaluation for this pilot; '
    'never read them or baseline.json. The separate 96-report external partition is unused. '
    'Record exact edits, rationale, tested IDs and outcomes in /work/notes.md. '
    'Use session exec and session forward, and session finish when done.'
)


class FinqaGrader:
    def grade(self, text, row):
        result = grade(text, True, row['answer'])
        return Grade(result['correct'], result['prediction'] if result['parsed'] else None)


def first_per_report(rows, reports, *, numeric_only=False):
    """Select in source order, optionally requiring an existing scalar reference."""
    selected = {}
    wanted = set(reports)
    if len(wanted) != len(reports):
        raise ValueError('Duplicate report IDs')
    for row in rows:
        report = row['id'].split('/page_')[0]
        if numeric_only and numeric(str(row['qa']['answer'])) is None:
            continue
        if report in wanted and report not in selected:
            selected[report] = row
    if set(selected) != wanted:
        raise ValueError('Missing reserved report')
    return [selected[report] for report in reports]


def convert(row):
    answer = str(row['qa']['answer'])
    if numeric(answer) is None:
        raise ValueError(f'Unsupported reference: {row["id"]}; do not silently drop it')
    return {'id': row['id'], 'problem': prompt(row), 'answer': answer, 'source': 'finqa'}


def prepare(root, out):
    plan = verify(root)
    splits = plan['reserved_report_splits']
    groups = [set(v) for v in splits.values()]
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
        raise ValueError('Report partitions overlap')
    source = json.loads((root / 'data/dev.json').read_text())
    first_practice = first_per_report(source, splits['practice'])
    practice = [convert(r) for r in first_per_report(source, splits['practice'], numeric_only=True)]
    selection = [convert(r) for r in first_per_report(source, splits['selection'])]
    assert len(practice) == 64 and len(selection) == 32
    out.mkdir(parents=True, exist_ok=True)
    files = {'practice.jsonl': practice, 'selection-private.jsonl': selection}
    hashes = {}
    for name, rows in files.items():
        text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        path = out / name
        if path.exists() and path.read_text() != text:
            raise ValueError('Frozen fixture changed')
        if not path.exists():
            path.write_text(text)
        hashes[name] = digest(path)
    manifest = {
        'source_plan_sha256': digest(root / 'plan.json'), 'files': hashes,
        'model_path': plan['model_path'], 'model_files': plan['model_files'],
        'practice_ids': [r['id'] for r in practice],
        'practice_selection': 'first source-order question with an existing numeric display reference per report',
        'unsupported_first_practice_ids': [r['id'] for r in first_practice if numeric(str(r['qa']['answer'])) is None],
        'selection_ids': [r['id'] for r in selection],
        'dev_ids': [r['id'] for r in practice[:16]],
        'holdout_ids': [r['id'] for r in practice[16:32]],
        'unused_practice_ids': [r['id'] for r in practice[32:]],
        'sampling': SAMPLING, 'context': CONTEXT, 'thinking': True,
        'prompt': 'unchanged original full-context FinQA prompt',
        'seed': 'wsplab item_seed: first 4 SHA256 bytes of item ID; same for both weights',
        'passes': 6, 'max_candidates': 2, 'max_python_runs': 20,
        'selection_rule': 'best DEV gain > one paired SE; otherwise restore original',
        'holdout_rule': 'freeze first; one confirmation; no subsequent tuning',
        'external_evaluation_used': False, 'paid_api_calls': 0,
        'metric': 'unchanged custom display-answer numeric grading; not official execution accuracy',
        'note': NOTE,
        'source_hashes': {p.name: digest(p) for p in
                          [Path(__file__), Path(__file__).with_name('finqa_screen.py')]},
    }
    dest = out / 'plan.json'
    if dest.exists() and json.loads(dest.read_text()) != manifest:
        raise ValueError('Frozen edit plan changed')
    write_json(dest, manifest)
    return manifest


class FinqaLM(LM):
    """Keep the original per-prompt remaining-context allowance on every arm."""
    def __init__(self, model, progress):
        super().__init__(model, thinking=True, max_model_len=CONTEXT, gpu_memory=.7,
                         enforce_eager=True, batch_size=2)
        self.progress = Path(progress)

    def _vllm(self, weights, prompts, n, sampling):
        lengths = [len(self.tok(p.text, add_special_tokens=False).input_ids) for p in prompts]
        if any(size > CONTEXT - 8192 for size in lengths):
            raise ValueError('Prompt leaves less than 8192 tokens; never truncate or drop it')
        with tempfile.TemporaryDirectory() as tmp:
            request = {'weights': str(weights), 'tokenizer': self.tokenizer_path,
                       'prompts': [p.text for p in prompts], 'seeds': [p.seed for p in prompts],
                       'max_tokens': [min(sampling['max_tokens'], CONTEXT - size) for size in lengths],
                       'sampling': {k: v for k, v in sampling.items() if k not in ('seed', 'max_tokens')},
                       'n': n, 'out': str(Path(tmp) / 'out.jsonl'), 'progress': str(self.progress)}
            path = Path(tmp) / 'request.json'
            write_json(path, request)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(path)],
                                    check=False, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError('FinQA generation worker failed: ' + result.stderr[-2500:])
            rows = [json.loads(line) for line in Path(request['out']).read_text().splitlines()]
        if len(rows) != len(prompts) or any(len(row) != n for row in rows):
            raise RuntimeError('Incomplete generation output')
        return [[Generation(r['text'], r['finish'], r['ids']) for r in row] for row in rows]


def worker(path):
    from vllm import LLM, SamplingParams
    request = json.loads(path.read_text())
    progress = Path(request['progress'])
    write_json(progress, {'status': 'loading_model', 'items': len(request['prompts']), 'at': time.time()})
    model = LLM(model=request['weights'], tokenizer=request['tokenizer'], dtype='bfloat16',
                gpu_memory_utilization=.7, max_model_len=CONTEXT, enforce_eager=True,
                seed=0, max_num_seqs=2, max_num_batched_tokens=2048)
    with open(request['out'], 'w') as f:
        for start in range(0, len(request['prompts']), 2):
            params = [SamplingParams(n=request['n'], seed=request['seeds'][i],
                                     max_tokens=request['max_tokens'][i], **request['sampling'])
                      for i in range(start, min(start + 2, len(request['prompts'])))]
            outputs = model.generate(request['prompts'][start:start + 2], params, use_tqdm=False)
            f.writelines(json.dumps([{'text': r.text, 'finish': r.finish_reason, 'ids': list(r.token_ids)}
                                    for r in output.outputs]) + '\n' for output in outputs)
            f.flush()
            write_json(progress, {'status': 'generating', 'completed': min(start + 2, len(request['prompts'])),
                                  'items': len(request['prompts']), 'at': time.time()})
    write_json(progress, {'status': 'generation_completed', 'items': len(request['prompts']), 'at': time.time()})


def run(root, out):
    import wsplab
    from wsplab.editors.agent import AgentEditor
    from wsplab.eval.lm_suite import LMSuite, read_jsonl
    from wsplab.kernel import Limits
    from wsplab.passes import ForwardLimits
    from wsplab.session.driver import Policy

    plan = prepare(root, out)
    editor = AgentEditor(out)
    write_json(out / 'status.json', {'status': 'starting_private_baseline', 'pid': os.getpid(), 'at': time.time()})
    try:
        suite = LMSuite(read_jsonl(out / 'practice.jsonl'), read_jsonl(out / 'selection-private.jsonl'),
                        FinqaGrader(), SAMPLING, benchmark='FinQA 32-report selection pilot', samples=1,
                        protocol='Original full-context prompt, thinking enabled; .6/.95/top-k 20, no presence '
                                 'penalty; 32768 total context, per-prompt full remaining output allowance.')
        runner = FinqaLM(plan['model_path'], out / 'generation-status.json')
        # Inspect only lengths in the supervisor. Never expose selection contents to the editor.
        for row in suite.practice + suite.evaluation:
            size = len(runner.tok(runner.render(row['problem']), add_special_tokens=False).input_ids)
            if size > CONTEXT - 8192:
                raise ValueError('Preflight failed: prompt too long')
        result = wsplab.run_lm(plan['model_path'], eval=suite, editor=editor, note=NOTE,
                              thinking=True, budget=6, max_runs=20, folder=out / 'run',
                              policy=Policy(handoff_every=10**9, require_all_passes=False, max_turns=40),
                              limits=Limits(seconds=180, memory_gb=6), runner=runner,
                              forward_limits=ForwardLimits(n_layers=28, full_attention=tuple(range(28)),
                                                           gen_max=32, max_tokens=CONTEXT), control=True)
        prepare(root, out)  # Verify frozen sources, data and original model after inference.
        report = {'verdict': result.verdict, 'reasons': result.reasons, 'baseline': result.baseline,
                  'final': result.final, 'patch': str(result.patch), 'report': str(result.folder / 'report.md')}
        write_json(out / 'summary.json', {**report, 'status': 'completed', 'paid_api_calls': 0,
                                         'external_evaluation_used': False, 'fixture_hashes_verified': True})
        write_json(out / 'status.json', {'status': 'completed', 'at': time.time()})
        editor.finished(report)
    except Exception as exc:  # noqa: BLE001 - report any daemon failure to its waiting session client
        report = {'verdict': 'error', 'reasons': [str(exc), traceback.format_exc()[-2000:]]}
        write_json(out / 'status.json', {'status': 'failed', **report, 'at': time.time()})
        editor.finished(report)
    # Let the waiting session client receive the final report before the socket exits.
    time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.wsplab/natural-finqa'))
    parser.add_argument('--out', type=Path, default=Path('.wsplab/fq-edit1'))
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.prepare:
        manifest = prepare(args.root.resolve(), args.out.resolve())
        print(json.dumps({'practice': len(manifest['practice_ids']), 'selection': len(manifest['selection_ids']),
                          'paid_api_calls': 0, 'external_evaluation_used': False}))
    else:
        run(args.root.resolve(), args.out.resolve())
