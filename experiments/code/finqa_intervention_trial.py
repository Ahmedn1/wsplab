"""Prepare or run a bounded FinQA peer-versus-one-solo weight-edit comparison.

Preparation is offline. --run-editors requires a separate recorded spending
approval. External evaluation additionally requires a completed protocol audit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from wsplab.council import Board, Peer
from wsplab.eval.lm_suite import LMEvaluation, LMSuite, read_jsonl
from wsplab.eval.stats import paired
from wsplab.passes import GenerationCache
from wsplab.session.protocol import parse

if __package__:
    from . import peer_trial as paid
    from .finqa_edit import SAMPLING, FinqaGrader, FinqaLM, convert, first_per_report
    from .finqa_intervention_guard import InterventionEditor
    from .finqa_screen import numeric, verify
    from .hotpot_screen import digest, write_json
else:
    import peer_trial as paid
    from finqa_edit import SAMPLING, FinqaGrader, FinqaLM, convert, first_per_report
    from finqa_intervention_guard import InterventionEditor
    from finqa_screen import numeric, verify
    from hotpot_screen import digest, write_json

CAPS = {'peers': {'openai': 2., 'anthropic': 5., 'gemini': 5.}, 'solo': {'anthropic': 12.}}
PRICES = {**paid.PRICES, 'openai:gpt-6.1-terra': (1.25, 5., .05)}
SETTINGS = {'peers': {'passes_each': 6, 'runs_each': 20, 'turns_each': 40, 'required_candidates': 2, 'diagnostic_passes': 1},
            'solo': {'passes_each': 18, 'runs_each': 60, 'turns_each': 120, 'required_candidates': 6, 'diagnostic_passes': 3}}
NOTE = (
    'Improve genuine FinQA reasoning by direct weight edits to an unmodified public Qwen3-0.6B. '
    'Keep the original full-context prompt, thinking and numerical scoring fixed. No gradients, '
    'optimizers, training, or fitting weights to a loss. Do NOT compute predictions, logits, losses '
    'or accuracy for any modified weights in Python, even for the last layer on saved activations. '
    'Use only the official forward tool to run any part of the model. Weight statistics and analysis '
    'of recorded activations are allowed. Use first 32 practice records as DEV and next 32 as '
    'internal HOLDOUT; freeze before one HOLDOUT comparison and never tune afterward. '
    'Use HOLDOUT only through generate problems IDs, never teacher-forced texts or custom prompts. '
    'Your HOLDOUT request and all subsequent messages are withheld from peers; DEV findings are shared. '
    'Do not change reference labels or retrieve data/checkpoints. Published labels can be imperfect. '
    'Actual intervention tests are REQUIRED, not merely diagnosis. Weak observational evidence can '
    'motivate a reversible causal experiment; do not demand causal proof before testing an edit. '
    'Each peer must test two distinct nonzero edited checkpoints; solo must test six. '
    'Before the first test, write /work/interventions.json as a JSON list with exactly that many entries, '
    'each containing id, tensors (exact tensor-name list), operation, strength, and rationale. Freeze '
    'this file before edited generation. Write the chosen id to /work/active_candidate.txt for each test. '
    'Each candidate is relative to original weights, not accumulated accidentally. Use load_original '
    'and drop_edit to reset; save_edit must change actual BF16 entries. Two strengths of one mechanistic '
    'hypothesis may be separate candidates. For each candidate use official generate, model=both, '
    'all first 32 DEV IDs in source order, n=1, max_tokens=32768. No custom prompts. '
    'Use at most one diagnostic-only pass per peer, three for solo, using original-model generation on '
    'all DEV IDs with n=1. No teacher-forced or synthetic screens in this follow-up. '
    'Aim to save the first edit by Python run 6. Save exact candidate tensor backups in /work to restore a winner. '
    'Reserve at least one pass for HOLDOUT. Examples of reversible hypotheses include damping a small '
    'selected neuron group or an attention-head output block. Choose the hypothesis yourself; respect '
    'frozen tensors, and do not rescale an entire tensor. After the mandatory tests, choose the best '
    'DEV gain only if positive; ties prefer original. For a retained edit freeze it, run one model=both, '
    'n=1 comparison on all 32 HOLDOUT IDs, then retain that exact edit only if HOLDOUT gain is positive '
    'or restore original. Never retune. No-change is acceptable AFTER the required tests; a budget '
    'or infrastructure failure must be reported explicitly and cannot count as completion. '
    'A prior local pilot found no net DEV benefit from removing or halving layer-21 MLP neuron 377; '
    'this does not restrict other hypotheses. A previous council/solo run diagnosed arithmetic, table '
    'choice and unit errors but tested no edits. Its small correlation screens found no convincing '
    'common neuron; that does not rule out useful interventions. Both arms receive this same summary, '
    'not prior editor transcripts, private candidate-selection scores or patches. '
    'The 32-question candidate-selection set is hidden, and the separate external evaluation is '
    'not accessed until all editing, candidate selection and protocol auditing have finished. '
    'Record exact edits, IDs and evidence in /work/notes.md. An improvement is not established '
    'by a plausible story or a shorter response. Finish when done; spending and pass budgets are ceilings.'
)


class ConfirmationBoard(Board):
    """A confirming peer cannot leak results into another peer's ongoing search."""
    def __init__(self, peers):
        super().__init__(peers)
        self.private = set()
        self.failed = False

    def abort(self):
        with self.condition:
            self.failed = True
            self.generation += 1
            self.condition.notify_all()

    def exchange(self, peer, post):
        with self.condition:
            if self.failed:
                raise RuntimeError('Council aborted because a peer failed')
            if peer in self.private:
                post = {'reply': 'Peer entered private confirmation.',
                        'result': 'Confirmation and subsequent messages withheld.'}
            result = super().exchange(peer, post)
            if self.failed:
                raise RuntimeError('Council aborted because a peer failed')
            return result


class ArmControl:
    def __init__(self, participants, board=None):
        self.ready = threading.Barrier(participants)
        self.cancelled = threading.Event()
        self.board = board

    def check(self):
        if self.cancelled.is_set():
            raise RuntimeError('Arm aborted because a participant failed')

    def abort(self):
        self.cancelled.set()
        self.ready.abort()
        if self.board is not None:
            self.board.abort()


class ReadyEditor:
    """No paid requests until every participant has initialized successfully."""
    def __init__(self, editor, control):
        self.editor, self.control, self.name = editor, control, editor.name
        self.started = False

    def reply(self, system, messages):
        if not self.started:
            self.control.ready.wait()
            self.started = True
        self.control.check()
        return self.editor.reply(system, messages)


def warm_tokenizers(runners):
    # Transformers lazy imports must finish in the main thread, before workers
    # can render prompts. GPU serialization alone does not cover rendering.
    for runner in runners:
        _ = runner.tok


def spending_caps(out):
    history = json.loads((out / 'budget-history.json').read_text())
    spent = {arm: dict.fromkeys(caps, 0.) for arm, caps in CAPS.items()}
    for entry in history['ledgers']:
        path = Path(entry['path'])
        if digest(path) != entry['sha256']:
            raise ValueError('Previous spending ledger changed')
        ledger = json.loads(path.read_text())
        if any(ledger['reserved'].values()):
            raise ValueError('Previous spending reservations remain')
        for provider, value in ledger['spent'].items():
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid historical charge')
            spent[entry['arm']][provider] += value
    peers = {p: round(CAPS['peers'][p] - value, 8) for p, value in spent['peers'].items()}
    solo_available = CAPS['solo']['anthropic'] - spent['solo']['anthropic']
    total = round(sum(peers.values()), 8)
    if any(v <= 0 for v in peers.values()) or total > solo_available:
        raise ValueError('Insufficient remaining matched budget')
    caps = {'peers': peers, 'solo': {'anthropic': total}}
    return caps, {**history, 'previous_spent': spent,
                  'previous_total': sum(sum(v.values()) for v in spent.values())}


class ConfirmationPeer(Peer):
    def __init__(self, editor, board, index, holdout_ids):
        super().__init__(editor, board, index)
        self.holdout_ids = set(holdout_ids)

    def reply(self, system, messages):
        reply = super().reply(system, messages)
        for block in parse(reply.text, {'forward'}).blocks:
            try:
                request = json.loads(block.body)
            except (ValueError, TypeError):
                continue
            if isinstance(request, dict) and (
                self.holdout_ids.intersection(request.get('problems') or []) or request.get('prompts')
            ):
                # Mark BEFORE execution, including a cache hit. Redact permanently,
                # since later Python output or prose can repeat the private result.
                self.board.private.add(self.index)
        return reply


def write_frozen(path, text):
    if path.exists() and path.read_text() != text:
        raise ValueError(f'Frozen file changed: {path.name}')
    if not path.exists():
        path.write_text(text)


def choose_winner(candidates, baseline):
    """The unchanged model participates; ties do not force a gratuitous edit."""
    eligible = [r for r in candidates if r['changed_entries'] and r['score'] > baseline]
    return max(eligible, key=lambda r: (r['score'], -r['changed_entries'], r['provider'])) if eligible else None


def prepare(root, out):
    caps, recovery = spending_caps(out)
    source = verify(root)
    reports = source['reserved_report_splits']
    groups = [set(v) for v in reports.values()]
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
        raise ValueError('Report partitions overlap')
    raw = json.loads((root / 'data/dev.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    files, selections = {}, {}
    for partition in ('practice', 'selection', 'evaluation'):
        available = {r['id'].split('/page_')[0] for r in raw if numeric(str(r['qa']['answer'])) is not None}
        eligible = [r for r in reports[partition] if r in available]
        chosen = first_per_report(raw, eligible, numeric_only=True)
        rows = [convert(r) for r in chosen]
        path = out / f'{partition}-private.jsonl'
        write_frozen(path, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        files[path.name] = digest(path)
        first = first_per_report(raw, eligible)
        selections[partition] = {'n': len(rows), 'ids': [r['id'] for r in rows],
                                'reports_without_numeric_reference': [r for r in reports[partition] if r not in available],
                                'replaced_unscorable_first_ids': [a['id'] for a, b in zip(first, chosen)
                                                                  if a['id'] != b['id']]}
    if [selections[p]['n'] for p in ('practice', 'selection')] != [64, 32] or not selections['evaluation']['n']:
        raise ValueError('Unexpected report counts')
    sources = [Path(__file__), Path(__file__).with_name('finqa_intervention_watch.py'), Path(__file__).with_name('finqa_intervention_guard.py'), Path(__file__).with_name('finqa_edit.py'),
               Path(__file__).with_name('finqa_screen.py'), Path(paid.__file__)]
    import wsplab.council.peers
    import wsplab.pipeline
    sources += [Path(wsplab.pipeline.__file__), Path(wsplab.council.peers.__file__)]
    plan = {'status': 'prepared_not_authorized', 'model_path': source['model_path'],
            'source_plan_sha256': digest(root / 'plan.json'), 'model_files': source['model_files'],
            'files': files, 'partitions': selections, 'models': paid.MODELS, 'solo_provider': 'anthropic',
            'caps': caps, 'total_additional_usd': sum(sum(c.values()) for c in caps.values()),
            'recovery': recovery, 'prices': PRICES, 'settings': SETTINGS,
            'proposed_cumulative_limits': {'openai': 22., 'anthropic': 47., 'gemini': 30.},
            'scratch_root': '/tmp/wsplab-finqa-intervention-comparison',
            'sampling': SAMPLING, 'gen_max': 64, 'samples': 1, 'max_output_tokens': 16384, 'note': NOTE,
            'question_rule': 'first source-order question with an existing numeric display reference per reserved report',
            'selection_rule': 'best private selection accuracy, including original; ties prefer original then fewer edits',
            'primary_test': 'selected peers minus selected solo on eligible external reports; exact paired two-sided p < .05',
            'secondary_test': 'each selected edit versus original; same-size random control for a positive edit claim',
            'council_win_rule': 'peers beat solo, original, and same-size random control; all gains positive and exact paired p < .05',
            'limits': 'exploratory follow-up after a no-edit run on the same development partitions; not an independent replication or a sharing-only ablation',
            'pricing_verified_date': '2026-11-07', 'pricing_expires': '2026-12-01',
            'source_hashes': {str(p.resolve()): digest(p) for p in sources}}
    write_frozen(out / 'plan.json', json.dumps(plan, indent=2) + '\n')
    return plan


def authorize(out, plan):
    approval = json.loads((out / 'approval.json').read_text())
    if (approval.get('approved') is not True or approval.get('plan_sha256') != digest(out / 'plan.json')
            or approval.get('caps') != plan.get('caps', CAPS) or not approval.get('user_message')):
        raise ValueError('Approval must match this exact plan and spending caps')
    if datetime.now(UTC).date().isoformat() >= plan['pricing_expires']:
        raise ValueError('Pricing verification expired; prepare a newly reviewed plan')


def preflight(out, plan):
    runner = FinqaLM(plan['model_path'], out / 'generation-status.json')
    sizes = {}
    for partition in ('practice', 'selection', 'evaluation'):
        rows = read_jsonl(out / f'{partition}-private.jsonl')
        lengths = [len(runner.tok(runner.render(r['problem']), add_special_tokens=False).input_ids) for r in rows]
        sizes[partition] = {'items': len(rows), 'max_prompt_tokens': max(lengths),
                            'minimum_output_allowance': 32768 - max(lengths)}
    if min(r['minimum_output_allowance'] for r in sizes.values()) < 8192:
        raise ValueError('Preflight: a prompt leaves insufficient output allowance')
    result = {'plan_sha256': digest(out / 'plan.json'), 'sizes': sizes, 'paid_api_calls': 0,
              'key_files_present': {p: Path(f'/tmp/{p}_key.txt').is_file() for p in paid.MODELS}}
    write_json(out / 'preflight.json', result)
    return result


class SerialRunner(FinqaLM):
    """Serialize all GPU operations; cache only unchanged-model responses across arms."""
    def __init__(self, model, out, lock, cache, fingerprint, control=None):
        super().__init__(model, out / 'generation-status.json')
        self.out, self.lock, self.shared_cache, self.fingerprint = out, lock, cache, fingerprint
        self.base = Path(model).resolve()
        self.calls = 0
        self.control = control

    def check(self):
        if self.control is not None:
            self.control.check()

    def generate(self, weights, prompts, n, sampling):
        with self.lock:
            self.check()
            base = Path(weights).resolve() == self.base
            keys = [GenerationCache.key(self.fingerprint, p, n, sampling, 'finqa-vllm-fixed') for p in prompts]
            outputs = [self.shared_cache.get(k) if base else None for k in keys]
            missing = [i for i, r in enumerate(outputs) if r is None]
            if missing:
                fresh = super().generate(weights, [prompts[i] for i in missing], n, sampling)
                for i, result in zip(missing, fresh):
                    outputs[i] = result
                    if base:
                        self.shared_cache.put(keys[i], result)
            self.calls += 1
            archive = self.out / 'private-generations'
            archive.mkdir(exist_ok=True)
            write_json(archive / f'{self.calls:04d}.json', {
                'weights': str(weights), 'reused_original': len(prompts) - len(missing),
                'items': [{'id': p.id, 'seed': p.seed, 'samples':
                           [{'text': g.text, 'finish': g.finish_reason, 'ids': g.token_ids} for g in result]}
                          for p, result in zip(prompts, outputs)]})
            return outputs

    def read(self, *args, **kwargs):
        with self.lock:
            self.check()
            return super().read(*args, **kwargs)

    def activations(self, *args, **kwargs):
        with self.lock:
            self.check()
            return super().activations(*args, **kwargs)


def suite(out, partition):
    return LMSuite(read_jsonl(out / 'practice-private.jsonl'), read_jsonl(out / f'{partition}-private.jsonl'),
                   FinqaGrader(), SAMPLING, benchmark=f'FinQA {partition} reports', samples=1,
                   protocol='Original full context, thinking on, unchanged scalar grader; .6/.95/top-k 20, '
                            'no presence penalty; 32768 total context with full remaining output allowance.')


def run_editors(root, out):
    import wsplab
    from wsplab.kernel import Limits
    from wsplab.passes import ForwardLimits
    from wsplab.patch.format import Patch
    from wsplab.session.driver import Policy

    plan = prepare(root, out)
    authorize(out, plan)
    ready = json.loads((out / 'preflight.json').read_text())
    if (ready['plan_sha256'] != digest(out / 'plan.json')
            or min(r['minimum_output_allowance'] for r in ready['sizes'].values()) < 8192):
        raise ValueError('Matching successful prompt preflight required before spending')
    if (out / 'editors-started.json').exists():
        raise ValueError('This trial already started; do not silently repeat paid sessions')
    for provider in paid.MODELS:
        if not Path(f'/tmp/{provider}_key.txt').is_file():
            raise ValueError(f'Missing configured {provider} key file')
    scratch = Path(plan['scratch_root'])
    if scratch.exists():
        raise ValueError('Scratch folder already exists; preserve it and investigate before spending')
    if shutil.disk_usage(scratch.parent).free < 20 * 2**30:
        raise ValueError('Need 20 GiB scratch space for candidate checkpoints before spending')
    scratch.mkdir()
    # The reused guarded ledger resolves rates from this module at request time.
    paid.PRICES = PRICES
    write_json(out / 'editors-started.json', {'at': time.time(), 'plan_sha256': digest(out / 'plan.json')})
    lock = threading.RLock()
    cache = GenerationCache(out / 'original-generations-private.jsonl')
    results = {}
    for arm in ('peers', 'solo'):
        providers = list(plan['caps'][arm])
        folder = out / arm
        folder.mkdir(exist_ok=False)
        ledger = paid.Ledger(folder / 'ledger.json', plan['caps'][arm])
        ledger.save()
        board = ConfirmationBoard(len(providers)) if arm == 'peers' else None
        control = ArmControl(len(providers), board)
        runners, editors = [], []
        for index, provider in enumerate(providers):
            dest = folder / provider
            dest.mkdir()
            transient = scratch / arm / provider
            transient.mkdir(parents=True)
            (dest / 'run').mkdir()
            for name in ('checkpoints', 'result'):
                (transient / name).mkdir()
                (dest / 'run' / name).symlink_to(transient / name, target_is_directory=True)
            runners.append(SerialRunner(plan['model_path'], dest, lock, cache,
                                        digest(out / 'plan.json'), control))
        warm_tokenizers(runners)
        for index, provider in enumerate(providers):
            editor = ReadyEditor(paid.Guarded(paid.editor(provider), provider, ledger, max_output=16384), control)
            if board is not None:
                editor = ConfirmationPeer(editor, board, index, plan['partitions']['practice']['ids'][32:])
            editor = InterventionEditor(editor, dest=folder / provider,
                                        dev_ids=plan['partitions']['practice']['ids'][:32],
                                        holdout_ids=plan['partitions']['practice']['ids'][32:],
                                        required=SETTINGS[arm]['required_candidates'],
                                        diagnostics=SETTINGS[arm]['diagnostic_passes'])
            editors.append(editor)
        write_json(folder / 'startup.json', {'tokenizers_ready': providers, 'at': time.time()})

        def one(index, providers=tuple(providers), folder=folder, runners=runners, editors=editors,
                control=control, board=board, arm=arm):
            try:
                provider = providers[index]
                dest = folder / provider
                runner, editor = runners[index], editors[index]
                settings = SETTINGS[arm]
                result = wsplab.run_lm(
                    plan['model_path'], eval=suite(out, 'selection'), editor=editor, note=NOTE,
                    thinking=True, budget=settings['passes_each'], max_runs=settings['runs_each'],
                    folder=dest / 'run', runner=runner, control=False,
                    policy=Policy(handoff_every=10**9, require_all_passes=True, max_nudges=3, warn_unsaved=14,
                                  max_turns=settings['turns_each']), limits=Limits(seconds=180, memory_gb=6),
                    forward_limits=ForwardLimits(n_layers=28, full_attention=tuple(range(28)),
                                                 gen_max=64, max_tokens=32768))
                editor.validate_finished()
                changed = Patch.load(result.patch).manifest['changed_entries_total']
                durable_patch = dest / 'submitted-patch'
                shutil.copytree(result.patch, durable_patch)
                row = {'provider': provider, 'score': result.final if changed else result.baseline,
                       'baseline': result.baseline, 'changed_entries': changed, 'patch': str(durable_patch),
                       'weights': str(result.weights), 'passes': result.state.passes,
                       'runs': result.state.runs, 'stop_reason': result.state.stop_reason,
                       'selection_verdict_only': result.verdict, 'peer_rounds': getattr(editor, 'rounds', None),
                       'interventions_tested': len(editor.tested), 'intervention_status': editor.snapshot()}
                write_json(dest / 'candidate.json', row)
                return row
            except BaseException as exc:
                control.abort()
                write_json(folder / providers[index] / 'failure.json',
                           {'type': type(exc).__name__, 'at': time.time()})
                raise
            finally:
                if board is not None:
                    board.leave(index)

        with ThreadPoolExecutor(len(providers)) as pool:
            futures = {pool.submit(one, i): i for i in range(len(providers))}
            collected = {futures[f]: f.result() for f in as_completed(futures)}
            candidates = [collected[i] for i in range(len(providers))]
        baseline = candidates[0]['baseline']
        if any(c['baseline'] != baseline for c in candidates):
            raise ValueError('Baseline mismatch across sessions')
        selected = choose_winner(candidates, baseline)
        results[arm] = {'candidates': candidates, 'baseline': baseline, 'selected': selected,
                        'spent_estimate': ledger.spent, 'arm_cap': sum(plan['caps'][arm].values())}
        write_json(folder / 'selection.json', results[arm])
    prepare(root, out)
    write_json(out / 'selections-frozen.json', results)
    write_json(out / 'status.json', {'status': 'awaiting_protocol_audit', 'at': time.time(),
                                    'external_evaluation_used': False})


def compare(a, b):
    ids = sorted(a.per_item)
    if ids != sorted(b.per_item):
        raise ValueError('Paired scores have different IDs')
    result = paired([a.per_item[i] for i in ids], [b.per_item[i] for i in ids]).as_dict()
    n = result['wrong_to_right'] + result['right_to_wrong']
    k = min(result['wrong_to_right'], result['right_to_wrong'])
    result['exact_two_sided_p'] = min(1., 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.
    return result


def evaluate(root, out):
    import tempfile

    from wsplab import patch as patches
    from wsplab.eval.gate import random_control
    from wsplab.io.safetensors import Checkpoint

    plan = prepare(root, out)
    audit = json.loads((out / 'protocol-audit.json').read_text())
    if (audit.get('all_sessions_valid') is not True or audit.get('plan_sha256') != digest(out / 'plan.json')
            or audit.get('selections_sha256') != digest(out / 'selections-frozen.json')):
        raise ValueError('A valid protocol audit tied to frozen selections is required')
    if (out / 'external-started.json').exists():
        raise ValueError('External evaluation already started; do not rerun silently')
    selected = json.loads((out / 'selections-frozen.json').read_text())
    write_json(out / 'external-started.json', {'at': time.time()})
    runner = SerialRunner(plan['model_path'], out, threading.RLock(),
                          GenerationCache(out / 'original-generations-private.jsonl'), digest(out / 'plan.json'))
    evaluator = LMEvaluation(runner, suite(out, 'evaluation'))
    base = Path(plan['model_path'])
    original = evaluator.evaluate(base)
    scores = {}
    controls = {}
    for arm in ('peers', 'solo'):
        candidate = selected[arm]['selected']
        scores[arm] = evaluator.evaluate(Path(candidate['weights'])) if candidate else original
        if candidate and scores[arm].mean > original.mean:
            patch = patches.Patch.load(candidate['patch'])
            with tempfile.TemporaryDirectory(dir=plan['scratch_root']) as temp:
                temp = Path(temp)
                noise = random_control(Checkpoint(base), patch, Checkpoint(candidate['weights']))
                randomized = patches.from_submission(Checkpoint(base), noise, temp / 'patch')
                model = patches.apply(randomized, Checkpoint(base), temp / 'model')
                controls[arm] = compare(evaluator.evaluate(model), scores[arm])
    primary = compare(scores['solo'], scores['peers'])
    original_comparisons = {k: compare(original, v) for k, v in scores.items()}

    def supported(result):
        return result is not None and result['difference'] > 0 and result['exact_two_sided_p'] < .05

    result = {'status': 'completed', 'original': original.mean, 'peers': scores['peers'].mean,
              'solo': scores['solo'].mean, 'peers_minus_solo': primary,
              'arms_minus_original': original_comparisons,
              'random_controls': controls,
              'peer_advantage_detected': primary['difference'] > 0 and primary['exact_two_sided_p'] < .05,
              'council_weight_edit_win': all(supported(r) for r in
                                             (primary, original_comparisons['peers'], controls.get('peers'))),
              'scope': 'one exploratory trial versus preselected Claude solo; not superiority over all solo models',
              'selection': selected}
    prepare(root, out)
    write_json(out / 'external-results.json', result)
    write_json(out / 'status.json', {'status': 'completed', 'at': time.time(), 'external_evaluation_used': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.wsplab/natural-finqa'))
    parser.add_argument('--out', type=Path, default=Path('.wsplab/finqa-intervention-comparison'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run-editors', action='store_true')
    mode.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    if args.run_editors:
        run_editors(args.root.resolve(), args.out.resolve())
    elif args.evaluate:
        evaluate(args.root.resolve(), args.out.resolve())
    else:
        plan = prepare(args.root.resolve(), args.out.resolve())
        checked = preflight(args.out.resolve(), plan)
        print(json.dumps({'status': plan['status'], 'counts': {k: v['n'] for k, v in plan['partitions'].items()},
                          'caps': plan['caps'], 'total_additional_usd': plan['total_additional_usd'],
                          'prompt_preflight': checked['sizes'], 'key_files_present': checked['key_files_present']}))
