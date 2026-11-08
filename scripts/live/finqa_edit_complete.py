"""Complete the already-precommitted local pilot; no new candidates or paid calls."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def main(root):
    work = root / 'run/work'
    status = root / 'completion-status.json'
    executable = Path(sys.executable).with_name('wsplab')

    def command(args, label):
        write(status, {'status': label, 'at': time.time(), 'paid_api_calls': 0})
        with (root / f'completion-{label}.log').open('w') as output:
            subprocess.run([str(executable), 'session', *args], stdout=output,
                           stderr=subprocess.STDOUT, check=True)

    write(status, {'status': 'waiting_for_second_candidate', 'at': time.time(), 'paid_api_calls': 0})
    deadline = time.monotonic() + 7200
    while True:
        state = json.loads((root / 'run/session/state.json').read_text())
        if state['passes'] == 3:
            break
        if state['done'] or state['passes'] > 3 or time.monotonic() > deadline:
            raise RuntimeError('Unexpected session state or timeout; do not continue automatically')
        time.sleep(5)
    if (work / 'freeze.json').exists():
        raise RuntimeError('Already frozen: refusing to execute a duplicate completion')
    command(['exec', str(root), '--why',
             ('Both precommitted DEV interventions have now been measured. I will apply the frozen '
             'retention rule using exact paired statistics, restore original weights if neither qualifies, '
             'and make no further candidate choices after this decision.'),
             '--file', str(root / 'commands/freeze.py')], 'freeze')
    frozen = json.loads((work / 'freeze.json').read_text())
    first = json.loads((work / 'candidate_1_dev.json').read_text())
    second = json.loads((work / 'candidate_2_dev.json').read_text())
    summary = {'candidate_1': first, 'candidate_2': second, 'frozen': frozen}
    explanation = (
        'Tested two precommitted direct weight edits to column 377 of '
        'model.layers.21.mlp.down_proj.weight: full ablation and half attenuation. '
        f'Original DEV score was {first["original_correct"]}/16; ablation scored '
        f'{first["edited_correct"]}/16 and half attenuation {second["edited_correct"]}/16. '
        'The candidate was motivated by a failure-associated recorded activation, not a proven '
        'semantic mechanism. No training, manual model replay, prompt change, or paid API call was used. '
    )
    if frozen['decision'] == 'retain_for_confirmation':
        command(['forward', str(root), '--why',
                 ('The edit is frozen from DEV under the predeclared rule. This is the only internal '
                 'HOLDOUT comparison on the next 16 practice records. Its result will be reported '
                 'without changing the chosen weights; private selection evaluation follows submission.'),
                 '--file', str(work / 'holdout-request.json')], 'holdout')
        state = json.loads((root / 'run/session/state.json').read_text())
        if state['passes'] != 4:
            raise RuntimeError('HOLDOUT pass did not complete')
        from wsplab.eval.stats import paired
        a = [json.loads(line) for line in (work / 'gen_4_original.jsonl').read_text().splitlines()]
        b = [json.loads(line) for line in (work / 'gen_4_edited.jsonl').read_text().splitlines()]
        reserved = json.loads((work / 'splits.json').read_text())['holdout']
        if [r['id'] for r in a] != [r['id'] for r in b] or [r['id'] for r in a] != reserved:
            raise RuntimeError('HOLDOUT IDs do not match the frozen reservation')
        comparison = paired([[float(r['correct'])] for r in a], [[float(r['correct'])] for r in b])
        summary['holdout'] = comparison.as_dict()
        explanation += (f'Submitting the frozen scale {frozen["scale"]} candidate. '
                        f'Internal HOLDOUT difference was {comparison.difference:+.4f}, '
                        f'SE {comparison.standard_error:.4f}, with {comparison.wrong_to_right} fixes '
                        f'and {comparison.right_to_wrong} regressions. No post-HOLDOUT tuning occurred. ')
    elif frozen['decision'] == 'restore_original':
        summary['holdout_used'] = False
        explanation += ('Neither candidate met the DEV retention rule, so all edits were dropped. '
                        'HOLDOUT was not consumed to compare identical weights. Submitting original '
                        'weights is a negative result for this attempt, not evidence that all direct '
                        'weight edits or a peer council must fail. ')
    else:
        raise RuntimeError('Unknown frozen decision')
    explanation += 'The reserved external evaluation was not used. This is a solo local pilot, not a council comparison.'
    write(root / 'pilot-analysis.json', summary)
    final = root / 'submission.txt'
    final.write_text(explanation + '\n')
    with (work / 'notes.md').open('a') as notes:
        notes.write('\nFINAL SUBMISSION\n' + explanation + '\n')
    command(['finish', str(root), '--file', str(final)], 'finish')
    outcome = json.loads((root / 'summary.json').read_text())
    if outcome['status'] != 'completed':
        raise RuntimeError('Session did not report successful completion')
    write(status, {'status': 'completed', 'verdict': outcome['verdict'], 'at': time.time(), 'paid_api_calls': 0})


if __name__ == '__main__':
    folder = Path('.wsplab/fq-edit1').resolve()
    try:
        main(folder)
    except Exception as exc:
        write(folder / 'completion-status.json', {'status': 'failed', 'error': str(exc), 'at': time.time()})
        raise
