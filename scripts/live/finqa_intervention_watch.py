"""Launch the approved editor trial and persist progress without exposing test answers."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    root = Path('.wsplab/finqa-intervention-comparison').resolve()
    if (root / 'watcher-launch.json').exists():
        raise RuntimeError('Watcher already launched; inspect the existing process')
    (root / 'watcher-launch.json').write_text(json.dumps({'pid': os.getpid(), 'at': time.time()}) + '\n')
    plan = read(root / 'plan.json')
    previous_spent = (plan.get('recovery') or {}).get('previous_total', 0.)
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
           'VLLM_USE_FLASHINFER_SAMPLER': '0', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'OMP_NUM_THREADS': '2',
           'VLLM_CACHE_ROOT': '/tmp/wsplab-finqa-intervention-comparison/cache/vllm',
           'TRITON_CACHE_DIR': '/tmp/wsplab-finqa-intervention-comparison/cache/triton',
           'TORCHINDUCTOR_CACHE_DIR': '/tmp/wsplab-finqa-intervention-comparison/cache/torchinductor'}
    with (root / 'editors.log').open('w') as log:
        process = subprocess.Popen([sys.executable, 'scripts/live/finqa_intervention_trial.py', '--run-editors'],
                                   env=env, stdout=log, stderr=subprocess.STDOUT)
        while True:
            code = process.poll()
            snapshot = {'status': 'editors_running' if code is None else
                        ('awaiting_protocol_audit' if code == 0 else 'failed'),
                        'pid': process.pid, 'returncode': code, 'at': time.time(), 'sessions': {}, 'spending': {},
                        'external_evaluation_used': False}
            snapshot['previous_attempt_spent'] = previous_spent
            snapshot['startup'] = {}
            snapshot['failures'] = {}
            for arm in ('peers', 'solo'):
                snapshot['startup'][arm] = read(root / arm / 'startup.json')
                ledger = read(root / arm / 'ledger.json')
                if ledger:
                    snapshot['spending'][arm] = {k: ledger[k] for k in ('caps', 'spent', 'reserved')}
                for provider in ('openai', 'anthropic', 'gemini'):
                    folder = root / arm / provider
                    if folder.exists():
                        failure = read(folder / 'failure.json')
                        if failure:
                            snapshot['failures'][f'{arm}/{provider}'] = failure
                        state = read(folder / 'run/session/state.json')
                        snapshot['sessions'][f'{arm}/{provider}'] = {
                            'state': {k: state.get(k) for k in ('turn', 'runs', 'passes', 'done', 'stop_reason')}
                            if state else {'phase': 'initializing_or_private_baseline'},
                            'generation': read(folder / 'generation-status.json'),
                            'interventions': read(folder / 'intervention-status.json')}
            if snapshot['failures'] and code is None:
                snapshot['status'] = 'stopping_after_failure'
            snapshot['total_spent_including_previous_attempt'] = previous_spent + sum(
                sum(ledger['spent'].values()) for ledger in snapshot['spending'].values())
            temp = root / 'monitor.tmp'
            temp.write_text(json.dumps(snapshot, indent=2) + '\n')
            temp.replace(root / 'monitor.json')
            if code is not None:
                break
            time.sleep(10)


if __name__ == '__main__':
    main()
