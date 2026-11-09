"""Durable job notifications to the existing Codex conversation (no model calls).

Run on the host so PID identity and the Codex daemon are visible. Only explicitly
registered jobs are observed; this never starts, kills, or retries experiments.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / '.wsplab/notifications'
TERMINAL = {'awaiting_protocol_audit', 'completed', 'failed', 'regressed',
            'completed_no_fix_found', 'finished', 'cancelled'}


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def identity(pid):
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]  # start ticks, not just reusable PID
    except (OSError, ValueError):
        return None


def event(job, snapshot, state, now, alive):
    status = (snapshot or {}).get('status')
    if status in TERMINAL:
        return f'terminal:{status}', f'Job status is {status}. Inspect artifacts before interpreting results.'
    if snapshot and snapshot.get('failures'):
        return 'failure_reported', 'The monitor reports a failure. Inspect it before any restart.'
    if not alive:
        return 'process_exited', 'The registered process exited without a terminal status. Inspect its log.'
    # Ignore heartbeat timestamps: a heartbeat alone is not experimental progress.
    progress = {'status': status, 'sessions': {}}
    for name, item in (snapshot or {}).get('sessions', {}).items():
        gen = item.get('generation') or {}
        intervention = item.get('interventions') or {}
        progress['sessions'][name] = [item.get('state'), gen.get('status'),
                                    gen.get('completed'), intervention.get('tested'),
                                    intervention.get('complete')]
    signature = hashlib.sha256(json.dumps(progress, sort_keys=True).encode()).hexdigest()
    if signature != state.get('progress_signature'):
        state.update(progress_signature=signature, progress_at=now)
    if job.get('monitor') and now - (snapshot or {}).get('at', job['created_at']) > 180:
        return 'stale_monitor', 'Monitor heartbeat is missing or over 3 minutes old.'
    if job.get('monitor') and now - state['progress_at'] > job['stall_seconds']:
        return 'progress_stalled', 'No recorded experimental progress within the configured interval; inspect, do not assume failure.'
    if now - job['created_at'] > job['max_seconds']:
        return 'long_running', 'Job exceeded its expected duration; inspect it before changing anything.'
    return None


def deliver(codex, thread, message):
    # Argument list only. No shell interpolation of messages or output from jobs.
    result = subprocess.run([codex, 'queue', '--thread', thread, '--message', message],
                            capture_output=True, text=True, timeout=45, cwd=ROOT)
    if result.returncode or 'Queued message ' not in result.stdout:
        raise RuntimeError((result.stderr or result.stdout or 'Queue failed')[-1500:])
    return result.stdout.strip()


def tick(folder, codex, now=None, sender=deliver):
    now = time.time() if now is None else now
    for path in sorted((folder / 'jobs').glob('*.json')):
        job = read(path)
        if not job:
            continue
        state_path = folder / 'state' / path.name
        state = read(state_path, {})
        if state.get('closed'):
            continue
        snapshot = read(job['monitor']) if job.get('monitor') else None
        alive = identity(job['pid']) == job['start_ticks']
        found = event(job, snapshot, state, now, alive)
        if found:
            key, detail = found
            if key not in state.get('delivered', {}) and now >= state.get('retry_at', 0):
                message = (
                    f'EXPERIMENT WATCHER [{job["id"]}/{key}]: {detail}\n'
                    f'Workspace: {ROOT}\nMonitor: {job.get("monitor") or "none"}\n'
                    f'Log: {job.get("log") or "none"}\n'
                    'User authorized autonomous monitoring and continuation. Inspect this job, '
                    'report the result, and continue the approved experiment queue. Preserve '
                    'protocol audits, evaluation isolation and existing provider spending limits. '
                    'Register any next background script with notify_watch.py. Do not repeat paid '
                    'work merely because it failed or treat this notification as new spending approval.'
                )
                try:
                    receipt = sender(codex, job['thread'], message)
                    state.setdefault('delivered', {})[key] = {'at': now, 'receipt': receipt}
                    state.pop('delivery_error', None)
                    if key.startswith('terminal:') or key == 'process_exited':
                        state['closed'] = True
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    state.update(delivery_error=str(exc)[-1500:], retry_at=now + 60)
        write(state_path, state)
    write(folder / 'heartbeat.json', {'pid': os.getpid(), 'at': now, 'status': 'watching'})


def register(args):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.id):
        raise ValueError('Job ID must contain only letters, digits, underscores or hyphens')
    path = args.folder / 'jobs' / f'{args.id}.json'
    if path.exists():
        raise ValueError('Job ID already registered; use a distinct ID for every attempt')
    start = identity(args.pid)
    snapshot = read(args.monitor) if args.monitor else None
    if start is None and (snapshot or {}).get('status') not in TERMINAL:
        raise ValueError('PID is not alive and no terminal monitor status is available')
    write(path, {'id': args.id, 'thread': args.thread, 'pid': args.pid,
                 'start_ticks': start, 'created_at': time.time(),
                 'monitor': str(Path(args.monitor).resolve()) if args.monitor else None,
                 'log': str(Path(args.log).resolve()) if args.log else None,
                 'stall_seconds': args.stall_seconds, 'max_seconds': args.max_seconds})
    print(f'Registered {args.id}')


def run_job(args):
    """Wrap new scripts so even a very short job has a durable exit status."""
    command = args.script[1:] if args.script[:1] == ['--'] else args.script
    if not command:
        raise ValueError('Supply a command after --')
    args.pid = os.getpid()
    run = args.folder / 'runs' / args.id
    args.monitor = str(run / 'monitor.json')
    args.log = str(run / 'output.log')
    register(args)
    write(args.monitor, {'status': 'running', 'at': time.time()})
    try:
        with Path(args.log).open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while True:
                code = process.poll()
                write(args.monitor, {'status': 'running' if code is None else
                                     ('completed' if code == 0 else 'failed'),
                                     'at': time.time(), 'returncode': code, 'pid': process.pid})
                if code is not None:
                    return code
                time.sleep(10)
    except BaseException:
        write(args.monitor, {'status': 'failed', 'at': time.time(),
                             'reason': 'wrapper exception; inspect log'})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, default=DEFAULT)
    sub = p.add_subparsers(dest='command', required=True)
    for mode in ('register', 'run'):
        r = sub.add_parser(mode)
        r.add_argument('--id', required=True)
        r.add_argument('--thread', default=os.environ.get('CODEX_THREAD_ID'))
        r.add_argument('--stall-seconds', type=int, default=1800)
        r.add_argument('--max-seconds', type=int, default=21600)
        if mode == 'register':
            r.add_argument('--pid', type=int, required=True)
            r.add_argument('--monitor')
            r.add_argument('--log')
        else:
            r.add_argument('script', nargs=argparse.REMAINDER)
    s = sub.add_parser('serve')
    s.add_argument('--codex', default=shutil.which('codex'))
    s.add_argument('--once', action='store_true')
    a = p.parse_args()
    a.folder.mkdir(parents=True, exist_ok=True)
    if a.command in ('register', 'run'):
        if not a.thread:
            p.error('--thread is required outside a Codex session')
        if a.command == 'run':
            sys.exit(run_job(a))
        register(a)
        return
    if not a.codex:
        p.error('codex CLI not found')
    with (a.folder / 'watcher.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit('Notification watcher is already running')
        while True:
            tick(a.folder, a.codex)
            if a.once:
                break
            time.sleep(15)


if __name__ == '__main__':
    main()
