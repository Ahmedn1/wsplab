import importlib.util
import os
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'notify_watch', Path(__file__).resolve().parents[1] / "code" / "notify_watch.py")
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def fixture_job(tmp_path):
    monitor = tmp_path / 'monitor.json'
    job = {'id': 'test', 'thread': 'thread', 'pid': os.getpid(),
           'start_ticks': w.identity(os.getpid()), 'created_at': 100,
           'monitor': str(monitor), 'log': None, 'stall_seconds': 1800,
           'max_seconds': 21600}
    w.write(tmp_path / 'jobs/test.json', job)
    return job, monitor


def test_terminal_delivered_once_across_restart(tmp_path):
    _, monitor = fixture_job(tmp_path)
    w.write(monitor, {'status': 'awaiting_protocol_audit', 'at': 100})
    sent = []
    def send(*args):
        sent.append(args)
        return 'Queued message receipt'
    w.tick(tmp_path, 'codex', 100, send)
    w.tick(tmp_path, 'codex', 200, send)
    assert len(sent) == 1
    assert 'awaiting_protocol_audit' in sent[0][-1]
    assert w.read(tmp_path / 'state/test.json')['closed']


def test_failure_retries_without_marking_delivered(tmp_path):
    _, monitor = fixture_job(tmp_path)
    w.write(monitor, {'status': 'failed', 'at': 100})
    calls = []
    def send(*args):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError('daemon unavailable')
        return 'Queued message receipt'
    w.tick(tmp_path, 'codex', 100, send)
    assert not w.read(tmp_path / 'state/test.json').get('closed')
    w.tick(tmp_path, 'codex', 130, send)
    assert len(calls) == 1
    w.tick(tmp_path, 'codex', 161, send)
    assert len(calls) == 2
    assert w.read(tmp_path / 'state/test.json')['closed']


def test_pid_reuse_is_exit(tmp_path):
    job, monitor = fixture_job(tmp_path)
    job['start_ticks'] = 'not-this-process'
    w.write(tmp_path / 'jobs/test.json', job)
    w.write(monitor, {'status': 'running', 'at': 100})
    sent = []
    w.tick(tmp_path, 'codex', 100, lambda *args: sent.append(args) or 'receipt')
    assert 'process_exited' in sent[0][-1]


def test_heartbeat_does_not_mask_stall(tmp_path):
    job, _ = fixture_job(tmp_path)
    state = {}
    assert w.event(job, {'status': 'running', 'at': 100}, state, 100, True) is None
    assert w.event(job, {'status': 'running', 'at': 2001}, state, 2001, True)[0] == 'progress_stalled'
    snapshot = {'status': 'running', 'at': 2002, 'sessions': {'peer': {'state': {'passes': 1}}}}
    assert w.event(job, snapshot, state, 2002, True) is None


def test_stale_monitor_and_reported_failure(tmp_path):
    job, _ = fixture_job(tmp_path)
    assert w.event(job, None, {}, 500, True)[0] == 'stale_monitor'
    assert w.event(job, {'failures': {'peer': 'failed'}}, {}, 500, True)[0] == 'failure_reported'


def test_short_script_has_terminal_status(tmp_path):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, str(Path(w.__file__)), '--folder', str(tmp_path),
                             'run', '--id', 'short', '--thread', 'test', '--',
                             sys.executable, '-c', 'raise SystemExit(7)'])
    assert result.returncode == 7
    assert w.read(tmp_path / 'runs/short/monitor.json')['status'] == 'failed'
