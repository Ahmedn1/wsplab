"""Offline checks for fair limits, private selection, cached inference and spend gates."""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import finqa_peer_trial as trial

from wsplab.editors.base import ScriptedEditor
from wsplab.eval.suites import Scores
from wsplab.passes import GenerationCache
from wsplab.runners.base import Generation, Prompt


def test_confirmation_results_and_later_messages_never_reach_other_peers():
    board = trial.ConfirmationBoard(2)
    a = ScriptedEditor(['```forward\n{"type":"generate","problems":["holdout"]}\n```',
                        'I saw secret result 99.', 'finished'])
    b = ScriptedEditor(['DEV hypothesis', 'still tuning DEV', 'finished'])
    peers = [trial.ConfirmationPeer(e, board, i, ['holdout']) for i, e in enumerate((a, b))]
    chats = [[{'role': 'user', 'content': 'begin'}] for _ in peers]
    for step in range(3):
        with ThreadPoolExecutor(2) as pool:
            replies = list(pool.map(lambda pair: pair[0].reply('task', pair[1]), zip(peers, chats)))
        for i, reply in enumerate(replies):
            chats[i] += [{'role': 'assistant', 'content': reply.text},
                         {'role': 'user', 'content': 'secret result 99' if i == 0 else 'DEV score 5'}]
    assert 'secret result 99' in str(a.seen)
    assert 'secret result 99' not in str(b.seen)
    assert 'withheld' in str(b.seen)
    assert 'DEV score 5' in str(a.seen)


def test_equal_arm_ceilings_and_all_three_peers():
    assert set(trial.CAPS['peers']) == {'openai', 'anthropic', 'gemini'}
    assert list(trial.CAPS['solo']) == ['anthropic']
    assert sum(trial.CAPS['peers'].values()) == sum(trial.CAPS['solo'].values()) == 12
    for key in ('passes_each', 'runs_each', 'turns_each'):
        assert 3 * trial.SETTINGS['peers'][key] == trial.SETTINGS['solo'][key]


def test_failed_startup_unblocks_waiting_editors_without_paid_calls():
    control = trial.ArmControl(3)
    delegates = [ScriptedEditor(['must not be called']) for _ in range(2)]
    editors = [trial.ReadyEditor(e, control) for e in delegates]
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(e.reply, 'system', []) for e in editors]
        deadline = time.monotonic() + 2
        while control.ready.n_waiting < 2 and time.monotonic() < deadline:
            time.sleep(.005)
        control.abort()  # Third participant fails initialization.
        for f in futures:
            with pytest.raises(threading.BrokenBarrierError):
                f.result(timeout=2)
    assert all(not e.seen for e in delegates)


def test_board_abort_unblocks_waiters_and_blocks_future_rounds():
    board = trial.ConfirmationBoard(3)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(board.exchange, 0, {'reply': 'DEV', 'result': 'x'})
        board.abort()
        with pytest.raises(RuntimeError, match='aborted'):
            future.result(timeout=2)
    with pytest.raises(RuntimeError, match='aborted'):
        board.exchange(1, {})


def test_warmup_initializes_tokenizers_before_parallel_rendering():
    main_thread = threading.get_ident()

    class Runner:
        initialized = False

        @property
        def tok(self):
            if not self.initialized:
                assert threading.get_ident() == main_thread
                self.initialized = True
            return 'ready'

    runners = [Runner() for _ in range(3)]
    trial.warm_tokenizers(runners)
    with ThreadPoolExecutor(3) as pool:
        assert list(pool.map(lambda r: r.tok, runners)) == ['ready'] * 3


def test_recovery_deducts_failed_spending_and_matches_remaining_arm_caps(tmp_path):
    old = tmp_path / 'previous'
    (old / 'peers').mkdir(parents=True)
    ledger = {'caps': trial.CAPS['peers'], 'spent': {'openai': .31, 'anthropic': 0., 'gemini': 2.01},
              'reserved': dict.fromkeys(trial.CAPS['peers'], 0.)}
    path = old / 'peers/ledger.json'
    path.write_text(json.dumps(ledger))
    (tmp_path / 'recovery.json').write_text(json.dumps(
        {'previous_run': str(old), 'ledger_sha256': trial.digest(path)}))
    caps, recovery = trial.spending_caps(tmp_path)
    assert sum(caps['peers'].values()) == pytest.approx(caps['solo']['anthropic'])
    assert caps['solo']['anthropic'] == pytest.approx(9.68)
    assert recovery['previous_total'] + sum(sum(c.values()) for c in caps.values()) <= 24
    for p in ledger['spent']:
        assert ledger['spent'][p] + caps['peers'][p] <= trial.CAPS['peers'][p]
    path.write_text('{}')
    with pytest.raises(ValueError, match='ledger changed'):
        trial.spending_caps(tmp_path)


def test_selection_keeps_original_over_regressions_and_ties():
    candidates = [{'provider': 'a', 'score': .4, 'changed_entries': 100},
                  {'provider': 'b', 'score': .5, 'changed_entries': 10}]
    assert trial.choose_winner(candidates, .5) is None
    candidates += [{'provider': 'c', 'score': .6, 'changed_entries': 30},
                   {'provider': 'd', 'score': .6, 'changed_entries': 20}]
    assert trial.choose_winner(candidates, .5)['provider'] == 'd'


def test_spending_requires_approval_for_exact_plan_and_caps(tmp_path):
    (tmp_path / 'plan.json').write_text('{}')
    plan = {'pricing_expires': '2999-01-01'}
    with pytest.raises(FileNotFoundError):
        trial.authorize(tmp_path, plan)
    approval = {'approved': True, 'plan_sha256': trial.digest(tmp_path / 'plan.json'),
                'caps': trial.CAPS, 'user_message': 'Test fixture only; not real authorization'}
    (tmp_path / 'approval.json').write_text(json.dumps(approval))
    trial.authorize(tmp_path, plan)
    (tmp_path / 'plan.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='Approval'):
        trial.authorize(tmp_path, plan)


def test_external_evaluation_requires_audit_before_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(trial, 'prepare', lambda *args: {})
    with pytest.raises(FileNotFoundError):
        trial.evaluate(tmp_path, tmp_path)
    assert not (tmp_path / 'external-started.json').exists()


def test_pairing_rejects_changed_ids_and_reports_exact_test():
    a = Scores({'a': [0.], 'b': [1.]})
    b = Scores({'a': [1.], 'b': [1.]})
    result = trial.compare(a, b)
    assert result['wrong_to_right'] == 1 and result['difference'] == .5
    assert result['exact_two_sided_p'] == 1
    with pytest.raises(ValueError, match='different IDs'):
        trial.compare(a, Scores({'c': [1.], 'b': [1.]}))


def test_gpu_serialization_and_original_cache_never_substitutes_for_edits(tmp_path, monkeypatch):
    calls, active, maximum = [], 0, 0

    def generate(self, weights, prompts, n, sampling):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        calls.append(str(weights))
        time.sleep(.02)
        active -= 1
        return [[Generation(str(weights), 'stop', [1])] for p in prompts]

    monkeypatch.setattr(trial.FinqaLM, 'generate', generate)
    model = tmp_path / 'base'
    lock = threading.RLock()
    cache = GenerationCache(tmp_path / 'cache.jsonl')
    runners = []
    for name in ('a', 'b'):
        out = tmp_path / name
        out.mkdir()
        runners.append(trial.SerialRunner(model, out, lock, cache, 'frozen-model-and-protocol'))
    prompts = [Prompt('q1', 'same prompt', 123)]
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda r: r.generate(model, prompts, 1, {'max_tokens': 100}), runners))
    assert len(calls) == 1 and results[0][0][0].text == results[1][0][0].text
    edits = [tmp_path / 'edit1', tmp_path / 'edit2']
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda pair: pair[0].generate(pair[1], prompts, 1, {'max_tokens': 100}),
                                zip(runners, edits)))
    assert len(calls) == 3 and maximum == 1
    assert [r[0][0].text for r in results] == [str(p) for p in edits]
