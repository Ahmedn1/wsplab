"""Intervention gates tested against real BF16 patch materialization, without inference."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scripts.live import finqa_intervention_trial as trial
from scripts.live.finqa_intervention_guard import InterventionEditor

from wsplab import patch as patches
from wsplab.editors.base import ScriptedEditor
from wsplab.io import Checkpoint, bf16_bits, save
from wsplab.passes.base import PassInvalid, PassResult
from wsplab.session.driver import Policy, Session
from wsplab.session.messages import V5


@pytest.fixture
def experiment(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    base_path = tmp_path / 'base.safetensors'
    save(base_path, {'w': ('BF16', bf16_bits(np.ones((4, 4), np.float32)))})
    base = Checkpoint(base_path)

    class Forward:
        def __init__(self):
            self.work = work
            self.value = None
            self.calls = 0
            self.gain = -.1

        def parse(self, body):
            return json.loads(body)

        def set(self, value):
            self.value = np.ones((4, 4), np.float32)
            self.value[0, 0] = value

        def edits_used(self):
            return (['w'], hashlib.sha256(self.value.tobytes()).hexdigest()) if self.value is not None else ([], None)

        def edited(self):
            folder = tmp_path / self.edits_used()[1]
            patches.from_submission(base, {'w': self.value}, folder / 'patch')
            return folder / 'model.safetensors'

        def run(self, body, number):
            self.calls += 1
            return PassResult('generate', '/work/result', {'ok': True, 'paired': {'accuracy_difference': self.gain}})

    f = Forward()
    editor = InterventionEditor(ScriptedEditor(['stop', 'stop']), dest=tmp_path,
                                dev_ids=['dev'], holdout_ids=['holdout'], required=2, diagnostics=1)
    session = SimpleNamespace(kernel=f, tools={'forward': f}, state=SimpleNamespace(budget=6, passes=0, runs=0),
                              policy=Policy(), messages=V5, notice=lambda text: None)
    editor.attach(session)
    plan = [{'id': c, 'tensors': ['w'], 'operation': 'single entry scaling', 'strength': v, 'rationale': 'test'}
            for c, v in [('a', .5), ('b', .75)]]
    (work / 'interventions.json').write_text(json.dumps(plan))

    def run(candidate='a', holdout=False):
        (work / 'active_candidate.txt').write_text(candidate)
        result = editor.run(json.dumps({'type': 'generate', 'model': 'both',
                                        'problems': ['holdout' if holdout else 'dev'], 'n': 1}), session.state.passes + 1)
        session.state.passes += 1
        return result

    return editor, f, session, run


def test_no_edit_finish_is_not_accepted(experiment, tmp_path):
    guard, forward, _, _ = experiment
    session = Session('test', guard, forward, {'forward': forward}, budget=6,
                      folder=tmp_path / 'session', policy=Policy(max_nudges=1))
    result = session.run()
    assert result.state.nudges == 2 and forward.calls == 0
    with pytest.raises(RuntimeError, match='not completed'):
        guard.validate_finished()


def test_rounding_and_duplicate_checkpoints_do_not_count(experiment):
    guard, f, _, run = experiment
    f.set(1.00001)
    with pytest.raises(PassInvalid, match='zero model entries'):
        run()
    assert f.calls == 0
    f.set(.5)
    run()
    f.set(.50001)  # Different float file, identical BF16 checkpoint.
    with pytest.raises(PassInvalid, match='already tested'):
        run('b')
    assert len(guard.tested) == f.calls == 1


def test_two_real_regressions_allow_restoring_original(experiment):
    guard, f, _, run = experiment
    f.set(.5); run()
    f.set(.75); run('b')
    assert not guard.complete()
    f.value = None
    guard.validate_finished()


def test_positive_dev_winner_requires_one_frozen_holdout(experiment):
    guard, f, _, run = experiment
    f.set(.5); f.gain = .1; run()
    f.set(.75); f.gain = -.1; run('b')
    with pytest.raises(PassInvalid, match='frozen positive-gain'):
        run(holdout=True)
    f.value = None
    assert not guard.complete()
    f.set(.5); f.gain = .05; run(holdout=True)
    guard.validate_finished()
    with pytest.raises(PassInvalid, match='no further inference'):
        run(holdout=True)
    f.set(.25)
    assert not guard.complete()


def test_failed_holdout_requires_original(experiment):
    guard, f, _, run = experiment
    f.set(.5); f.gain = .1; run()
    f.set(.75); f.gain = -.1; run('b')
    f.set(.5); run(holdout=True)
    assert not guard.complete()
    f.value = None
    guard.validate_finished()


def test_plan_cannot_be_changed_after_first_test(experiment):
    _guard, f, _, run = experiment
    f.set(.5); run()
    path = f.work / 'interventions.json'
    plan = json.loads(path.read_text()); plan[1]['strength'] = .1
    path.write_text(json.dumps(plan))
    f.set(.75)
    with pytest.raises(PassInvalid, match='plan is frozen'):
        run('b')


def test_diagnostics_cannot_replace_interventions(experiment):
    guard, f, _, _ = experiment
    body = json.dumps({'type': 'generate', 'model': 'original', 'problems': ['dev']})
    guard.run(body, 1)
    with pytest.raises(PassInvalid, match='Diagnostic allowance'):
        guard.run(body, 2)
    assert not guard.complete() and f.calls == 1


def test_spending_history_is_deducted_and_immutable(tmp_path):
    path = tmp_path / 'old.json'
    path.write_text(json.dumps({'spent': {'openai': .5, 'anthropic': 1., 'gemini': 3.},
                                'reserved': {'openai': 0., 'anthropic': 0., 'gemini': 0.}}))
    (tmp_path / 'budget-history.json').write_text(json.dumps(
        {'ledgers': [{'path': str(path), 'sha256': trial.digest(path), 'arm': 'peers'}]}))
    caps, history = trial.spending_caps(tmp_path)
    assert sum(caps['peers'].values()) == caps['solo']['anthropic'] == 7.5
    assert history['previous_total'] + 2 * 7.5 < 24
    path.write_text('{}')
    with pytest.raises(ValueError, match='ledger changed'):
        trial.spending_caps(tmp_path)
