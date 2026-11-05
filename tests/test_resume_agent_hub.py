"""Crash recovery, a coding agent as the editor, and sharing patches."""

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

import wsplab
from wsplab import patch as P
from wsplab.cli import main
from wsplab.editors import ScriptedEditor
from wsplab.eval.suites import ArrayScorer
from wsplab.io import Checkpoint, save
from wsplab.kernel import Limits
from wsplab.kernel.client import KernelError, sandbox_python
from wsplab.patch.hub import TAG, model_card, pull, push
from wsplab.runners.torch import mlp, state_to_checkpoint

LIMITS = Limits(seconds=60, memory_gb=4)


@pytest.fixture
def sandbox():
    try:
        sandbox_python()
    except KernelError:
        pytest.skip("no sandbox environment")


def data(n=200):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, 2)).astype(np.float32)
    return x, (x[:, 0] > 0).astype(np.int64)


def broken():
    net = mlp([2, 2])
    with torch.no_grad():
        net.layers[0].weight.copy_(torch.tensor([[1.0, 0.25], [-1.0, 0.25]]))
        net.layers[0].bias.zero_()
    return net


GOOD = "w = load('layers.0.weight'); w[0, 0] = -1.0; w[1, 0] = 1.0; save_edit('layers.0.weight', w)"


class Crashing(ScriptedEditor):
    def reply(self, system, messages):
        if not self.replies:
            raise KeyboardInterrupt
        return super().reply(system, messages)


def test_a_crashed_run_resumes_where_it_stopped(tmp_path, sandbox):
    x, y = data()
    scorer = ArrayScorer(x, y)
    calls = {"n": 0}
    original_score = scorer.score

    def counted(call, ids):
        calls["n"] += 1
        return original_score(call, ids)

    scorer.score = counted
    kw = dict(build={"mlp": [2, 2]}, description="a classifier", practice=ArrayScorer(*data(50)), evaluate=None, budget=0,
              folder=tmp_path / "r", limits=LIMITS, control=False)
    held = ArrayScorer(x + 100, y, prefix="eval")
    count = {"n": 0}
    real = held.score

    def counted_held(call, ids):
        count["n"] += 1
        return real(call, ids)

    held.score = counted_held
    kw["evaluate"] = held
    first = ["Printing a marker.\n```python\nopen('/work/marker', 'w').write('x')\nsecret = 1\n```"]
    with pytest.raises(KeyboardInterrupt):
        wsplab.run(broken(), editor=Crashing(first), **kw)
    assert count["n"] == 1
    state = json.loads((tmp_path / "r" / "session" / "state.json").read_text())
    assert state["turn"] == 1 and state["runs"] == 1

    resume = ScriptedEditor(["Checking.\n```python\nprint('secret' in globals())\n```", f"Fixing.\n```python\n{GOOD}\n```", "Done."])
    r = wsplab.run(broken(), editor=resume, resume=True, **kw)
    assert r.state.runs == 3 and r.state.turn == 4
    assert count["n"] == 2  # the baseline was not measured again, only the result
    message = resume.seen[0][-1]["content"]
    assert message.startswith("[note: the harness was restarted")
    assert (tmp_path / "r" / "work" / "marker").exists()
    assert "False" in (tmp_path / "r" / "session" / "tool_calls.jsonl").read_text().splitlines()[1]


def test_a_crash_after_the_session_does_not_ask_the_editor_again(tmp_path, sandbox, monkeypatch):
    x, y = data()
    held = ArrayScorer(x * 2, y, prefix="eval")
    kw = dict(build={"mlp": [2, 2]}, description="a classifier", practice=ArrayScorer(*data(50)), evaluate=held, budget=0,
              folder=tmp_path / "r", limits=LIMITS, control=False)
    real, calls = held.score, {"n": 0}

    def fails_on_the_final_evaluation(call, ids):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("the GPU ran out of memory")
        return real(call, ids)

    held.score = fails_on_the_final_evaluation
    with pytest.raises(RuntimeError, match="out of memory"):
        wsplab.run(broken(), editor=ScriptedEditor([f"Fixing.\n```python\n{GOOD}\n```", "Fixed the rows."]), **kw)
    idle = ScriptedEditor([])
    r = wsplab.run(broken(), editor=idle, resume=True, **kw)
    assert idle.seen == [] and r.final == 1.0 and r.answer == "Fixed the rows."


def agent_files(tmp_path):
    x, y = data()
    np.savez(tmp_path / "practice.npz", inputs=x[:100], targets=y[:100])
    np.savez(tmp_path / "eval.npz", inputs=x[100:], targets=y[100:])
    state_to_checkpoint(broken().state_dict(), tmp_path / "m.safetensors")


def test_a_coding_agent_can_be_the_editor(tmp_path, sandbox, capsys):
    agent_files(tmp_path)
    folder = tmp_path / "s"
    assert main(["session", "start", "--folder", str(folder), "--model", str(tmp_path / "m.safetensors"), "--build", "mlp:2,2",
                 "--description", "a classifier", "--note", "It answers backwards", "--practice", str(tmp_path / "practice.npz"),
                 "--eval", str(tmp_path / "eval.npz"), "--budget", "1", "--memory-gb", "4"]) == 0
    brief = capsys.readouterr().out
    assert brief.startswith("TASK.") and "It answers backwards" in brief

    assert main(["session", "brief", str(folder)]) == 0
    assert capsys.readouterr().out == brief

    assert main(["session", "exec", str(folder), "--why", "Looking at the tensors.", "--code", "print(NAMES)"]) == 0
    assert "OUTPUT of Python run 1 of at most 300" in capsys.readouterr().out

    assert main(["session", "exec", str(folder), "--why", "The rows are swapped, so I swap them back.", "--code", GOOD]) == 0
    capsys.readouterr()
    assert main(["session", "forward", str(folder), '{"model": "both"}', "--why", "Measuring it."]) == 0
    assert "Forward pass 1 of 1 done" in capsys.readouterr().out

    assert main(["session", "finish", str(folder), "I swapped the two rows."]) == 0
    out = capsys.readouterr().out
    assert "verdict:" in out and "patch:" in out and "final: 1.0" in out
    patch = P.Patch.load(folder / "run" / "result" / "patch")
    assert patch.meta["editor"] == "agent-cli" and patch.meta["tool_free"] is False


def test_agent_gets_the_error_instead_of_hanging(tmp_path, sandbox, capsys):
    agent_files(tmp_path)
    folder = tmp_path / "s"
    code = main(["session", "start", "--folder", str(folder), "--model", str(tmp_path / "missing.safetensors"), "--build", "mlp:2,2",
                 "--description", "c", "--no-eval", "--budget", "0"])
    assert code == 1 or "verdict: error" in capsys.readouterr().out


def make_patch(tmp_path):
    save(tmp_path / "m.safetensors", {"w": ("F32", np.ones((4, 4), np.float32))})
    base = Checkpoint(tmp_path / "m.safetensors")
    w = base.float("w")
    w[0, 0] = 2
    meta = {"name": "demo", "base": {"repo": "org/model", "revision": "abc"}, "license": "MIT",
            "evidence": {"verdict": "confirmed"}, "editor": "someone"}
    return base, P.from_submission(base, {"w": w}, tmp_path / "patch", meta=meta)


def test_model_card_says_what_a_hash_does_not_prove(tmp_path):
    _, patch = make_patch(tmp_path)
    card = model_card(patch)
    assert f"- {TAG}" in card and "base_model: org/model" in card and "**confirmed**" in card
    assert "do not prove" in card and "license: mit" in card


class FakeHub:
    """Stores uploads in a folder, like the Hub's two calls we use."""

    def __init__(self, root):
        self.root = root

    def create_repo(self, repo, private, exist_ok):
        (self.root / repo).mkdir(parents=True, exist_ok=True)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id):
        target = self.root / repo_id / path_in_repo
        if isinstance(path_or_fileobj, bytes):
            target.write_bytes(path_or_fileobj)
        else:
            shutil.copyfile(path_or_fileobj, target)

    def download(self, repo_id, filename, revision, local_dir):
        target = local_dir / filename
        shutil.copyfile(self.root / repo_id / filename, target)
        return str(target)


def test_hub_round_trip(tmp_path):
    base, _ = make_patch(tmp_path)
    hub = FakeHub(tmp_path / "hub")
    url = push(tmp_path / "patch", "me/demo", api=hub)
    assert url == "https://huggingface.co/me/demo"
    assert (tmp_path / "hub/me/demo/README.md").exists()
    patch = pull("me/demo", tmp_path / "got", download=hub.download)
    assert P.verify(patch, base)[0]["ok"]


def test_a_tampered_patch_on_the_hub_is_refused(tmp_path):
    make_patch(tmp_path)
    hub = FakeHub(tmp_path / "hub")
    push(tmp_path / "patch", "me/demo", api=hub)
    with open(tmp_path / "hub/me/demo/edit.npz", "ab") as f:
        f.write(b"tampered")
    with pytest.raises(P.PatchError, match="was changed"):
        pull("me/demo", tmp_path / "got", download=hub.download)


def test_imports_stay_light():
    """Patch tools must work with numpy alone."""
    code = ("import sys, wsplab.patch, wsplab.io\n"
            "bad = [m for m in ('torch', 'transformers', 'vllm', 'openai', 'anthropic') if m in sys.modules]\n"
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)

