"""The whole job with a scripted editor and no LLM: baseline, session in the real sandbox, edit, final eval, patch, infer."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import wsplab
from wsplab import patch as P
from wsplab.editors import ScriptedEditor
from wsplab.eval.suites import ArrayScorer, EpisodeScorer
from wsplab.io import Checkpoint
from wsplab.kernel import Limits
from wsplab.kernel.client import KernelError, sandbox_python
from wsplab.runners.torch import TorchRunner, mlp
from wsplab.session.driver import Policy

LIMITS = Limits(seconds=60, memory_gb=4)


@pytest.fixture(autouse=True)
def sandbox_needed():
    try:
        sandbox_python()
    except KernelError:
        pytest.skip("no sandbox environment")


def data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2)).astype(np.float32)
    return x, (x[:, 0] > 0).astype(np.int64)


def working_mlp():
    """Unit 0 reads x0 and class 1 reads unit 0: right by construction, with no training."""
    net = mlp([2, 4, 2], "relu")
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        w0, w1 = net.layers[0].weight, net.layers[2].weight
        w0[0, 0] = 1.0
        w0[1, 0] = -1.0
        w1[1, 0] = 1.0
        w1[0, 1] = 1.0
        w1[:, 2:] = 0.5  # units 2 and 3 never fire: this keeps the fix from being a whole-tensor rescale
    return net


def sabotaged_mlp():
    net = working_mlp()
    with torch.no_grad():
        net.layers[2].weight[:, :2] *= -1
    return net


def block(kind, body, why="Testing a hypothesis about the output layer."):
    return f"{why}\n```{kind}\n{body}\n```"


FIX = """w = load('layers.2.weight')
w[:, :2] = -w[:, :2]
save_edit('layers.2.weight', w)"""
SCRIPT = [block("python", "print(NAMES)"), block("forward", '{"model": "original"}'), block("python", FIX),
          block("forward", '{"model": "both"}'), "I flipped the sign of the two output columns."]


def run_mlp(tmp_path, replies, **kw):
    x, y = data()
    kw.setdefault("budget", 2)
    return wsplab.run(sabotaged_mlp(), build={"mlp": [2, 4, 2], "activation": "relu"}, description="a tiny classifier",
                      note="This model cannot tell left from right", evaluate=ArrayScorer(x, y),
                      editor=ScriptedEditor(replies), folder=tmp_path / "run", limits=LIMITS, **kw)


def test_full_lifecycle_fixes_a_sabotaged_mlp(tmp_path):
    r = run_mlp(tmp_path, list(SCRIPT))
    assert r.baseline == 0.0 and r.final == 1.0
    assert r.verdict == "confirmed"
    assert r.state.passes == 2 and r.state.done
    report = r.report
    assert report["comparison"]["edited"] == 1.0 and report["control"] is not None
    assert (r.folder / "report.md").read_text().startswith("# wsplab run: confirmed")
    x, y = data(seed=9)
    assert (r.infer(x).argmax(1) == y).all()


def test_optional_pass_policy_is_explained_in_the_actual_session_prompt(tmp_path):
    result = run_mlp(tmp_path, ["Keep the original weights."], policy=Policy(require_all_passes=False))
    prompt = (result.folder / "session/turns/t001/prompt.txt").read_text()
    assert "2 forward passes are a maximum, not a requirement" in prompt
    assert "ends the session immediately" in prompt
    assert "It continues until all 2 forward passes are used" not in prompt
    assert result.state.stop_reason == "finished" and result.state.passes == 0


def test_required_pass_policy_keeps_the_original_prompt(tmp_path):
    result = run_mlp(tmp_path, list(SCRIPT))
    prompt = (result.folder / "session/turns/t001/prompt.txt").read_text()
    assert "It continues until all 2 forward passes are used" in prompt
    assert "forward passes are a maximum, not a requirement" not in prompt


def test_patch_rebuilds_the_evaluated_model_and_carries_its_evidence(tmp_path):
    r = run_mlp(tmp_path, list(SCRIPT))
    patch = P.Patch.load(r.patch)
    assert patch.meta["evidence"]["verdict"] == "confirmed"
    assert patch.meta["harness"] == "reconstructed" and patch.meta["prompt_fidelity"] == "paper"
    assert patch.meta["editor"] == "scripted" and patch.meta["transcript_sha256"]
    base = Checkpoint(tmp_path / "run" / "base" / "model.safetensors")
    out = Checkpoint(P.apply(patch, base, tmp_path / "again.safetensors"))
    assert np.array_equal(out.float("layers.2.weight"), Checkpoint(r.weights).float("layers.2.weight"))


def test_the_note_reaches_the_editor_and_eval_data_does_not(tmp_path):
    editor_script = list(SCRIPT)
    r = run_mlp(tmp_path, editor_script)
    prompt = (r.folder / "session" / "turns" / "t001" / "prompt.txt").read_text()
    assert prompt.rstrip().endswith("NOTE FROM THE USER. This model cannot tell left from right")
    assert "EXPLAIN EVERY STEP." in prompt and "RULES." in prompt
    inputs = np.load(r.folder / "practice" / "inputs.npy")
    assert len(inputs) == 200 and sorted(p.name for p in (r.folder / "practice").iterdir()) == ["inputs.npy", "targets.npy"]


def test_whole_tensor_negation_is_refused_so_nothing_is_fixed(tmp_path):
    negate = "w = load('layers.0.weight'); print(save_edit('layers.0.weight', w * -1))"
    r = run_mlp(tmp_path, [block("python", negate), "I negated it."], budget=0)
    assert r.verdict == "no_fix_found"
    assert "constant" in (r.folder / "session" / "tool_calls.jsonl").read_text()


def test_no_edit_means_no_fix_found(tmp_path):
    r = run_mlp(tmp_path, [block("forward", '{"model": "original"}'), block("forward", '{"model": "original", "items": ["item_0"]}'),
                           "I changed nothing."])
    assert r.verdict == "no_fix_found" and r.final is None


def test_harmful_edit_regresses(tmp_path):
    x, y = data()
    good = working_mlp()
    r = wsplab.run(good, build={"mlp": [2, 4, 2]}, description="a classifier", evaluate=ArrayScorer(x, y), budget=0,
                   editor=ScriptedEditor([block("python", "w = load('layers.2.weight'); w[:, :2] = -w[:, :2]; "
                                                          "save_edit('layers.2.weight', w)"), "Done."]),
                   folder=tmp_path / "run", limits=LIMITS)
    assert r.verdict == "regressed" and r.final == 0.0 and r.baseline == 1.0


def test_useless_edit_is_not_significant(tmp_path):
    x, y = data()
    r = wsplab.run(sabotaged_mlp(), build={"mlp": [2, 4, 2]}, description="a classifier", evaluate=ArrayScorer(x, y), budget=0,
                   editor=ScriptedEditor([block("python", "b = load('layers.2.bias'); b[0] += 0.0001; "
                                                          "save_edit('layers.2.bias', b)"), "Done."]),
                   folder=tmp_path / "run", limits=LIMITS)
    assert r.verdict == "not_significant"


def test_regression_suite_can_veto(tmp_path):
    x, y = data()
    other = ArrayScorer(x, 1 - y, name="other skill")
    r = run_mlp(tmp_path, list(SCRIPT), regressions={"other skill": other})
    assert r.verdict == "regressed" and any("other skill" in reason for reason in r.reasons)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1, bias=False)
        self.head = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.head(torch.relu(self.conv(x)).mean((2, 3)))


def build_cnn():
    return TinyCNN()


def cnn_state(broken: bool):
    net = TinyCNN()
    with torch.no_grad():
        net.conv.weight.zero_()
        net.conv.weight[0, 0, 1, 1] = 1.0
        net.conv.weight[1, 0, 1, 1] = -1.0
        net.head.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        if broken:
            net.head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    return net


def test_full_lifecycle_fixes_a_sabotaged_cnn(tmp_path):
    rng = np.random.default_rng(3)
    sign = rng.choice([-1.0, 1.0], 300).astype(np.float32)
    x = sign[:, None, None, None] * np.abs(rng.standard_normal((300, 1, 6, 6))).astype(np.float32)
    y = (sign > 0).astype(np.int64)
    fix = "h = load('head.weight'); save_edit('head.weight', h.flip(0))"
    r = wsplab.run(cnn_state(True), build=build_cnn, description="a small CNN", evaluate=ArrayScorer(x, y), budget=1,
                   editor=ScriptedEditor([block("python", fix), block("forward", '{"model": "both", "activations": ["conv"]}'),
                                          "Swapped the output rows."]),
                   folder=tmp_path / "run", limits=LIMITS)
    assert r.verdict == "confirmed" and r.final == 1.0
    acts = list((r.folder / "work").glob("fwd_1_edited/act_conv.npy"))
    assert acts and np.load(acts[0]).shape[1:] == (2, 6, 6)


def test_model_factory_from_a_file(tmp_path):
    source = tmp_path / "net.py"
    source.write_text("from torch import nn\ndef build():\n    return nn.Linear(2, 2)\n")
    runner = TorchRunner(f"{source}:build")
    assert isinstance(runner.build(), nn.Linear)


class Corridor:
    """Reach position 3 by moving right. Gymnasium's interface, without the dependency."""

    def reset(self, seed=None):
        self.pos = 0
        return np.array([1.0, 0.0], np.float32), {}

    def step(self, action):
        self.pos += 1 if action == 1 else -1
        done = self.pos >= 3 or self.pos <= -3
        return np.array([1.0, 0.0], np.float32), float(self.pos >= 3), done, False, {}


def test_a_game_policy_goes_through_the_same_path(tmp_path):
    net = mlp([2, 2], "relu")
    with torch.no_grad():
        net.layers[0].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
        net.layers[0].bias.zero_()
    fix = "w = load('layers.0.weight'); w[0, 0] = 0.0; w[1, 0] = 1.0; save_edit('layers.0.weight', w)"
    r = wsplab.run(net, build={"mlp": [2, 2]}, description="a policy for a corridor", budget=1,
                   practice=EpisodeScorer(Corridor, range(4)), evaluate=EpisodeScorer(Corridor, range(100, 112)),
                   editor=ScriptedEditor([block("python", fix), block("forward", '{"model": "edited"}'), "Fixed."]),
                   folder=tmp_path / "run", limits=LIMITS, control=False)
    assert r.baseline == 0.0 and r.final == 1.0 and r.verdict == "confirmed"


def test_read_mode_gives_an_answer_and_no_patch(tmp_path):
    r = wsplab.run(working_mlp(), description="a classifier", mode="read", budget=0, folder=tmp_path / "run", limits=LIMITS,
                   editor=ScriptedEditor([block("python", "load('layers.0.weight')[0, 0].item()"), "It reads x0."]))
    assert r.answer == "It reads x0." and r.patch is None and r.verdict == "unmeasured"
    prompt = (r.folder / "session" / "turns" / "t001" / "prompt.txt").read_text()
    assert "save_edit" not in prompt and "forward-pass tool with a budget" not in prompt


def test_write_mode_starts_from_zero(tmp_path):
    x, y = data()
    code = """w0 = load('layers.0.weight'); print(float(w0.abs().sum()))
w0[0, 0] = 1.0; w0[1, 0] = -1.0; save_edit('layers.0.weight', w0)
w1 = load('layers.2.weight'); w1[1, 0] = 1.0; w1[0, 1] = 1.0; save_edit('layers.2.weight', w1)"""
    r = wsplab.run(working_mlp(), build={"mlp": [2, 4, 2]}, description="a two-class classifier", mode="write",
                   evaluate=ArrayScorer(x, y), budget=1, control=False, folder=tmp_path / "run", limits=LIMITS,
                   editor=ScriptedEditor([block("python", code), block("forward", '{"model": "edited"}'), "Wrote it."]))
    assert r.final == 1.0 and r.verdict == "confirmed"
    assert "0.0" in json.loads((r.folder / "session" / "tool_calls.jsonl").read_text().splitlines()[0])["out"]


def test_inline_edit_needs_no_kernel(tmp_path):
    x, y = data()
    reply = "The output columns are flipped.\n```edits\nlayers.2.weight[:, 0:2] *= -1\n```"
    r = wsplab.run(sabotaged_mlp(), build={"mlp": [2, 4, 2]}, description="a classifier", evaluate=ArrayScorer(x, y),
                   access="inline", budget=0, editor=ScriptedEditor([reply]), folder=tmp_path / "run", limits=LIMITS)
    assert r.final == 1.0 and r.verdict == "confirmed"
    prompt = (r.folder / "session" / "turns" / "t001" / "prompt.txt").read_text()
    assert "layers.2.weight, shape 2 x 4:" in prompt


def test_inline_refuses_networks_too_large_for_context(tmp_path):
    with pytest.raises(ValueError, match="too large"):
        wsplab.run(mlp([800, 800]), description="big", access="inline", mode="read", budget=0, folder=tmp_path / "r",
                   editor=ScriptedEditor(["x"]), limits=LIMITS)


def test_overlapping_practice_and_eval_are_refused(tmp_path):
    x, y = data()
    with pytest.raises(ValueError, match="both the practice data and the evaluation"):
        wsplab.run(sabotaged_mlp(), build={"mlp": [2, 4, 2]}, description="c", practice=ArrayScorer(x, y),
                   evaluate=ArrayScorer(x, y), editor=ScriptedEditor([]), folder=tmp_path / "r", limits=LIMITS)


def test_edit_mode_needs_an_eval_unless_waived(tmp_path):
    with pytest.raises(ValueError, match="no_eval"):
        wsplab.run(working_mlp(), description="c", budget=0, editor=ScriptedEditor([]), folder=tmp_path / "r")
    r = wsplab.run(working_mlp(), description="c", budget=0, no_eval=True, folder=tmp_path / "r", limits=LIMITS,
                   editor=ScriptedEditor([block("python", "w = load('layers.0.bias'); w[0] = 1; save_edit('layers.0.bias', w)"),
                                          "Done."]))
    assert r.verdict == "unmeasured" and Path(r.patch).exists()


def test_full_custom_prompt(tmp_path):
    x, y = data()
    r = wsplab.run(sabotaged_mlp(), build={"mlp": [2, 4, 2]}, description="a classifier", evaluate=ArrayScorer(x, y), budget=0,
                   prompt="Fix {{network}} any way you like.", editor=ScriptedEditor([block("python", FIX), "Fixed."]),
                   folder=tmp_path / "run", limits=LIMITS)
    assert (r.folder / "session/turns/t001/prompt.txt").read_text() == "Fix a classifier any way you like."
    assert P.Patch.load(r.patch).meta["prompt_fidelity"] == "custom"
