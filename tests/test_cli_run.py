"""The command line, end to end, from weight files to a patched inference."""

import json

import numpy as np
import pytest
import torch

from wsplab import doctor
from wsplab.cli import main, parse_build
from wsplab.kernel.client import KernelError, sandbox_python
from wsplab.runners.torch import mlp, state_to_checkpoint


def test_parse_build():
    assert parse_build("mlp:2,4,2:tanh") == {"mlp": [2, 4, 2], "activation": "tanh"}
    assert parse_build("net.py:build") == "net.py:build"


@pytest.fixture
def files(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 2)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    np.savez(tmp_path / "data.npz", inputs=x[:150], targets=y[:150])
    np.savez(tmp_path / "held.npz", inputs=x[150:], targets=y[150:])
    np.save(tmp_path / "x.npy", x[:5])
    net = mlp([2, 2])
    with torch.no_grad():
        net.layers[0].weight.copy_(torch.tensor([[1.0, 0.25], [-1.0, 0.25]]))
        net.layers[0].bias.zero_()
    state_to_checkpoint(net.state_dict(), tmp_path / "m.safetensors")
    return tmp_path


def test_run_infer_and_eval_from_the_command_line(files, capsys):
    try:
        sandbox_python()
    except KernelError:
        pytest.skip("no sandbox environment")
    fix = "w = load('layers.0.weight'); w[0, 0] = -1.0; w[1, 0] = 1.0; save_edit('layers.0.weight', w)"
    script = [f"Swapping the rows.\n```python\n{fix}\n```", "Explaining.\n```forward\n{\"model\": \"both\"}\n```", "Fixed."]
    (files / "script.json").write_text(json.dumps(script))
    common = ["--model", str(files / "m.safetensors"), "--build", "mlp:2,2"]
    assert main(["eval", *common, "--data", str(files / "held.npz")]) == 0
    assert "accuracy: 0.000000" in capsys.readouterr().out
    assert main(["run", *common, "--description", "a classifier", "--note", "It answers backwards", "--practice",
                 str(files / "data.npz"), "--eval", str(files / "held.npz"), "--editor", f"script:{files / 'script.json'}",
                 "--budget", "1", "--out", str(files / "run"), "--memory-gb", "4", "--no-control"]) == 0
    out = capsys.readouterr().out
    assert "verdict: confirmed" in out
    patch = str(files / "run" / "result" / "patch")
    assert main(["eval", *common, "--patch", patch, "--data", str(files / "held.npz")]) == 0
    assert "accuracy: 1.000000" in capsys.readouterr().out
    assert main(["infer", *common, "--patch", patch, "--input", str(files / "x.npy"), "--out", str(files / "y.npy")]) == 0
    x = np.load(files / "x.npy")
    assert (np.load(files / "y.npy").argmax(1) == (x[:, 0] > 0)).all()


def test_doctor_reports_each_requirement(capsys):
    assert main(["doctor"]) in (0, 1)
    out = capsys.readouterr().out
    for name in ("python", "bubblewrap", "sandbox environment", "vllm", "GPU"):
        assert name in out


def test_setup_commands_use_a_cpu_only_environment_without_model_runtimes(tmp_path):
    commands = doctor.setup_commands(tmp_path / "env")
    flat = " ".join(" ".join(c) for c in commands)
    assert "whl/cpu" in flat and "transformers" not in flat and "vllm" not in flat


def test_user_errors_are_clean(tmp_path, capsys):
    assert main(["patch", "inspect", str(tmp_path / "missing")]) == 1
    assert "error:" in capsys.readouterr().err
