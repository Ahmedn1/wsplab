import numpy as np
from torch import nn

from wsplab.runners.torch import TorchRunner, state_to_checkpoint


def test_activation_snapshot_precedes_inplace_relu(tmp_path):
    def build():
        return nn.Sequential(nn.Identity(), nn.ReLU(inplace=True))

    weights = state_to_checkpoint(build().state_dict(), tmp_path / "model.safetensors")
    runner = TorchRunner(build)
    inputs = np.array([[-2., 3.]], dtype=np.float32)
    captured = runner.capture(weights, inputs.copy(), ["0", "1"])
    np.testing.assert_array_equal(captured["0"], inputs)
    np.testing.assert_array_equal(captured["1"], [[0., 3.]])
    np.testing.assert_array_equal(runner.call(weights)(inputs.copy()), [[0., 3.]])
    assert all(not module._forward_hooks for module in runner.module(weights).modules())


def test_activation_snapshot_precedes_inplace_residual_addition(tmp_path):
    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.before = nn.Identity()
            self.after = nn.Identity()

        def forward(self, inputs):
            hidden = self.before(inputs.clone())
            hidden.add_(inputs)
            return self.after(hidden)

    weights = state_to_checkpoint(Residual().state_dict(), tmp_path / "model.safetensors")
    inputs = np.array([[-2., 3.]], dtype=np.float32)
    captured = TorchRunner(Residual).capture(weights, inputs, ["before", "after"])
    np.testing.assert_array_equal(captured["before"], inputs)
    np.testing.assert_array_equal(captured["after"], inputs * 2)


def test_forward_tool_captures_activations_in_scorer_batches(tmp_path):
    import json

    from wsplab.eval.suites import ArrayScorer
    from wsplab.passes.torch_forward import TorchForward

    def build():
        return nn.Sequential(nn.Linear(3, 4), nn.ReLU())

    weights = state_to_checkpoint(build().state_dict(), tmp_path / "model.safetensors")
    runner = TorchRunner(build)
    calls = []
    capture = runner.capture
    runner.capture = lambda w, x, names: calls.append(len(x)) or capture(w, x, names)
    inputs = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
    scorer = ArrayScorer(inputs, np.zeros(10, dtype=np.int64), batch=4)
    tool = TorchForward(runner, scorer, work=tmp_path / "work", base=weights, edited=lambda: weights,
                        edit_digest=lambda: None)
    tool.run(json.dumps({"model": "original", "activations": ["1"]}), 1)
    assert calls == [4, 4, 2]
    saved = np.load(tmp_path / "work/fwd_1/act_1.npy")
    np.testing.assert_allclose(saved, capture(weights, inputs, ["1"])["1"], rtol=1e-6)
