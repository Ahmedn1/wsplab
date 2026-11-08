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
