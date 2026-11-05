"""Run any PyTorch network from a checkpoint. The weights always come from the checkpoint file, so the base and an
edited model run through exactly the same code."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Callable

import numpy as np

from ..io.safetensors import LONG_NAMES, Checkpoint, save, to_float

ACTIVATIONS = ("relu", "tanh", "gelu", "sigmoid")


def mlp(sizes: list[int], activation: str = "relu"):
    """The networks of the paper's small experiments: Linear layers named layers.<i>."""
    from torch import nn

    if activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {ACTIVATIONS}")
    act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "sigmoid": nn.Sigmoid}[activation]
    layers: list = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(act())
    module = nn.Sequential()
    module.add_module("layers", nn.Sequential(*layers))
    return module


def load_factory(spec: str) -> Callable:
    """"path/to/model.py:build": your own code, which you trust, returning an nn.Module."""
    path, _, name = spec.rpartition(":")
    if not path or not name:
        raise ValueError(f"expected path/to/model.py:function, got {spec!r}")
    loader = importlib.util.spec_from_file_location("wsplab_user_model", path)
    if loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return getattr(module, name)


def resolve_build(build) -> Callable:
    if callable(build):
        return build
    if isinstance(build, dict):
        return lambda: mlp(build["mlp"], build.get("activation", "relu"))
    if isinstance(build, str):
        return load_factory(build)
    raise ValueError("`build` must be a function, path/to/model.py:function, or {'mlp': [sizes], 'activation': ...}")


def state_to_checkpoint(state: dict, path: str | os.PathLike) -> Path:
    """Write a PyTorch state dict as a safetensors file. bfloat16 stays bit-exact."""
    import torch

    names = {torch.float32: "F32", torch.float64: "F64", torch.float16: "F16", torch.bfloat16: "BF16",
             torch.int64: "I64", torch.int32: "I32", torch.uint8: "U8", torch.bool: "BOOL"}
    tensors = {}
    for name, value in state.items():
        value = value.detach().cpu().contiguous()
        if value.dtype not in names:
            raise ValueError(f"{name}: unsupported dtype {value.dtype}")
        dtype = names[value.dtype]
        array = value.view(torch.uint16).numpy() if dtype == "BF16" else value.numpy()
        tensors[name] = (dtype, array)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save(path, tensors)
    return path


def zero_checkpoint(base: Checkpoint, path: str | os.PathLike) -> Path:
    """A copy of a model with every float tensor at zero: the start of a from-scratch session."""
    tensors = {}
    for name in base.names():
        dtype, arr = base.raw(name)
        tensors[name] = (dtype, np.zeros(arr.shape, arr.dtype) if dtype in LONG_NAMES else np.array(arr))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save(path, tensors)
    return path


def checkpoint_to_state(checkpoint: Checkpoint) -> dict:
    import torch

    state = {}
    for name in checkpoint.names():
        dtype, arr = checkpoint.raw(name)
        if dtype in ("BF16", "F16", "F32", "F64"):
            state[name] = torch.from_numpy(to_float(dtype, arr).copy())
        else:
            state[name] = torch.from_numpy(np.array(arr))
    return state


class TorchRunner:
    """`build` makes the module; weights are loaded from a checkpoint. Runs on `device` with gradients off."""

    def __init__(self, build, device: str = "cpu"):
        self.build = resolve_build(build)
        self.device = device
        self._cache: dict[str, object] = {}

    def module(self, weights: str | os.PathLike):
        path = Path(weights)
        key = f"{path}:{self._stamp(path)}"
        if key not in self._cache:
            self._cache.clear()
            module = self.build()
            state = checkpoint_to_state(Checkpoint(path))
            module.load_state_dict(state, strict=True)
            module.to(self.device).eval()
            for p in module.parameters():
                p.requires_grad_(False)
            self._cache[key] = module
        return self._cache[key]

    @staticmethod
    def _stamp(path: Path) -> int:
        files = [path] if path.is_file() else sorted(path.glob("*.safetensors"))
        return hash(tuple((f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in files))

    def module_names(self, weights) -> list[str]:
        return [n for n, m in self.module(weights).named_modules() if n and not list(m.children())]

    def call(self, weights) -> Callable:
        """A function from a numpy batch to a numpy batch."""
        import torch

        module = self.module(weights)

        def run(inputs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                out = module(torch.as_tensor(np.asarray(inputs), device=self.device))
            return out.detach().cpu().numpy()

        return run

    def capture(self, weights, inputs: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
        """The outputs of named modules on a batch."""
        import torch

        module = self.module(weights)
        modules = dict(module.named_modules())
        missing = [n for n in names if n not in modules]
        if missing:
            raise ValueError(f"unknown modules: {missing}")
        got: dict[str, np.ndarray] = {}
        handles = [modules[n].register_forward_hook(
            lambda _m, _i, o, n=n: got.__setitem__(n, o.detach().cpu().numpy())) for n in names]
        try:
            with torch.no_grad():
                module(torch.as_tensor(np.asarray(inputs), device=self.device))
        finally:
            for h in handles:
                h.remove()
        return got
