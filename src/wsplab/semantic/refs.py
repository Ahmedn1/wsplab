"""Named parts of a network. A Ref is an address: a tensor and a slice of it, with verbs that lower to whole-tensor
save_edit, so audits and patches work the same at every level.

Refs never compute statistics. Inspecting weights is the editor's job, done with its own code: this layer only says
where things are.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class Ref:
    """`index` selects this part of `tensor`. `output_axis` is the axis along which the part writes into the next layer,
    which is what project_out acts on."""

    tensor: str
    index: tuple
    output_axis: int
    load: Callable
    save: Callable

    def read(self):
        return self.load(self.tensor)[self.index]

    def write(self, value):
        """Replace this part's values."""
        full = np.array(_as_numpy(self.load(self.tensor)))
        full[self.index] = _as_numpy(value)
        return self.save(self.tensor, full)

    def scale(self, factor: float):
        full = np.array(_as_numpy(self.load(self.tensor)))
        full[self.index] = full[self.index] * factor
        return self.save(self.tensor, full)

    def zero(self):
        """Writes +0.0: scaling by zero would give -0.0 for negative weights, which is a different bit pattern."""
        return self.write(0.0)

    def add(self, delta):
        full = np.array(_as_numpy(self.load(self.tensor)))
        full[self.index] = full[self.index] + _as_numpy(delta)
        return self.save(self.tensor, full)

    def project_out(self, direction, strength: float = 1.0):
        """W' = W - strength * u (u^T W) on this part, with u the unit direction along the output axis. Strength 1 removes
        the direction from what this part writes, and 2 reflects it. The paper's edits have this form."""
        u = _as_numpy(direction).astype(np.float64)
        u = u / np.linalg.norm(u)
        full = np.array(_as_numpy(self.load(self.tensor)), dtype=np.float64)
        part = np.moveaxis(full[self.index], self.output_axis, 0)
        if part.shape[0] != u.size:
            raise ValueError(f"{self.tensor}: direction has {u.size} entries, the output axis has {part.shape[0]}")
        flat = part.reshape(u.size, -1)
        flat = flat - strength * np.outer(u, u @ flat)
        full[self.index] = np.moveaxis(flat.reshape(part.shape), 0, self.output_axis)
        return self.save(self.tensor, full.astype(np.float32))


class Parts:
    """An indexable collection: parts[3], with a clear error past the end."""

    def __init__(self, count: int, make: Callable, label: str):
        self.count = count
        self.make = make
        self.label = label

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, i: int):
        if not 0 <= i < self.count:
            raise IndexError(f"{self.label} {i} is out of range: there are {self.count}")
        return self.make(i)
