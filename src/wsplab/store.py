"""The paper's editing surface: load, load_original, save_edit, drop_edit, edited.

The prompts specify these, but their code was not released, so this is a reconstruction. Edits are whole float tensors
written to <edited>/<name>.npy, exactly what a session submits. numpy only: this module also runs inside the sandbox.
"""

from __future__ import annotations

import hashlib
import os
import urllib.parse
from pathlib import Path

import numpy as np

from .io.safetensors import LONG_NAMES, Checkpoint, cast_to, to_bits, to_float

# Inferred: the prompts say such edits are refused, not how close to a constant multiple counts.
RESCALE_TOLERANCE = 1e-6


class EditRefused(ValueError):
    """save_edit refuses this edit."""


class EditsLocked(RuntimeError):
    """This block was not approved to change weights."""


def _as_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value)


def edit_file_name(name: str) -> str:
    return urllib.parse.quote(name, safe="") + ".npy"


def tensor_name_of(file_name: str) -> str:
    return urllib.parse.unquote(file_name[:-4])


class WeightStore:
    """mode: "read" refuses every edit; "edit" and "write" allow them. `names` limits what is visible, for example to the
    text part of a multimodal model. `max_changed` caps how many entries all edits together may change (after rounding
    to the model's dtype); an edit that would go over it is refused."""

    def __init__(self, checkpoint: Checkpoint, edited_dir: str | os.PathLike, *, mode: str = "edit", frozen=(),
                 refuse_rescale: bool = True, names: list[str] | None = None, max_changed: int | None = None):
        if mode not in ("read", "edit", "write"):
            raise ValueError(f"unknown mode {mode!r}")
        self.checkpoint = checkpoint
        self.dir = Path(edited_dir)
        self.mode = mode
        self.frozen = set(frozen)
        self.refuse_rescale = refuse_rescale
        self.names = list(names) if names is not None else checkpoint.names()
        self.locked = False
        self.max_changed = max_changed
        self._counts: dict[str, tuple[int, int]] = {}
        unknown = set(self.names) - set(checkpoint.names())
        if unknown:
            raise ValueError(f"unknown tensors: {sorted(unknown)[:5]}")

    def _path(self, name: str) -> Path:
        return self.dir / edit_file_name(name)

    def _check_name(self, name: str) -> None:
        if name not in self.names:
            raise KeyError(f"unknown tensor {name!r}; see NAMES")

    def load_original(self, name: str) -> np.ndarray:
        self._check_name(name)
        return self.checkpoint.float(name)

    def load(self, name: str) -> np.ndarray:
        """The current edit if there is one, else the original."""
        self._check_name(name)
        p = self._path(name)
        return np.load(p, allow_pickle=False) if p.exists() else self.checkpoint.float(name)

    def save_edit(self, name: str, value) -> str:
        """Store an edited tensor. Returns what actually changes once it is rounded to the model's dtype."""
        self._check_name(name)
        if self.mode == "read":
            raise EditRefused("this session only reads the weights: save_edit is not available")
        if self.locked:
            raise EditsLocked("this block was not approved to change weights")
        if name in self.frozen:
            raise EditRefused(f"{name} is out of scope and cannot be edited")
        dtype, base = self.checkpoint.raw(name)
        array = _as_array(value)
        if tuple(array.shape) != tuple(base.shape):
            raise EditRefused(f"{name}: shape {tuple(array.shape)} differs from {tuple(base.shape)}")
        bits = cast_to(array, dtype, name)
        original = to_float(dtype, base).astype(np.float64)
        if self.refuse_rescale and self._is_rescale(original, array.astype(np.float64)):
            raise EditRefused(f"{name}: an edit that only multiplies a whole tensor by a constant is refused")
        changed = int((bits != to_bits(base)).sum())
        if self.max_changed is not None:
            total = changed + sum(self.changed_entries(n) for n in self.edited() if n != name)
            if total > self.max_changed:
                raise EditRefused(f"{name}: the edits would change {total:,} entries in total; the limit is "
                                  f"{self.max_changed:,} (nothing was saved)")
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f".{os.getpid()}.tmp.npy"
        np.save(tmp, array.astype(np.float64 if dtype == "F64" else np.float32))
        tmp.replace(self._path(name))
        return f"saved {name}: {changed:,} of {array.size:,} entries differ from the original in {LONG_NAMES[dtype]}"

    @staticmethod
    def _is_rescale(original: np.ndarray, new: np.ndarray) -> bool:
        denominator = float((original * original).sum())
        if denominator == 0:
            return False
        c = float((original * new).sum()) / denominator
        if abs(c - 1) <= RESCALE_TOLERANCE or c == 0:
            return False
        return float(np.linalg.norm(new - c * original)) <= RESCALE_TOLERANCE * float(np.linalg.norm(new))

    def changed_entries(self, name: str) -> int:
        """Entries of one saved edit that differ from the original after rounding to the model's dtype."""
        p = self._path(name)
        if not p.exists():
            return 0
        stamp = p.stat().st_mtime_ns
        if self._counts.get(name, (None,))[0] != stamp:
            dtype, base = self.checkpoint.raw(name)
            bits = cast_to(np.load(p, allow_pickle=False), dtype, name)
            self._counts[name] = (stamp, int((bits != to_bits(base)).sum()))
        return self._counts[name][1]

    def drop_edit(self, name: str) -> None:
        self._check_name(name)
        if self.locked:
            raise EditsLocked("this block was not approved to change weights")
        self._path(name).unlink(missing_ok=True)

    def edited(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(tensor_name_of(f.name) for f in self.dir.glob("*.npy") if not f.name.startswith("."))

    def digest(self) -> str | None:
        """A short hash of the current edits: it tells a session which edit a measurement belongs to."""
        names = self.edited()
        if not names:
            return None
        h = hashlib.sha256()
        for name in names:
            h.update(name.encode())
            with open(self._path(name), "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
        return h.hexdigest()[:10]
