"""Rebuild edited tensors from a base model and a patch, bit for bit, and check every hash.

Derived from exp8_math/code/apply_edit.py (MIT). Changes: any float dtype; sharded checkpoints; an edited tensor is
written over its bytes in a copy of its shard, so nothing else changes and the model never has to fit in memory;
untouched files are hard-linked; manifests are checked against the base header first.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from ..io.safetensors import Checkpoint, to_bits
from .diff import lowrank_recon
from .format import Patch, PatchError, bits_dtype, dtype_of, sha


def rebuild(base_bits: np.ndarray, record: dict, z, i: int) -> np.ndarray:
    """The edited tensor's raw bits. Indices and values are bounds-checked: a patch is untrusted input."""
    dtype = dtype_of(record)
    if record["form"] == "dense":
        out = np.asarray(z[f"t{i}_bits"])
        if out.dtype != base_bits.dtype or out.shape != base_bits.shape:
            raise PatchError(f"{record['name']}: dense values do not fit the tensor")
        return out
    if record["form"] == "lowrank":
        a, b = np.asarray(z[f"t{i}_a"]), np.asarray(z[f"t{i}_b"])
        if a.ndim != 2 or b.ndim != 2 or a.shape != (base_bits.shape[0], b.shape[0]) or b.shape[1] != base_bits.shape[1]:
            raise PatchError(f"{record['name']}: low-rank factors do not fit the tensor")
        e = lowrank_recon(dtype, base_bits, a, b).reshape(-1)
    else:
        e = base_bits.copy().reshape(-1)
    idx, val = np.asarray(z[f"t{i}_idx"]), np.asarray(z[f"t{i}_val"])
    if idx.ndim != 1 or idx.shape != val.shape or not np.issubdtype(idx.dtype, np.integer):
        raise PatchError(f"{record['name']}: bad sparse entries")
    if val.dtype != bits_dtype(dtype):
        raise PatchError(f"{record['name']}: values are {val.dtype}, expected {bits_dtype(dtype)}")
    if idx.size and (idx.min() < 0 or idx.max() >= e.size):
        raise PatchError(f"{record['name']}: index outside the tensor")
    e[idx] = val
    return e.reshape(base_bits.shape)


def base_bits_of(base: Checkpoint, record: dict) -> np.ndarray:
    """The base tensor's bits, after checking that the patch was made on exactly this tensor."""
    name = record["name"]
    if name not in base:
        raise PatchError(f"{name}: not a tensor of the base model")
    dtype, arr = base.raw(name)
    if dtype != dtype_of(record) or list(arr.shape) != record["shape"]:
        raise PatchError(f"{name}: the base tensor is {dtype} {list(arr.shape)}, the patch expects "
                         f"{dtype_of(record)} {record['shape']}")
    bits = to_bits(arr)
    if sha(bits) != record["base_sha256"]:
        raise PatchError(f"{name}: the base tensor differs from the one the patch was made on")
    return bits


def edited_tensors(patch: Patch, base: Checkpoint):
    """Yield (record, edited bits) one tensor at a time, with every hash checked."""
    for i, record in enumerate(patch.tensors):
        bits = rebuild(base_bits_of(base, record), record, patch.z, i)
        if sha(bits) != record["edited_sha256"]:
            raise PatchError(f"{record['name']}: rebuilt tensor does not match edited_sha256")
        yield record, bits


def verify(patch: Patch, base: Checkpoint) -> list[dict]:
    """Rebuild every tensor without writing anything. Raises PatchError on any mismatch."""
    return [{"name": r["name"], "form": r["form"], "changed_entries": r["changed_entries"], "ok": True}
            for r, _ in edited_tensors(patch, base)]


def _link_or_copy(src: Path, dst: Path, *, link: bool) -> None:
    src = src.resolve()
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copyfile(src, dst)


def apply(patch: Patch, base: Checkpoint, out: str | os.PathLike, *, link: bool = True) -> Path:
    """Write the edited model. `out` is a folder, or a .safetensors path when the base is a single file.

    Files the patch does not touch are hard-linked when `link` is true: they share their disk space with the base model,
    so do not modify them in place.
    """
    out = Path(out)
    records = list(edited_tensors(patch, base))
    if out.suffix == ".safetensors":
        if len(base.files) != 1:
            raise PatchError("a sharded model must be written to a folder")
        out.parent.mkdir(parents=True, exist_ok=True)
        targets = {next(iter(base.files)): out}
    else:
        out.mkdir(parents=True, exist_ok=True)
        targets = {shard: out / shard for shard in base.files}
        for f in base.dir.iterdir():
            if f.is_file() and f.name not in targets and not (out / f.name).exists() and f.suffix != ".safetensors":
                _link_or_copy(f, out / f.name, link=link)
    touched = {base.where[r["name"]] for r, _ in records}
    for shard, dst in targets.items():
        if dst.exists():
            dst.unlink()
        _link_or_copy(base.files[shard].path, dst, link=link and shard not in touched)
    for record, bits in records:
        shard = base.files[base.where[record["name"]]]
        start = shard.offset + shard.info[record["name"]]["data_offsets"][0]
        with open(targets[base.where[record["name"]]], "r+b") as f:
            f.seek(start)
            f.write(np.ascontiguousarray(bits).tobytes())
    return out
