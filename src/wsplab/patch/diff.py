"""Exact record of an edit: the edited tensors minus the base model, bit for bit.

Derived from exp8_math/code/make_edit_diff.py (MIT). Changes: float16/32/64 tensors; 1-D and N-D tensors; a dense form
for edits of true rank above the limit; 64-bit indices for tensors past 2^31 entries; rank search that stays inside the
spectrum; submissions that round back to the base are recorded instead of crashing; one tensor in memory at a time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ..io.safetensors import DT, LONG_NAMES, Checkpoint, cast_bits, cast_to, to_bits, to_float
from .format import FORMAT, FORMAT_VERSION, Patch, PatchError, sha

SPARSE_LIMIT = 0.3
MAX_RANK = 8
RANK_TOL = 1e-6


def lowrank_recon(dtype: str, base_bits: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """base + the rank-one terms, summed one at a time in float64: a single a @ b could round differently per BLAS."""
    w = to_float(dtype, base_bits.view(DT[dtype])).astype(np.float64)
    for k in range(a.shape[1]):
        w = w + np.outer(a[:, k], b[k])
    return cast_bits(w.astype(np.float32) if dtype != "F64" else w, dtype)


def _flat_index(idx: np.ndarray, size: int) -> np.ndarray:
    return idx.astype(np.int32 if size < 2**31 else np.int64)


def _try_lowrank(dtype: str, base_bits: np.ndarray, edited_bits: np.ndarray, target: np.ndarray):
    """(factors a, b, singular values) of target - base if a low-rank record is smaller than a dense one, else None."""
    if base_bits.ndim != 2 or min(base_bits.shape) == 0:
        return None
    d = target.astype(np.float64) - to_float(dtype, base_bits.view(DT[dtype])).astype(np.float64)
    u, s, vt = np.linalg.svd(d, full_matrices=False)
    if s[0] == 0:
        return None
    limit = min(MAX_RANK, len(s))
    r = limit
    for k in range(1, limit):
        if s[k] <= RANK_TOL * s[0]:
            r = k
            break
    a, b = u[:, :r] * s[:r], vt[:r]
    recon = lowrank_recon(dtype, base_bits, a, b)
    fix = np.flatnonzero(recon != edited_bits)
    factor_bytes = 8 * r * sum(base_bits.shape)
    fix_bytes = fix.size * (4 + base_bits.itemsize)
    if factor_bytes + fix_bytes >= base_bits.size * base_bits.itemsize // 2:
        return None
    return a, b, s, r, fix


def diff_tensor(i: int, name: str, dtype: str, base_bits: np.ndarray, edited_bits: np.ndarray,
                submitted: np.ndarray | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    """The manifest record and arrays of tensor number i. Bits are compared, so NaNs and signed zeros stay exact."""
    if base_bits.shape != edited_bits.shape:
        raise PatchError(f"{name}: shape {edited_bits.shape} differs from the base {base_bits.shape}")
    mask = base_bits != edited_bits
    base_f = to_float(dtype, base_bits.view(DT[dtype])).astype(np.float64)
    edit_f = to_float(dtype, edited_bits.view(DT[dtype])).astype(np.float64)
    norm = float(np.linalg.norm(base_f))
    rec = {"name": name, "shape": list(base_bits.shape), "dtype": LONG_NAMES[dtype],
           "changed_entries": int(mask.sum()),
           "relative_change": float(np.linalg.norm(edit_f - base_f) / norm) if norm else float("inf"),
           "base_sha256": sha(base_bits), "edited_sha256": sha(edited_bits)}
    if mask.ndim == 2:
        rec["changed_columns"] = int(mask.any(0).sum())
        rec["changed_rows"] = int(mask.any(1).sum())
    del base_f, edit_f
    arrays: dict[str, np.ndarray] = {}
    if mask.mean() <= SPARSE_LIMIT:
        idx = np.flatnonzero(mask)
        arrays[f"t{i}_idx"] = _flat_index(idx, mask.size)
        arrays[f"t{i}_val"] = edited_bits.reshape(-1)[idx]
        rec["form"] = "sparse"
        return rec, arrays
    target = submitted if submitted is not None else to_float(dtype, edited_bits.view(DT[dtype]))
    low = _try_lowrank(dtype, base_bits, edited_bits, target)
    if low is None:
        arrays[f"t{i}_bits"] = edited_bits
        rec["form"] = "dense"
        return rec, arrays
    a, b, s, r, fix = low
    arrays[f"t{i}_a"], arrays[f"t{i}_b"] = a, b
    arrays[f"t{i}_idx"] = _flat_index(fix, mask.size)
    arrays[f"t{i}_val"] = edited_bits.reshape(-1)[fix]
    rec.update(form="lowrank", rank=int(r), singular_values_kept=[float(x) for x in s[:r]],
               next_singular_value=float(s[r]) if r < len(s) else 0.0, fixups=int(fix.size))
    return rec, arrays


def write_patch(out: str | os.PathLike, tensors: list[dict], arrays: dict[str, np.ndarray], *, no_effect=(),
                meta: dict | None = None) -> Patch:
    """Write a patch folder and return it, loaded and validated."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "edit.npz", **arrays)
    manifest = {"format_version": FORMAT_VERSION, "tensors": tensors, "changed_tensors": len(tensors),
                "changed_entries_total": sum(t["changed_entries"] for t in tensors),
                "no_effect": sorted(no_effect),
                "note": "changed_entries counts entries whose bits differ from the base model; an entry the edit "
                        "touched but that rounds back to the same value is not counted"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    write_meta(out, meta or {})
    return Patch.load(out)


def write_meta(path: Path, meta: dict) -> None:
    """Write patch.json with the digests of the folder's current contents."""
    p = Patch(path, json.loads((path / "manifest.json").read_text()))
    body = {"format": FORMAT, **meta, "digests": p.digests()}
    (path / "patch.json").write_text(json.dumps(body, indent=1) + "\n")


def from_checkpoints(base: Checkpoint, edited: Checkpoint, out: str | os.PathLike, *, meta: dict | None = None) -> Patch:
    """The patch that turns `base` into `edited`. Both must have the same tensor names, shapes and dtypes."""
    if set(base.names()) != set(edited.names()):
        raise PatchError(f"different tensors: {sorted(set(base.names()) ^ set(edited.names()))[:5]}")
    tensors, arrays = [], {}
    for name in sorted(base.names()):
        dt, b = base.raw(name)
        dt2, e = edited.raw(name)
        if dt != dt2:
            raise PatchError(f"{name}: dtype changed from {dt} to {dt2}")
        bb, eb = to_bits(b), to_bits(e)
        if np.array_equal(bb, eb):
            continue
        rec, arr = diff_tensor(len(tensors), name, dt, bb, eb)
        tensors.append(rec)
        arrays.update(arr)
    return write_patch(out, tensors, arrays, meta=meta)


def from_submission(base: Checkpoint, submission: dict[str, np.ndarray], out: str | os.PathLike, *,
                    meta: dict | None = None) -> Patch:
    """The patch for edited float tensors, rounded to each tensor's dtype in the base model, as the evaluation does."""
    tensors, arrays, no_effect = [], {}, []
    for name in sorted(submission):
        if name not in base:
            raise PatchError(f"{name}: not a tensor of the base model")
        dt, b = base.raw(name)
        sub = np.asarray(submission[name])
        if tuple(sub.shape) != tuple(b.shape):
            raise PatchError(f"{name}: shape {sub.shape} differs from the base {b.shape}")
        bb, eb = to_bits(b), cast_to(sub, dt, name)
        if np.array_equal(bb, eb):
            no_effect.append(name)
            continue
        rec, arr = diff_tensor(len(tensors), name, dt, bb, eb, sub)
        tensors.append(rec)
        arrays.update(arr)
    return write_patch(out, tensors, arrays, no_effect=no_effect, meta=meta)


def read_submission(folder: str | os.PathLike) -> dict[str, np.ndarray]:
    """A folder of <tensor name>.npy files, such as the /work/edited folder of a session."""
    return {f.name[:-4]: np.load(f, allow_pickle=False) for f in sorted(Path(folder).glob("*.npy"))}


def import_release(source: str | os.PathLike, out: str | os.PathLike, *, name=None, base_repo=None, base_revision=None,
                   license=None, attribution=None) -> dict:
    """Copy a released edit folder (edit.npz + manifest.json) unchanged and add patch.json."""
    import shutil

    source, out = Path(source), Path(out)
    Patch.load(source)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("edit.npz", "manifest.json"):
        shutil.copyfile(source / f, out / f)
    meta = {"name": name or source.parent.name, "base": {"repo": base_repo, "revision": base_revision},
            "license": license, "attribution": attribution, "evidence": {"source": "imported release"}}
    write_meta(out, meta)
    return meta
