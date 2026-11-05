"""Combine patches. Patches touching different tensors always combine; two patches on one tensor must be chained: the
later one made on the result of the earlier one, which the per-tensor hashes prove."""

from __future__ import annotations

import os

import numpy as np

from ..io.safetensors import Checkpoint
from .apply import base_bits_of, rebuild
from .diff import diff_tensor, write_patch
from .format import Patch, PatchError, dtype_of, sha


def stack(patches: list[Patch], base: Checkpoint, out: str | os.PathLike, *, meta: dict | None = None) -> Patch:
    """One patch equal to applying `patches` in order to `base`."""
    chain: dict[str, list[tuple[Patch, int]]] = {}
    for patch in patches:
        for i, record in enumerate(patch.tensors):
            chain.setdefault(record["name"], []).append((patch, i))
    tensors, arrays = [], {}
    for name in sorted(chain):
        first_patch, first_i = chain[name][0]
        first = first_patch.tensors[first_i]
        original = base_bits_of(base, first)
        bits = original
        for patch, i in chain[name]:
            record = patch.tensors[i]
            if sha(bits) != record["base_sha256"]:
                raise PatchError(f"{name}: patches conflict: they edit this tensor but are not chained")
            bits = rebuild(bits, record, patch.z, i)
            if sha(bits) != record["edited_sha256"]:
                raise PatchError(f"{name}: rebuilt tensor does not match edited_sha256")
        if np.array_equal(original, bits):
            continue
        rec, arr = diff_tensor(len(tensors), name, dtype_of(first), original, bits)
        tensors.append(rec)
        arrays.update(arr)
    return write_patch(out, tensors, arrays, meta=meta)
