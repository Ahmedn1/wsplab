"""The patch format: a folder holding edit.npz, manifest.json and patch.json.

edit.npz and manifest.json are the format of the paper's releases (exp8_math/runs/<run>/edit/), and those files can be
read as they are ("version 1"). Version 2 adds other tensor dtypes, a dense form and 64-bit indices. patch.json is
wsplab's addition: what the patch applies to, the evidence for it, and the digests that tie the folder together.

Per tensor i of manifest["tensors"], edit.npz holds:
  sparse   t<i>_idx (flat indices), t<i>_val (new values as raw bits)
  lowrank  t<i>_a (rows x r), t<i>_b (r x cols), float64, plus fixups t<i>_idx, t<i>_val, where
           rounding base + a[:, 0] b[0] + ... + a[:, r-1] b[r-1] differs from the edited tensor
  dense    t<i>_bits (the whole edited tensor as raw bits)
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..io.safetensors import DT, FLOATS, SHORT_NAMES
from ..upstream import sha256_file

FORMAT = "wsplab-patch"
FORMAT_VERSION = 2
FORMS = ("sparse", "lowrank", "dense")
_NPZ_SLACK = 1 << 24


class PatchError(ValueError):
    """A patch that is malformed, tampered with, or does not fit the base model."""


def sha(a: np.ndarray) -> str:
    """sha256 of the array's bytes, without copying them."""
    return hashlib.sha256(np.ascontiguousarray(a).reshape(-1).view(np.uint8).data).hexdigest()


@dataclass
class Patch:
    """A patch folder, validated on load. Nothing in it is executed, and arrays are read with pickling disabled."""

    path: Path
    manifest: dict
    meta: dict = field(default_factory=dict)
    _z: np.lib.npyio.NpzFile | None = None

    @classmethod
    def load(cls, path: str | os.PathLike) -> Patch:
        path = Path(path)
        if not (path / "manifest.json").is_file() or not (path / "edit.npz").is_file():
            raise PatchError(f"{path}: not a patch folder (needs manifest.json and edit.npz)")
        manifest = json.loads((path / "manifest.json").read_text())
        meta = json.loads((path / "patch.json").read_text()) if (path / "patch.json").exists() else {}
        patch = cls(path, manifest, meta)
        patch._check_digests()
        patch._check_manifest()
        patch._check_archive()
        return patch

    @property
    def tensors(self) -> list[dict]:
        return self.manifest["tensors"]

    @property
    def version(self) -> int:
        return int(self.manifest.get("format_version", 1))

    @property
    def z(self) -> np.lib.npyio.NpzFile:
        if self._z is None:
            self._z = np.load(self.path / "edit.npz", allow_pickle=False)
        return self._z

    def digests(self) -> dict:
        return {"manifest_sha256": sha256_file(self.path / "manifest.json"),
                "edit_sha256": sha256_file(self.path / "edit.npz")}

    def _check_digests(self) -> None:
        want = self.meta.get("digests")
        if want is None:
            return
        got = self.digests()
        for key, value in got.items():
            if want.get(key) != value:
                raise PatchError(f"{self.path}: {key} does not match patch.json: the patch was changed")

    def _check_manifest(self) -> None:
        tensors = self.manifest.get("tensors")
        if not isinstance(tensors, list):
            raise PatchError("manifest: no tensor list")
        seen = set()
        for t in tensors:
            name = t.get("name")
            if not isinstance(name, str) or not name or name in seen:
                raise PatchError(f"manifest: bad or repeated tensor name {name!r}")
            parts = name.replace("\\", "/").split("/")
            if name.startswith(("/", ".")) or "\0" in name or ".." in parts or "." in parts or "" in parts:
                raise PatchError(f"manifest: unsafe tensor name {name!r}")
            seen.add(name)
            if t.get("form") not in FORMS:
                raise PatchError(f"{name}: unknown form {t.get('form')!r}")
            if SHORT_NAMES.get(t.get("dtype")) not in FLOATS:
                raise PatchError(f"{name}: unsupported dtype {t.get('dtype')!r}")
            shape = t.get("shape")
            if not isinstance(shape, list) or not all(isinstance(n, int) and n >= 0 for n in shape):
                raise PatchError(f"{name}: bad shape {shape!r}")
            for key in ("base_sha256", "edited_sha256"):
                if not isinstance(t.get(key), str) or len(t[key]) != 64:
                    raise PatchError(f"{name}: missing {key}")

    def _check_archive(self) -> None:
        """Bound the archive by what the manifest can need before anything is decompressed."""
        limit = _NPZ_SLACK
        for t in self.tensors:
            limit += 24 * int(np.prod(t["shape"], dtype=np.int64)) + 8 * 64 * sum(t["shape"])
        try:
            with zipfile.ZipFile(self.path / "edit.npz") as archive:
                total = sum(i.file_size for i in archive.infolist())
        except zipfile.BadZipFile as e:
            raise PatchError(f"edit.npz: {e}") from e
        if total > limit:
            raise PatchError(f"edit.npz: {total:,} bytes uncompressed is more than its tensors can need")


def dtype_of(record: dict) -> str:
    return SHORT_NAMES[record["dtype"]]


def bits_dtype(dtype: str) -> np.dtype:
    return np.dtype(f"u{np.dtype(DT[dtype]).itemsize}")
