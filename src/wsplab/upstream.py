"""Pinned access to the paper's repository, RikiyaT/lm_understand-logs.

Files are fetched on demand into a local cache and checked against provenance/upstream.lock.json when a hash is known.
Nothing fetched here is executed: it is read as data.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

REPO = "RikiyaT/lm_understand-logs"
COMMIT = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/"


def home() -> Path:
    """The wsplab state folder: $WSPLAB_HOME, else ~/.wsplab."""
    return Path(os.environ.get("WSPLAB_HOME", Path.home() / ".wsplab"))


def cache_dir() -> Path:
    return home() / "upstream" / COMMIT


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(path: str, expected_sha256: str | None = None) -> Path:
    """Local path of `path` (relative to the repo root) at the pinned commit, downloading it if needed."""
    if path.startswith("/") or ".." in Path(path).parts:
        raise ValueError(f"not a repository path: {path}")
    out = cache_dir() / path
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")
        with urllib.request.urlopen(RAW + path, timeout=120) as r, open(tmp, "wb") as f:
            while block := r.read(1 << 20):
                f.write(block)
        tmp.replace(out)
    if expected_sha256 is not None:
        got = sha256_file(out)
        if got != expected_sha256:
            raise ValueError(f"{path}: sha256 {got} does not match the pinned {expected_sha256}")
    return out
