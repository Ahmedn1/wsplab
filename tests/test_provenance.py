"""Milestone 0: upstream hashes match the pin, and the wheel carries nothing it must not."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from wsplab import upstream

ROOT = Path(__file__).resolve().parent.parent
LOCK = json.loads((ROOT / "provenance" / "upstream.lock.json").read_text())


def test_lock_is_for_the_pinned_commit():
    assert LOCK["commit"] == upstream.COMMIT and LOCK["repo"] == upstream.REPO


@pytest.mark.network
@pytest.mark.parametrize("path", sorted(LOCK["files"]))
def test_upstream_file_matches_lock(path):
    upstream.fetch(path, LOCK["files"][path])


def test_fetch_refuses_paths_outside_the_repo():
    with pytest.raises(ValueError):
        upstream.fetch("../../etc/passwd")


def test_notice_credits_the_paper():
    notice = (ROOT / "NOTICE").read_text()
    for needle in ("Takehi", "CC BY 4.0", upstream.COMMIT):
        assert needle in notice


def test_wheel_has_no_gpl_or_benchmark_data(tmp_path):
    subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path), str(ROOT)], check=True,
                   capture_output=True)
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        assert any(n.endswith("wsplab/__init__.py") for n in names)
        assert "wsplab/prompts/templates/lm_edit.txt" in names and "wsplab/prompts/parts/tools_generic.txt" in names
        assert not any("experiments/" in n or "tests/" in n for n in names)
        for n in names:
            assert not n.endswith((".jsonl", ".npz", ".safetensors", ".parquet")), n
            data = z.read(n)
            for banned in (b"Kenny Young", b"GNU GENERAL PUBLIC LICENSE", b"minatar/environments"):
                assert banned not in data, (n, banned)
