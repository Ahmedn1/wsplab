"""Write provenance/upstream.lock.json: the sha256 of every upstream file wsplab derives from, at the pinned commit."""

import json
from pathlib import Path

from wsplab import upstream

FILES = [
    "LICENSE",
    "README.md",
    "harness/instructions.md",
    "harness/sandbox.py",
    "exp7_sycophancy/runs/attention_tool/sandbox/prelude.py",
    "exp8_math/README.md",
    "exp8_math/code/apply_edit.py",
    "exp8_math/code/build_ckpt.py",
    "exp8_math/code/evaluate.py",
    "exp8_math/code/make_edit_diff.py",
    "exp8_math/code/mathqa.py",
    "exp8_math/code/merge.py",
    "exp8_math/code/profiles.py",
    "exp8_math/code/st_numpy.py",
    "exp8_math/notices/001_pass_size.txt",
    "exp8_math/notices/002_tool_facts.txt",
    "exp8_math/notices/003_submit_best.txt",
    "exp8_math/notices/004_failure_counts.txt",
    "exp8_math/runs/p50c/turns/t001/prompt.txt",
    "exp8_math/runs/p50c/message.txt",
    "exp8_math/gemma_followup/runs/g50b/turns/t001/prompt.txt",
]

root = Path(__file__).resolve().parent.parent
lock = {"repo": upstream.REPO, "commit": upstream.COMMIT, "files": {}}
for f in FILES:
    lock["files"][f] = upstream.sha256_file(upstream.fetch(f))
    print(lock["files"][f][:12], f)
(root / "provenance" / "upstream.lock.json").write_text(json.dumps(lock, indent=1) + "\n")
