"""Milestone 9 acceptance: the authors' final logged pass of run p50c, repeated through wsplab with their released edit.

    python scripts/conformance_p50c.py <request number> <out folder>

Sampling cannot match token for token (their notice 002 says so even for their own server), so the check is statistical.
"""

import json
import os
import sys
from pathlib import Path

from wsplab import patch as P
from wsplab import upstream
from wsplab.eval import MathGrader
from wsplab.eval.graders import INSTRUCTION
from wsplab.eval.lm_suite import read_jsonl
from wsplab.io import Checkpoint
from wsplab.passes import ForwardLimits, LMForward
from wsplab.runners import LM

REV = "2fc06364715b967f1860aea9cf38778875588b17"
QWEN = Path(os.environ["HF_HOME"]) / f"hub/models--Qwen--Qwen3.5-0.8B/snapshots/{REV}"
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768}


def main(k: str, out: str) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    run = "exp8_math/runs/p50c"
    request = upstream.fetch(f"{run}/requests/fwd_{k}.json.claimed").read_text()
    logged = json.loads(upstream.fetch(f"{run}/requests/fwd_{k}.done.json").read_text())
    upstream.fetch(f"{run}/edit/manifest.json")
    patch = P.Patch.load(upstream.fetch(f"{run}/edit/edit.npz").parent)
    edited = P.apply(patch, Checkpoint(QWEN), out / "model")
    rows = {}
    for name in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl"):
        rows.update({r["id"]: r for r in read_jsonl(upstream.fetch(f"exp8_math/practice/{name}"))})
    runner = LM(QWEN, thinking=True, gpu_memory=0.85)
    tool = LMForward(runner, practice=rows, grader=MathGrader(),
                     limits=ForwardLimits(n_layers=24, full_attention=(3, 7, 11, 15, 19, 23)), sampling=SAMPLING, work=out,
                     base=QWEN, edited=lambda: edited, edit_digest=lambda: "p50c",
                     user_message=lambda row: f"{row['problem']}\n{INSTRUCTION}")
    result = tool.run(request, int(k))
    report = {"logged": {m: logged.get(m) for m in ("original", "edited", "paired")},
              "wsplab": {m: result.summary.get(m) for m in ("original", "edited", "paired")}, "seconds": result.summary.get("s")}
    (out / "comparison.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"logged": logged.get("paired"), "wsplab": result.summary.get("paired"), "s": result.summary.get("s")}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
