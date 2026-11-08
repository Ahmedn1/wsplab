"""Offline end-to-end check of the Spurious ResNet trial with scripted editors: sandbox, GPU forward passes with
activations, patch, selection and the final evaluation. No paid calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wsplab
from wsplab.editors import Spending
from wsplab.editors.base import ScriptedEditor
from wsplab.kernel import Limits
from wsplab.session.driver import Policy

if __package__:
    from . import spurious_resnet as task
    from .spurious_trial import NOTE, describe
else:
    import spurious_resnet as task
    from spurious_trial import NOTE, describe


def fence(kind, body):
    return f"Plumbing check: this step tests the harness, not a hypothesis.\n\n```{kind}\n{body}\n```"


INSPECT = """import numpy as np
x = np.load('/practice/inputs.npy', mmap_mode='r'); y = np.load('/practice/targets.npy')
print(x.shape, x.dtype, int((y < 0).sum()), 'spurious', int((y >= 0).sum()), 'clean')
w = load('fc.weight'); print(tuple(w.shape), float(w.abs().mean()))"""

EDIT = """import numpy as np
w = load('fc.weight').clone()
k = {k}
top = w[k].argsort(descending=True)[:16]
w[k, top] = w[k, top] * 0.5
save_edit('fc.weight', w)
print(edited())"""


def run(fixture, out):
    model, practice, evaluation, split = task.load_fixture(fixture)
    classes = json.loads((Path(fixture) / "classes.json").read_text())
    editors = []
    for i, k in enumerate(split["target_classes"][:3]):
        editors.append(ScriptedEditor([
            fence("python", INSPECT),
            fence("python", EDIT.format(k=k)),
            fence("forward", json.dumps({"model": "both", "activations": ["avgpool"]})),
            "Final: halved the 16 largest fc weights of one target class as a plumbing check.",
        ], name=f"scripted-{i}"))
    result = wsplab.search(
        model, editors=editors, spending=Spending(max_tokens=10**9), shared=True, build=task.build,
        description=describe(split, classes), note=NOTE, practice=practice, evaluate=evaluation,
        folder=Path(out) / "run", seed=task.SEED, max_runs=10, budget=2, device="cuda",
        limits=Limits(seconds=300, memory_gb=8), policy=Policy(handoff_every=10**9, require_all_passes=False, max_turns=10))
    summary = {"verdict": result.result.verdict, "baseline": result.result.baseline, "final": result.result.final,
               "candidates": [{k: c[k] for k in ("editor", "changed_entries", "score", "gain", "stop_reason")}
                              for c in result.candidates]}
    (Path(out) / "dryrun.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.fixture, args.out), indent=2))
