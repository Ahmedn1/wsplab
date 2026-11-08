"""Run the routed benchmark with supplied editors; CLI runs a free scripted smoke test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import wsplab
from wsplab.editors import ScriptedEditor, Spending
from wsplab.eval.suites import ArrayScorer
from wsplab.kernel import Limits
from wsplab.session.driver import Policy

if __package__:
    from . import modular_net as task
else:
    import modular_net as task


DESCRIPTION = (
    "Ten independent binary digit detectors. Input columns 0:64 are an 8x8 grayscale image, "
    "row-major in [0,1]; columns 64:74 are a one-hot query for digit 0-9. "
    "Each branch is Linear(64,16), ReLU, Linear(16,8), ReLU, Linear(8,2). "
    "The query selects one branch's logit difference (class 1 minus class 0). "
    "Output logits are [0, selected_difference], "
    "so label 1 means the image matches the queried digit and label 0 means it does not. "
    "Every image is queried only once. Practice includes balanced positive/negative queries per digit."
)


def run_benchmark(*, editors, spending, folder, shared, seed=21, max_runs=40, budget=6, extra_note=""):
    """Same fixture, selection set, and per-member limits for shared/unshared arms.

    One editor means one solo attempt; repeated independent editors give a
    best-of-N solo baseline using the identical candidate-selection procedure.
    The caller supplies editors and spending; no credentials or prices are guessed.
    """
    folder = Path(folder)
    if not editors:
        raise ValueError("supply at least one editor")
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("use a new output directory for each benchmark run")
    torch.set_num_threads(1)
    splits = task.image_splits(seed)
    net = task.damage(task.trained(seed))
    practice = ArrayScorer(*task.queries(*splits["practice"], seed), name="routed digit queries")
    evaluation = ArrayScorer(*task.queries(*splits["evaluation"], seed), name="held-out routed queries", prefix="eval")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "experiment.json").write_text(json.dumps({
        "benchmark": "routed-digits-v1", "seed": seed, "shared": shared,
        "editors": [editor.name for editor in editors], "max_runs_per_editor": max_runs,
        "passes_per_editor": budget, "selection": "metric", "max_usd": spending.max_usd,
        "max_tokens": spending.max_tokens, "prices": spending.prices,
    }, indent=2) + "\n")
    try:
        return wsplab.search(net, editors=editors, spending=spending, shared=shared,
                            build=task.build, description=DESCRIPTION,
                            note="This network used to be accurate, but it has been damaged in several places." + extra_note,
                            practice=practice, evaluate=evaluation, folder=folder, seed=seed,
                            max_runs=max_runs, budget=budget, limits=Limits(seconds=120, memory_gb=4),
                            policy=Policy(handoff_every=10**9, require_all_passes=False, max_turns=60))
    finally:
        (folder / "spending.json").write_text(json.dumps({"usd": spending.usd, "tokens": spending.tokens}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # Known injected fault, deliberately given to scripted editors ONLY to test
    # the plumbing. This is not evidence that an LLM can discover a repair.
    code = ("The smoke fixture has a known output-swap fault in branch zero. I swap its output rows back.\n"
            "```python\nfor suffix in ['weight', 'bias']:\n    name = 'branches.0.l3.' + suffix\n"
            "    save_edit(name, load_original(name).flip(0))\n```")
    editors = [ScriptedEditor([code, "Done."], name=f"smoke-{i}") for i in range(2)]
    result = run_benchmark(editors=editors, spending=Spending(max_tokens=1000), folder=args.out,
                           shared=True, budget=0, max_runs=2)
    if not all(candidate["changed_entries"] > 0 for candidate in result.candidates):
        raise RuntimeError("smoke failure: a scripted repair was not applied")
    if result.result.final is None or result.result.final <= result.result.baseline:
        raise RuntimeError("smoke failure: known repair did not improve held-out accuracy")
    print(json.dumps({"smoke_only": True, "verdict": result.result.verdict,
                      "baseline": result.result.baseline, "final": result.result.final,
                      "candidates": len(result.candidates)}, indent=2))


if __name__ == "__main__":
    main()
