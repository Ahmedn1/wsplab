"""Paid, bounded three-provider peer/independent trial on the frozen Spurious ResNet fixture.

Arms: "peers" (OpenAI, Claude and Gemini share every step) and "independent" (the same three, alone). The independent arm
gives each solo result and the no-communication portfolio (its best candidate on the selection set)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import wsplab
from wsplab.editors import Spending
from wsplab.kernel import Limits
from wsplab.runners import torch as torch_runner
from wsplab.session.driver import Policy

if __package__:
    from . import spurious_resnet as task
    from .peer_trial import MODELS, PRICES, Guarded, Ledger, editor
else:
    import spurious_resnet as task
    from peer_trial import MODELS, PRICES, Guarded, Ledger, editor

PASSES, MAX_RUNS, MAX_TURNS = 8, 40, 60

NOTE = (
    "This is an unmodified public ImageNet ResNet-50. It calls many photos of a class's typical background or context "
    "(the 'spurious cue') that class, even when the object is absent: for example water or coral as 'snorkel', ice as "
    "'puck'. Edit the weights so that spurious photos of the target classes are no longer called the absent class, while "
    "clean photos of those classes and of every other class keep their accuracy. All three score groups matter equally; "
    "a fix that destroys a class is not a fix. "
    "Do NOT train or optimize any loss, and do not fit weights by regression or least squares. Do NOT compute "
    "predictions, logits, losses or accuracy for modified weights inside Python, including replaying the fc layer (or "
    "any layer) on recorded activations: that is a forward pass and must use the official forward tool. Weight "
    "statistics, correlations and analysis of already-recorded activations and outputs are allowed. Do not load "
    "public data or checkpoints. Save a plausible candidate early; restore the original if evidence does not support it. "
    "Keep diagnostic output compact; a paid-token ceiling applies."
)


def describe(split, classes):
    targets = ", ".join(f"{k} {classes[k]}" for k in split["target_classes"])
    return (
        "torchvision ResNet-50 (IMAGENET1K_V1): conv1 7x7/2, bn1, relu, maxpool, then layer1..layer4 of Bottleneck "
        "blocks [3,4,6,3] (1x1, 3x3, 1x1 convolutions with batch norm; downsample on the first block of each stage), "
        "global average pooling (module 'avgpool', 2048 channels) and fc Linear(2048,1000). Tensor names are "
        "torchvision's (e.g. layer4.2.conv3.weight, fc.weight, fc.bias). The input is uint8 NCHW RGB 224x224; the network "
        "normalizes it inside (ImageNet mean/std), so pass practice inputs unchanged. Outputs are 1000 ImageNet logits. "
        "Practice targets: a clean photo of class y has target y; a spurious photo of class k (cue without the object) "
        "has target -(k+1). Item score: a clean photo scores when the top-1 class equals its label; a spurious photo "
        "scores when the top-1 class is NOT k. Items are weighted so the mean is the average of three groups: spurious "
        "photos of target classes, clean photos of target classes, clean photos of all other classes. "
        f"The {len(split['target_classes'])} target classes (index name): {targets}."
    )


def run(arm, fixture, out, caps):
    fixture, out = Path(fixture).resolve(), Path(out).resolve()
    if set(caps) != set(MODELS) or any(v <= 0 for v in caps.values()):
        raise ValueError("supply positive caps for all three providers")
    if not (fixture / "baseline.json").exists():
        raise ValueError("prepare and verify the fixture first")
    model, practice, evaluation, split = task.load_fixture(fixture)
    classes = json.loads((fixture / "classes.json").read_text())
    description = describe(split, classes)
    out.mkdir(parents=True, exist_ok=False)
    plan = {"arm": arm, "fixture": str(fixture), "fixture_sha256": task.digest(fixture / "manifest.json"),
            "task_sha256": task.digest(Path(task.__file__)), "trial_sha256": task.digest(Path(__file__)),
            "runner_sha256": task.digest(Path(torch_runner.__file__)), "models": MODELS, "providers": list(MODELS),
            "caps": caps, "prices": PRICES, "effort": "high", "max_output_tokens": 16384, "passes": PASSES,
            "max_runs": MAX_RUNS, "max_turns": MAX_TURNS, "seed": task.SEED, "note": NOTE,
            "description": description, "created_at": time.time(),
            "guard": "reported-prefix-tokens-v2-with-missing-output-ceiling"}
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    ledger = Ledger(out / "ledger.json", caps)
    editors = [Guarded(editor(p), p, ledger, max_output=16384) for p in MODELS]
    spending = Spending(max_usd=sum(caps.values()), prices=PRICES)
    started = time.monotonic()
    try:
        result = wsplab.search(
            model, editors=editors, spending=spending, shared=arm == "peers", build=task.build,
            description=description, note=NOTE, practice=practice, evaluate=evaluation,
            folder=out / "run", seed=task.SEED, max_runs=MAX_RUNS, budget=PASSES, device="cuda",
            limits=Limits(seconds=300, memory_gb=8),
            policy=Policy(handoff_every=10**9, require_all_passes=False, max_turns=MAX_TURNS))
        summary = {"arm": arm, "status": "completed", "verdict": result.result.verdict,
                   "candidates": result.candidates, "selection": result.selection,
                   "seconds": time.monotonic() - started, "spend": ledger.spent}
    except Exception as exc:
        summary = {"arm": arm, "status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:500],
                   "seconds": time.monotonic() - started, "spend": ledger.spent}
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        raise
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("arm", "status", "verdict", "spend")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["peers", "independent"], required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    for provider in MODELS:
        parser.add_argument(f"--{provider}-cap", type=float, required=True)
    args = parser.parse_args()
    run(args.arm, args.fixture, args.out, {p: getattr(args, f"{p}_cap") for p in MODELS})
