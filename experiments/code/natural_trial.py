"""Paid, bounded three-provider peer/independent trial on a frozen natural fixture."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import wsplab
from wsplab.editors import Spending
from wsplab.kernel import Limits
from wsplab.runners import torch as torch_runner
from wsplab.session.driver import Policy

if __package__:
    from . import natural_cifar as task
    from .peer_trial import MODELS, PRICES, Guarded, Ledger, editor
else:
    import natural_cifar as task
    from peer_trial import MODELS, PRICES, Guarded, Ledger, editor


CAPS = {"openai": .35, "anthropic": 2., "gemini": 3.5}
NOTE = (
    "Improve ordinary CIFAR-100 fine-label classification by editing this unmodified pretrained model. "
    "No damage or corruption has been injected. Some errors may not be fixable without training; "
    "a defensible no-change result is preferable to an unsupported repair. "
    "Do NOT train or optimize any loss. Do NOT compute predictions, logits, losses or accuracy for "
    "modified weights inside Python, including replaying the classifier on saved activations. "
    "That is a forward pass and must use the official forward tool. Weight statistics, correlations "
    "and analysis of already-recorded outputs are allowed. Do not reload public data or checkpoints. "
    "Keep your internal practice holdout separate from development and freeze before confirming. "
    "Save a plausible candidate early; retain or restore the original if evidence does not support it. "
    "Use concise analysis and compact diagnostic outputs; a paid-token ceiling also applies."
)


def run(arm, fixture, out, caps=None):
    fixture, out = Path(fixture).resolve(), Path(out).resolve()
    caps = dict(CAPS if caps is None else caps)
    if set(caps) != set(MODELS) or any(value <= 0 for value in caps.values()):
        raise ValueError("supply positive caps for all three providers")
    if not (fixture / "baseline.json").exists():
        raise ValueError("prepare and verify the natural baseline first")
    model, practice, evaluation = task.load_fixture(fixture)
    classes = json.loads((fixture / "classes.json").read_text())
    description = (
        "an unmodified public CIFAR-100 ResNet-20, 3 residual blocks in each of three stages "
        "with widths 16,32,64; 3x3 convolutions, batch normalization, ReLU, global average pooling "
        "and Linear(64,100). Stride-2 transitions use 1x1 projection shortcuts. The input is NCHW "
        "RGB 32x32, converted to [0,1] and normalized by channel mean [.5070,.4865,.4409] and "
        "std [.2673,.2564,.2761]. Inputs are already normalized; never renormalize them. "
        "Outputs are 100 fine-class logits. Class indices in order: " + json.dumps(classes)
    )
    out.mkdir(parents=True, exist_ok=False)
    plan = {"arm": arm, "fixture": str(fixture), "fixture_sha256": task.digest(fixture / "manifest.json"),
            "task_sha256": task.digest(Path(task.__file__)), "models": MODELS, "providers": list(MODELS),
            "caps": caps, "prices": PRICES, "effort": "high", "max_output_tokens": 16384,
            "runner_sha256": task.digest(Path(torch_runner.__file__)),
            "passes": 8, "max_runs": 40, "max_turns": 60, "seed": task.SEED,
            "note": NOTE, "description": description, "created_at": time.time(),
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
            folder=out / "run", seed=task.SEED, max_runs=40, budget=8,
            limits=Limits(seconds=180, memory_gb=4),
            policy=Policy(handoff_every=10**9, require_all_passes=False, max_turns=60))
        summary = {"arm": arm, "status": "completed", "verdict": result.result.verdict,
                   "baseline": result.result.baseline, "final": result.result.final,
                   "candidates": result.candidates, "selection": result.selection,
                   "seconds": time.monotonic() - started, "spend": ledger.spent}
    except Exception as exc:
        summary = {"arm": arm, "status": "failed", "error_type": type(exc).__name__,
                   "seconds": time.monotonic() - started, "spend": ledger.spent}
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        raise
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["peers", "independent"], required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    for provider, cap in CAPS.items():
        parser.add_argument(f"--{provider}-cap", type=float, default=cap)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.arm, args.fixture, args.out, {p: getattr(args, f"{p}_cap") for p in CAPS})
