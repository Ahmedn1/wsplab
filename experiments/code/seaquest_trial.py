"""Paid, bounded three-provider trial on MinAtar Seaquest: peers (OpenAI, Claude, Gemini sharing every step) or the same
three working alone. Starts from the network Astra wrote in the paper. Rules follow the paper's harness."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import wsplab
from wsplab.editors import Spending
from wsplab.kernel import Limits
from wsplab.runners.torch import TorchRunner
from wsplab.session.driver import Policy

if __package__:
    from . import minatar_seaquest as task
    from .peer_trial import MODELS, PRICES, Guarded, Ledger, editor
else:
    import minatar_seaquest as task
    from peer_trial import MODELS, PRICES, Guarded, Ledger, editor

# The cheaper model of each provider. Prices are conservative: Haiku at its long-prompt rate (and fresh input at the
# cache-write rate, as the ledger does for Anthropic); Luna and Gemini 3.8 Flash from published trackers.
CHEAP = {"openai": "gpt-6-luna", "anthropic": "claude-haiku-5-5", "gemini": "gemini-3.8-flash"}
CHEAP_PRICES = {"openai:gpt-6-luna": (.10, .50, .01), "anthropic:claude-haiku-5-5": (.625, 2.50, .05),
                "gemini:gemini-3.8-flash": (.75, 3.75, .075)}


def use_tier(tier):
    """Point the shared editor factory and spending guard at the chosen tier (they read these tables at call time)."""
    if tier == "cheap":
        MODELS.clear()
        MODELS.update(CHEAP)
        PRICES.update(CHEAP_PRICES)


TEAM = ("The team: Peer A is the CRITIC (OpenAI), Peer B is the PROPOSER (Claude), Peer C is the ANALYST (Gemini). "
        "Everyone keeps their own weights and sees the others' steps and results. Only one edit can win, chosen on "
        "fresh episodes, and it will usually be the proposer's, so help it be as good as possible. ")
ROLES = {
    "openai": TEAM + "You are the CRITIC. Whenever a teammate tests a candidate edit, rebuild it in your own weights from "
              "the code they showed and look for where it fails: episodes and situations where it does worse than the "
              "original, new or more frequent ways to die (for example running out of oxygen), and actions it now "
              "takes wrongly. Report concrete failure cases with numbers. Then propose and test a patch that keeps the "
              "candidate's gains and fixes a failure you found. Do not start your own unrelated line of edits.",
    "anthropic": TEAM + "You are the PROPOSER. Design, test and refine the best edit you can, and submit it. Read the "
                 "analyst's diagnoses before choosing what to change, and treat the critic's failure reports as tests "
                 "your edit must pass: fold their working patches into your candidate when the evidence supports it.",
    "gemini": TEAM + "You are the ANALYST. Do not edit early. Diagnose the original policy: from recorded episodes "
              "(steps, actions, logits, activations) and the weights, find exactly how and when it fails (which units, "
              "which situations, how often), and report short, precise, testable hypotheses the proposer can act on. "
              "Keep each reply short; spend passes on diagnostic episodes rather than on long analysis. Later, check "
              "whether the proposer's candidate fixes the failure modes you found.",
}


SEED = 20261108
PASSES, MAX_RUNS, MAX_TURNS = 8, 40, 60
MAX_CHANGED = 1024   # 5% of the 20,486 parameters

NOTE = (
    "Make this policy score higher by editing its weights. "
    "RULES (as in the original weight-space programming work). Not allowed: any training (no gradients, autograd, "
    "optimizers or iterative fitting of weights to a loss); running or simulating the network in Python instead of "
    "through the forward tool, including replaying a layer on recorded inputs or playing the game in Python with the "
    "network to score a candidate. Allowed: any analysis of the weights and of forward-tool results (recorded inputs, "
    "actions, logits, activations), code about the game's rules that does not run the network, and any edit you compute "
    "directly, including closed-form calculations. A session that simulates the network is disqualified and its edit is "
    "discarded. "
    "AGAINST OVERFITTING: fix DEV episodes for choosing edits and separate HOLDOUT episodes for confirming them before "
    "any edit; every pass plays at least 100 episodes; freeze your best candidate before its HOLDOUT check. The final "
    "score comes from fresh episodes you never see, so an edit that only helps particular practice episodes will not "
    "count. Restore the original weights if no candidate is supported by the evidence."
)


def run(arm, fixture, out, caps, tier="strong", roles=False):
    use_tier(tier)
    fixture, out = Path(fixture).resolve(), Path(out).resolve()
    if set(caps) != set(MODELS):
        raise ValueError("supply caps for all three providers")
    model = fixture / "model.safetensors"
    practice = task.SeaquestScorer(task.PRACTICE_SEEDS, "Seaquest practice episodes")
    evaluation = task.SeaquestScorer(task.EVAL_SEEDS, "Seaquest evaluation episodes")
    work, _ = practice.split(.5, SEED)         # the same split search() makes internally
    runner = TorchRunner(task.build, "cpu")
    description = task.description()
    out.mkdir(parents=True, exist_ok=False)
    plan = {"arm": arm, "tier": tier, "roles": ROLES if roles else None, "fixture": str(fixture), "manifest": json.loads((fixture / "manifest.json").read_text()),
            "task_sha256": task.digest(Path(task.__file__)), "trial_sha256": task.digest(Path(__file__)),
            "models": MODELS, "caps": caps, "prices": PRICES, "effort": "high", "max_output_tokens": 16384,
            "passes": PASSES, "require_all_passes": True, "max_runs": MAX_RUNS, "max_turns": MAX_TURNS,
            "max_changed": MAX_CHANGED, "seed": SEED, "note": NOTE, "description_sha256":
            __import__("hashlib").sha256(description.encode()).hexdigest(), "created_at": time.time(),
            "guard": "reported-prefix-tokens-v2-with-missing-output-ceiling"}
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    ledger = Ledger(out / "ledger.json", caps)
    editors = [Guarded(editor(p), p, ledger, max_output=16384) for p in MODELS]
    if roles:
        from legacy_council import Role

        editors = [Role(e, ROLES[p]) for e, p in zip(editors, MODELS)]
    started = time.monotonic()
    try:
        result = wsplab.search(
            model, editors=editors, spending=Spending(max_usd=sum(caps.values()), prices=PRICES),
            shared=arm == "peers", build=task.build, description=description, note=NOTE, practice=practice,
            evaluate=evaluation, folder=out / "run", seed=SEED, max_runs=MAX_RUNS, budget=PASSES, device="cpu",
            max_changed=MAX_CHANGED, limits=Limits(seconds=300, memory_gb=6),
            tools=lambda w, base, store, edited: task.tools(w, base, store, edited, work, runner),
            policy=Policy(handoff_every=10**9, require_all_passes=True, max_turns=MAX_TURNS))
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
    parser.add_argument("--tier", choices=["strong", "cheap"], default="strong")
    parser.add_argument("--roles", action="store_true", help="give the peers different jobs (proposer, critic, analyst)")
    for provider in MODELS:
        parser.add_argument(f"--{provider}-cap", type=float, required=True)
    args = parser.parse_args()
    run(args.arm, args.fixture, args.out, {p: getattr(args, f"{p}_cap") for p in MODELS}, args.tier, args.roles)
