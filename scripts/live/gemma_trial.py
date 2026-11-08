"""Paid, bounded three-provider trial on the paper's small-gain case: Gemma 3 270M IT on math.

In the paper's follow-up, Astra raised Gemma's MATH-500 score only from 6.8% to 7.0-7.7% across 5-50 passes. Here
OpenAI, Claude and Gemini edit it as peers (sharing every step) and, separately, alone. The sessions use the paper's
prompt (the v5 template the Gemma runs used), sampling and lenient grading, with 10 passes each.

Each session's own end-of-run score is on a SELECTION set held out from the practice problems, which picks each arm's
candidate. MATH-500 and the guard checks are run only after the protocol audit (gemma_report.py)."""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

if __package__:
    from .peer_trial import MODELS, PRICES, Guarded, Ledger, editor
else:
    from peer_trial import MODELS, PRICES, Guarded, Ledger, editor

MODEL, REVISION, NAME = "google/gemma-3-270m-it", "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3", "Gemma 3 270M IT"
SAMPLING = {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "max_tokens": 8192}   # the model card's, as in the paper
SEED = 20261108
PASSES, MAX_RUNS, MAX_TURNS = 10, 40, 60
MAX_CHANGED = 500_000       # about 0.5% of the editable (non-embedding) weights
SELECTION = {"math": 400, "gsm8k": 200}
SCRATCH = Path("/tmp/wsplab-gemma-trial")

NOTE = (
    "Additional rules for this run. A session that runs or simulates the network, or any part of it, in the kernel "
    "(for example replaying layers on recorded activations to score a candidate) is disqualified and its edit is "
    "discarded. Closed-form edits you compute directly are allowed, as the rules above say. The final score comes from "
    "a separate math benchmark you never see, with further checks on other math problems and on general text, so an "
    "edit that only helps particular practice problems, or that damages the model elsewhere, will not count."
)


def digest_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def suites():
    """Editor practice (the paper's practice minus a held-out selection set) and the selection suite."""
    from wsplab.eval.lm_suite import INSTRUCTION, MATH_PRACTICE, LMSuite, read_jsonl
    from wsplab.eval.graders import get_grader
    from wsplab.upstream import fetch

    rows = []
    for name in MATH_PRACTICE:
        rows += read_jsonl(fetch(f"exp8_math/practice/{name}"))
    rng = np.random.default_rng(SEED)
    held = set()
    for source, k in SELECTION.items():
        ids = [r["id"] for r in rows if r["source"] == source]
        held |= set(rng.choice(ids, k, replace=False).tolist())
    practice = [r for r in rows if r["id"] not in held]
    selection = [{"id": r["id"], "problem": r["problem"], "answer": r["answer"], "source": r["source"]}
                 for r in rows if r["id"] in held]
    suite = LMSuite(practice, selection, get_grader("math-lenient"), SAMPLING, benchmark="held-out practice problems",
                    samples=4, user_message=lambda row: f"{row['problem']}\n{INSTRUCTION}")
    return suite


class LockedLM:
    """One LM runner shared by every session: GPU calls are serialized, and answers of the unedited model are cached
    across sessions (same prompt, seed, n and sampling give the same answers)."""

    def __init__(self, runner, base: Path):
        self.runner, self.base = runner, Path(base).resolve()
        self.lock = threading.RLock()
        self.cache: dict[str, list] = {}
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.runner, name)

    def _is_base(self, weights):
        return Path(weights).resolve() == self.base

    def generate(self, weights, prompts, n, sampling):
        with self.lock:
            self.calls += 1
            if not self._is_base(weights):
                return self.runner.generate(weights, prompts, n, sampling)
            keys = [json.dumps([p.text, p.seed, n, sampling], sort_keys=True) for p in prompts]
            missing = [i for i, k in enumerate(keys) if k not in self.cache]
            if missing:
                fresh = self.runner.generate(weights, [prompts[i] for i in missing], n, sampling)
                for i, result in zip(missing, fresh):
                    self.cache[keys[i]] = result
            return [self.cache[k] for k in keys]

    def read(self, *args, **kwargs):
        with self.lock:
            return self.runner.read(*args, **kwargs)

    def activations(self, *args, **kwargs):
        with self.lock:
            return self.runner.activations(*args, **kwargs)


def run(out, caps, scripted=None, budget=PASSES):
    """`scripted` (offline checks only): a function (arm, index) -> editor replacing the paid editors."""
    import wsplab
    from wsplab.council import Board, Peer
    from wsplab.kernel import Limits
    from wsplab.passes import ForwardLimits
    from wsplab.patch.format import Patch
    from wsplab.pipeline import resolve_model
    from wsplab.runners import LM
    from wsplab.session.driver import Policy

    out = Path(out).resolve()
    if set(caps) != set(MODELS):
        raise ValueError("supply caps for all three providers")
    for provider in MODELS:
        if scripted is None and not Path(f"/tmp/{provider}_key.txt").is_file():
            raise ValueError(f"missing {provider} key file")
    out.mkdir(parents=True, exist_ok=False)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    suite = suites()
    base = resolve_model(MODEL, REVISION)
    plan = {"model": MODEL, "revision": REVISION, "sampling": SAMPLING, "seed": SEED, "passes": PASSES,
            "require_all_passes": True, "max_runs": MAX_RUNS, "max_turns": MAX_TURNS, "max_changed": MAX_CHANGED,
            "selection_ids": [r["id"] for r in suite.evaluation], "practice_rows": len(suite.practice),
            "selection_samples": suite.samples, "note": NOTE, "models": MODELS, "caps": caps, "prices": PRICES,
            "trial_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "created_at": time.time(),
            "guard": "reported-prefix-tokens-v2-with-missing-output-ceiling"}
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    shared = LockedLM(LM(base, thinking=False, device="cuda"), base)
    _ = shared.tok   # finish lazy tokenizer imports in the main thread, before the session threads start

    def session(arm, index, editor_obj, board, ledger_path):
        provider = list(MODELS)[index]
        dest = out / arm / provider
        (dest / "run").mkdir(parents=True)
        for name in ("checkpoints", "result"):     # model copies go to /tmp (disk on /home is short)
            target = SCRATCH / out.name / arm / provider / name
            target.mkdir(parents=True, exist_ok=True)
            (dest / "run" / name).symlink_to(target, target_is_directory=True)
        try:
            result = wsplab.run(
                MODEL, revision=REVISION, name=NAME, eval=suite, editor=editor_obj, note=NOTE, budget=budget,
                max_runs=MAX_RUNS, folder=dest / "run", runner=shared, control=False, max_changed=MAX_CHANGED,
                limits=Limits(seconds=600, memory_gb=5), pass_time="a few minutes",
                policy=Policy(handoff_every=10**9, require_all_passes=True, max_turns=MAX_TURNS),
                forward_limits=ForwardLimits(n_layers=18, full_attention=(5, 11, 17), gen_max=1024, max_tokens=8192,
                                             tf_max_tokens=8192))
            changed = Patch.load(result.patch).manifest["changed_entries_total"]
            row = {"arm": arm, "provider": provider, "changed_entries": changed,
                   "selection_score": result.final if changed else result.baseline, "baseline": result.baseline,
                   "patch": str(result.patch), "weights": str(result.weights), "passes": result.state.passes,
                   "runs": result.state.runs, "turns": result.state.turn, "stop_reason": result.state.stop_reason,
                   "verdict_on_selection": result.verdict}
        except Exception as exc:  # noqa: BLE001 - one failed session is recorded, the others continue
            row = {"arm": arm, "provider": provider, "status": "failed", "error_type": type(exc).__name__,
                   "error": str(exc)[:500]}
        finally:
            if board is not None:
                board.leave(index)
        (dest / "candidate.json").write_text(json.dumps(row, indent=2) + "\n")
        return row

    jobs = []
    for arm in ("peers", "independent"):
        (out / arm).mkdir()
        ledger = Ledger(out / arm / "ledger.json", caps)
        board = Board(len(MODELS)) if arm == "peers" else None
        for index, provider in enumerate(MODELS):
            ed = scripted(arm, index) if scripted else Guarded(editor(provider), provider, ledger, max_output=16384)
            if board is not None:
                ed = Peer(ed, board, index)
            jobs.append((arm, index, ed, board, ledger))
    started = time.monotonic()
    with ThreadPoolExecutor(len(jobs)) as pool:
        rows = list(pool.map(lambda j: session(*j), jobs))
    summary = {"status": "awaiting_protocol_audit", "rows": rows, "seconds": time.monotonic() - started,
               "spend": {arm: json.loads((out / arm / "ledger.json").read_text())["spent"] for arm in ("peers", "independent")},
               "lm_calls": shared.calls}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "spend": summary["spend"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    for provider in MODELS:
        parser.add_argument(f"--{provider}-cap", type=float, required=True)
    args = parser.parse_args()
    run(args.out, {p: getattr(args, f"{p}_cap") for p in MODELS})
