"""Audit screen and held-out evaluation for the Gemma trial; no paid calls.

Without --evaluate: writes audit-code.py per session and audit-screen.json for manual review.
With --evaluate (needs protocol-audit.json): MATH-500 (8 samples, lenient grading), GSM8K test (first 500, 4 samples)
and per-passage log-likelihood on 300 WikiText-103 test passages, for the original, every candidate and a same-size
random control of each arm's pick."""
from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from pathlib import Path

import numpy as np

if __package__:
    from . import gemma_trial as trial
else:
    import gemma_trial as trial

ARMS = ("peers", "independent")
FLAGS = r"\b(?:backward|autograd|optim|lstsq|pinv|minimize|softmax|log_softmax|F\.linear|functional|rms_norm|" \
        r"layer_norm|scaled_dot_product|einsum|matmul)\b"


def paired(a, b):
    """a minus b over items; t-based SE and an exact sign test on items that differ."""
    d = np.asarray(a, float) - np.asarray(b, float)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    n, k = pos + neg, min(pos, neg)
    p = min(1., 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.
    return {"difference": float(d.mean()), "standard_error": float(d.std(ddof=1) / math.sqrt(len(d))),
            "better": pos, "worse": neg, "sign_test_p": p}


def rows(folder):
    return [json.loads((folder / arm / p / "candidate.json").read_text()) for arm in ARMS for p in trial.MODELS]


def audit(folder):
    out = []
    for r in rows(folder):
        run = folder / r["arm"] / r["provider"] / "run"
        calls = [json.loads(line) for line in (run / "session/tool_calls.jsonl").read_text().splitlines()]
        code = "\n\n".join(f"# --- record {i} ({c['kind']})\n" + (c.get("code") or c.get("request") or "")
                           for i, c in enumerate(calls))
        (run / "audit-code.py").write_text(code)
        body = [ln for ln in code.splitlines() if not ln.lstrip().startswith("#")]
        out.append({"arm": r["arm"], "provider": r["provider"], "changed_entries": r.get("changed_entries"),
                    "status": r.get("status", "completed"), "passes": r.get("passes"), "stop": r.get("stop_reason"),
                    "flags": sorted(set(re.findall(FLAGS, "\n".join(body)))),
                    "matrix_lines": [ln.strip()[:200] for ln in body if re.search(r"(?<![@\w])@(?!\w)|\b(?:matmul|einsum|mm|bmm)\s*\(", ln)]})
    (folder / "audit-screen.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def guard_suites():
    from datasets import load_dataset

    from wsplab.eval.graders import get_grader
    from wsplab.eval.lm_suite import INSTRUCTION, LMSuite, math_suite

    math500 = math_suite(trial.SAMPLING, samples=8, lenient=True)
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm_rows = [{"id": f"gsm8k_test_{i}", "problem": r["question"], "answer": r["answer"].split("####")[-1].strip()}
                for i, r in enumerate(gsm) if i < 500]
    gsm_suite = LMSuite([], gsm_rows, get_grader("math-lenient"), trial.SAMPLING, benchmark="GSM8K test", samples=4,
                        user_message=lambda row: f"{row['problem']}\n{INSTRUCTION}", check_overlap=False)
    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    texts = [t.strip() for t in wiki["text"] if len(t.strip()) > 400 and not t.strip().startswith("=")][:300]
    return math500, gsm_suite, texts


def nll(runner, weights, texts):
    out = runner.read(weights, texts, {"token_logprobs": True})
    lp, lengths = out["token_logprobs"], out["lengths"]
    return np.array([-lp[i, :lengths[i] - 1].mean() for i in range(len(texts))])  # column j scores token j+1


def evaluate(folder):
    from wsplab import patch as P
    from wsplab.eval.gate import random_control
    from wsplab.eval.lm_suite import LMEvaluation
    from wsplab.io.safetensors import Checkpoint
    from wsplab.pipeline import resolve_model
    from wsplab.runners import LM

    audit_doc = json.loads((folder / "protocol-audit.json").read_text())
    valid = {k: v["valid"] for k, v in audit_doc["sessions"].items()}
    base = resolve_model(trial.MODEL, trial.REVISION)
    runner = LM(base, thinking=False, device="cuda")
    math500, gsm, texts = guard_suites()

    def measure(weights):
        m = LMEvaluation(runner, math500).evaluate(weights)
        g = LMEvaluation(runner, gsm).evaluate(weights)
        ids_m, ids_g = sorted(m.per_item), sorted(g.per_item)
        return {"math": np.array([np.mean(m.per_item[i]) for i in ids_m]),
                "gsm": np.array([np.mean(g.per_item[i]) for i in ids_g]), "nll": nll(runner, weights, texts)}

    def summary(x):
        return {"math500": float(x["math"].mean()), "gsm8k": float(x["gsm"].mean()), "wikitext_nll": float(x["nll"].mean())}

    original = measure(base)
    table, per = [], {}
    for r in rows(folder):
        key = f"{r['arm']}/{r['provider']}"
        if r.get("status") == "failed":
            table.append({"session": key, "status": "failed", "error": r.get("error")})
            continue
        x = measure(Path(r["weights"])) if r["changed_entries"] else original
        per[key] = x
        table.append({"session": key, "valid": valid[key], "changed_entries": r["changed_entries"],
                      "selection_score": r["selection_score"], "selection_baseline": r["baseline"], **summary(x),
                      "math_minus_original": paired(x["math"], original["math"]),
                      "gsm_minus_original": paired(x["gsm"], original["gsm"]),
                      "nll_minus_original": paired(x["nll"], original["nll"])})

    def choose(arm, valid_only):
        cands = [r for r in rows(folder) if r["arm"] == arm and r.get("status") != "failed" and r["changed_entries"]
                 and (valid[f"{arm}/{r['provider']}"] or not valid_only)]
        best = max(cands, key=lambda r: r["selection_score"], default=None)
        if best is None or best["selection_score"] <= best["baseline"]:
            return "original", original
        return f"{arm}/{best['provider']}", per[f"{arm}/{best['provider']}"]

    views = {v: {a: choose(a, v == "valid_only") for a in ARMS} for v in ("valid_only", "as_run")}
    controls = {}
    picked = sorted({views[v][a][0] for v in views for a in ARMS} - {"original"})
    for key in picked:
        r = next(x for x in rows(folder) if f"{x['arm']}/{x['provider']}" == key)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            noise = random_control(Checkpoint(base), P.Patch.load(r["patch"]), Checkpoint(Path(r["weights"])))
            model = P.apply(P.from_submission(Checkpoint(base), noise, Path(tmp) / "patch"), Checkpoint(base),
                            Path(tmp) / "model")
            m = LMEvaluation(runner, math500).evaluate(model)
            controls[key] = float(np.mean([np.mean(v) for v in m.per_item.values()]))
    result = {"original": summary(original), "table": table, "random_control_math500": controls,
              "chosen": {v: {a: views[v][a][0] for a in ARMS} for v in views},
              "primary_peers_minus_portfolio_valid_only": {
                  k: paired(views["valid_only"]["peers"][1][k], views["valid_only"]["independent"][1][k])
                  for k in ("math", "gsm", "nll")},
              "peers_minus_portfolio_as_run": {
                  k: paired(views["as_run"]["peers"][1][k], views["as_run"]["independent"][1][k]) for k in ("math", "gsm", "nll")},
              "costs": {a: json.loads((folder / a / "ledger.json").read_text())["spent"] for a in ARMS}}
    (folder / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    out = evaluate(args.folder) if args.evaluate else audit(args.folder)
    Path(args.folder / ("comparison-print.json" if args.evaluate else "audit-print.json")).write_text(json.dumps(out, indent=2))
    print("written")
