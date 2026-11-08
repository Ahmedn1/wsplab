"""Audit screen and held-out evaluation for the Seaquest trial; no paid calls.

Without --evaluate: audit-code.py per session and audit-screen.json. With --evaluate (needs protocol-audit.json):
2,000 fresh evaluation episodes for the original, every candidate and a same-size random control of each arm's pick."""
from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from pathlib import Path

import numpy as np

if __package__:
    from . import minatar_seaquest as task
else:
    import minatar_seaquest as task

ARMS = ("peers", "independent")
FLAGS = r"\b(?:backward|autograd|optim|lstsq|pinv|minimize|tanh|softmax|minatar|Environment|seaquest|act\(|step\()"


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    n, k = pos + neg, min(pos, neg)
    p = min(1., 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.
    return {"difference": float(d.mean()), "standard_error": float(d.std(ddof=1) / math.sqrt(len(d))),
            "better": pos, "worse": neg, "sign_test_p": p}


def candidates(folder):
    out = []
    for arm in ARMS:
        search = json.loads((folder / arm / "run/search.json").read_text())
        for c in search["candidates"]:
            out.append({**c, "arm": arm, "key": f"{arm}/{c['editor'].split(':')[0]}",
                        "selected_as_run": c["run"] == search["selection"]["run"]})
    return out


def audit(folder):
    rows = []
    for c in candidates(folder):
        run = Path(c["folder"])
        calls = [json.loads(line) for line in (run / "session/tool_calls.jsonl").read_text().splitlines()]
        code = "\n\n".join(f"# --- record {i} ({x['kind']})\n" + (x.get("code") or x.get("request") or "")
                           for i, x in enumerate(calls))
        (run / "audit-code.py").write_text(code)
        body = [ln for ln in code.splitlines() if not ln.lstrip().startswith("#")]
        rows.append({"session": c["key"], "changed_entries": c["changed_entries"], "stop": c.get("stop_reason"),
                     "flags": sorted(set(re.findall(FLAGS, "\n".join(body)))),
                     "matrix_lines": [ln.strip()[:200] for ln in body
                                      if re.search(r"(?<![@\w])@(?!\w)|\b(?:matmul|einsum|dot|mm)\s*\(", ln)]})
    (folder / "audit-screen.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def evaluate(folder, fixture):
    from wsplab import patch as P
    from wsplab.eval.gate import random_control
    from wsplab.io.safetensors import Checkpoint
    from wsplab.runners.torch import TorchRunner

    valid = {k: v["valid"] for k, v in json.loads((folder / "protocol-audit.json").read_text())["sessions"].items()}
    base = Path(fixture) / "model.safetensors"
    runner = TorchRunner(task.build, "cpu")
    seeds = list(task.EVAL_SEEDS)

    def measure(weights):
        summary, _ = task.play(runner.call(weights), seeds)
        return summary

    def describe(s):
        ends = np.bincount(s[:, 2].astype(int), minlength=4)
        return {"mean_return": float(s[:, 0].mean()), "se": float(s[:, 0].std(ddof=1) / math.sqrt(len(s))),
                "mean_length": float(s[:, 1].mean()), "ends": {n: int(ends[c]) for n, c in task.END.items()}}

    original = measure(base)
    cands, per, table = candidates(folder), {}, []
    sel_base = None
    for c in cands:
        s = measure(Path(c["folder"]) / "result/model.safetensors") if c["changed_entries"] else original
        per[c["key"]] = s
        table.append({"session": c["key"], "valid": valid[c["key"]], "changed_entries": c["changed_entries"],
                      "selection_score": c["score"], "stop": c.get("stop_reason"), **describe(s),
                      "minus_original": paired(s[:, 0], original[:, 0])})
    sel_base = {c["arm"]: c["score"] - c["gain"] for c in cands}

    def choose(arm, valid_only):
        pool = [c for c in cands if c["arm"] == arm and c["changed_entries"] and (valid[c["key"]] or not valid_only)]
        best = max(pool, key=lambda c: c["score"], default=None)
        if best is None or best["score"] <= sel_base[arm]:
            return "original", original
        return best["key"], per[best["key"]]

    views = {v: {a: choose(a, v == "valid_only") for a in ARMS} for v in ("valid_only", "as_run")}
    controls = {}
    for key in sorted({views[v][a][0] for v in views for a in ARMS} - {"original"}):
        c = next(x for x in cands if x["key"] == key)
        run = Path(c["folder"])
        with tempfile.TemporaryDirectory() as tmp:
            noise = random_control(Checkpoint(base), P.Patch.load(run / "result/patch"),
                                   Checkpoint(run / "result/model.safetensors"))
            model = P.apply(P.from_submission(Checkpoint(base), noise, Path(tmp) / "patch"), Checkpoint(base),
                            Path(tmp) / "model.safetensors")
            controls[key] = describe(measure(model))
    result = {"original": describe(original), "episodes": len(seeds), "table": table, "random_controls": controls,
              "chosen": {v: {a: views[v][a][0] for a in ARMS} for v in views},
              "primary_peers_minus_portfolio_valid_only": paired(views["valid_only"]["peers"][1][:, 0],
                                                                 views["valid_only"]["independent"][1][:, 0]),
              "peers_minus_portfolio_as_run": paired(views["as_run"]["peers"][1][:, 0],
                                                     views["as_run"]["independent"][1][:, 0]),
              "peers_minus_each_solo": {c["key"]: paired(views["valid_only"]["peers"][1][:, 0], per[c["key"]][:, 0])
                                        for c in cands if c["arm"] == "independent"},
              "costs": {a: json.loads((folder / a / "ledger.json").read_text())["spent"] for a in ARMS}}
    (folder / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--fixture", type=Path, default=Path("~/.wsplab/seaquest/fixture").expanduser())
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    out = evaluate(args.folder, args.fixture) if args.evaluate else audit(args.folder)
    print("written")
