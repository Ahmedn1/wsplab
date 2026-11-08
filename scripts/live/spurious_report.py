"""Score the frozen Spurious ResNet candidates after both arms finish and the audit passes; no paid calls.

Writes audit-code.py per session (for manual review) and, with --evaluate, comparison.json with every candidate,
the recipe comparators, the pre-registered pass bar and paired tests."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from wsplab import patch as P
from wsplab.eval.gate import random_control
from wsplab.io.safetensors import Checkpoint
from wsplab.runners.torch import TorchRunner

if __package__:
    from . import spurious_resnet as task
else:
    import spurious_resnet as task

ARMS = ("peers", "independent")
FLAGS = r"\b(?:backward|autograd|optim|lstsq|pinv|solve|fit|minimize|softmax|log_softmax|argmax|conv2d|linear)\b"


def paired(a, b):
    """a minus b, per item; exact sign test on items that differ."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())
    n, k = n_pos + n_neg, min(n_pos, n_neg)
    p = min(1., 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.
    return {"difference": float(d.mean()), "standard_error": float(d.std(ddof=1) / math.sqrt(len(d))),
            "better": n_pos, "worse": n_neg, "sign_test_p": p}


def audit(folder):
    rows = []
    for arm in ARMS:
        search = json.loads((folder / arm / "run/search.json").read_text())
        for c in search["candidates"]:
            run = Path(c["folder"])
            calls = [json.loads(line) for line in (run / "session/tool_calls.jsonl").read_text().splitlines()]
            code = "\n\n".join(f"# --- record {i} ({c_['kind']})\n" + (c_.get("code") or c_.get("request") or "")
                               for i, c_ in enumerate(calls))
            (run / "audit-code.py").write_text(code)
            body = [ln for ln in code.splitlines() if not ln.lstrip().startswith("#")]
            rows.append({"arm": arm, "editor": c["editor"], "folder": str(run), "changed_entries": c["changed_entries"],
                         "flags": sorted(set(re.findall(FLAGS, "\n".join(body)))),
                         "matrix_lines": [ln.strip() for ln in body if re.search(r"@|\b(?:matmul|einsum|dot|mm)\s*\(", ln)]})
    (folder / "audit-screen.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def evaluate(folder, recipe_dir):
    plans = {a: json.loads((folder / a / "plan.json").read_text()) for a in ARMS}
    for key in ("fixture_sha256", "task_sha256", "trial_sha256", "models", "caps", "passes", "max_runs", "max_turns",
                "note", "description", "seed"):
        if plans["peers"][key] != plans["independent"][key]:
            raise ValueError(f"unmatched arms: {key}")
    audit = json.loads((folder / "protocol-audit.json").read_text())
    if "sessions" not in audit:
        raise ValueError("a completed protocol audit is required first")
    valid = {k: v["valid"] for k, v in audit["sessions"].items()}
    model, practice, evaluation, split = task.load_fixture(plans["peers"]["fixture"])
    targets = split["target_classes"]
    runner = TorchRunner(task.build, "cuda")
    _, selection = practice.split(.5, plans["peers"]["seed"])
    ids = evaluation.ids()

    def score(weights):
        call = runner.call(weights)
        out = evaluation.outputs(call, ids)
        per = task.make_metric(targets)(out, evaluation.targets)
        return per, {"balanced": float(per.mean()), **task.metrics(out, evaluation.targets, targets)}

    base_sel = selection.score(runner.call(model), selection.ids()).mean
    base_per, base = score(model)
    bar = {"target_spurious_hit": base["target_spurious_hit"] / 2, "target_clean_top1": base["target_clean_top1"] - .03,
           "clean_top1": base["clean_top1"] - .005}

    def passes(m):
        return (m["target_spurious_hit"] <= bar["target_spurious_hit"] and m["target_clean_top1"] >= bar["target_clean_top1"]
                and m["clean_top1"] >= bar["clean_top1"])

    rows, solo_per, pool = [], {}, {a: [] for a in ARMS}
    for arm in ARMS:
        search = json.loads((folder / arm / "run/search.json").read_text())
        for c in search["candidates"]:
            run = Path(c["folder"])
            weights = run / "result/model.safetensors"
            per, m = score(weights) if c["changed_entries"] else (base_per, base)
            key = f"{arm}/{c['editor'].split(':')[0]}"
            row = {"arm": arm, "editor": c["editor"], "protocol_valid": valid[key],
                   "selected_as_run": c["run"] == search["selection"]["run"],
                   "selection_score": c["score"], "changed_entries": c["changed_entries"], **m,
                   "passes_bar": passes(m), "minus_original": paired(per, base_per)}
            if c["changed_entries"]:
                noise = random_control(Checkpoint(model), P.Patch.load(run / "result/patch"), Checkpoint(weights))
                ctrl = P.apply(P.from_submission(Checkpoint(model), noise, run / "control/patch"), Checkpoint(model),
                               run / "control/model.safetensors")
                row["random_control"] = score(ctrl)[1]
            rows.append(row)
            if arm == "independent":
                solo_per[c["editor"]] = per
            pool[arm].append((c["score"], valid[key], per, c["editor"], bool(c["changed_entries"])))
    def choose(arm, valid_only):
        """Highest selection score; the original is kept if no candidate beats it (the harness's rule)."""
        cands = [x for x in pool[arm] if x[4] and (x[1] or not valid_only)]
        best = max(cands, key=lambda x: x[0], default=None)
        return (best[2], best[3]) if best and best[0] > base_sel else (base_per, "original")

    views = {name: {a: choose(a, name == "valid_only") for a in ARMS} for name in ("valid_only", "as_run")}
    chosen = {a: views["valid_only"][a][0] for a in ARMS}
    recipes = {}
    for name in ("s1", "s2"):
        per, m = score(Path(recipe_dir) / name / "model.safetensors")
        recipes[name] = {**m, "passes_bar": passes(m), "per": per}
    result = {"baseline": base, "bar": bar, "rows": rows,
              "recipes": {k: {kk: vv for kk, vv in v.items() if kk != "per"} for k, v in recipes.items()},
              "chosen": {v: {a: views[v][a][1] for a in ARMS} for v in views},
              "primary_peers_minus_portfolio_valid_only": paired(chosen["peers"], chosen["independent"]),
              "peers_minus_portfolio_as_run": paired(views["as_run"]["peers"][0], views["as_run"]["independent"][0]),
              "peers_minus_each_solo": {e: paired(chosen["peers"], per) for e, per in solo_per.items()},
              "peers_minus_recipe": {k: paired(chosen["peers"], v["per"]) for k, v in recipes.items()},
              "portfolio_minus_recipe": {k: paired(chosen["independent"], v["per"]) for k, v in recipes.items()},
              "costs": {a: json.loads((folder / a / "ledger.json").read_text())["spent"] for a in ARMS}}
    (folder / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--recipe", type=Path, default=Path("~/.wsplab/spurious-resnet/recipe").expanduser())
    args = parser.parse_args()
    out = evaluate(args.folder, args.recipe) if args.evaluate else audit(args.folder)
    print(json.dumps(out, indent=2)[:6000])
