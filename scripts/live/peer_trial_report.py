"""Post-run exploratory diagnostics; refuses to score until both arms have finished."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from wsplab.runners.torch import TorchRunner

if __package__:
    from . import modular_net as task
else:
    import modular_net as task


def report(folder):
    plans, summaries = {}, {}
    for arm in ("peers", "independent"):
        plans[arm] = json.loads((folder / arm / "plan.json").read_text())
        summaries[arm] = json.loads((folder / arm / "summary.json").read_text())
        if summaries[arm]["status"] != "completed":
            raise ValueError(f"{arm} failed; do not present a complete performance comparison")
    for key in ("seed", "task_sha256", "models", "effort", "max_runs", "passes", "caps"):
        if plans["peers"][key] != plans["independent"][key]:
            raise ValueError(f"arms differ on {key}")
    seed = plans["peers"]["seed"]
    inputs, targets = task.queries(*task.image_splits(seed)["evaluation"], seed)
    routes = inputs[:, 64:].argmax(1)
    runner = TorchRunner(task.build)
    rows, selected, solo_correct = [], {}, {}
    for arm in ("peers", "independent"):
        search = json.loads((folder / arm / "run/search.json").read_text())
        for candidate in search["candidates"]:
            run_folder = Path(candidate["folder"])
            correct = runner.call(run_folder / "result/model.safetensors")(inputs).argmax(1) == targets
            state = json.loads((run_folder / "session/state.json").read_text())
            calls_path = run_folder / "session/tool_calls.jsonl"
            calls = [json.loads(line) for line in calls_path.read_text().splitlines()] if calls_path.exists() else []
            code = "\n".join(call.get("code", "") for call in calls)
            flags = sorted(set(re.findall(r"\b(?:backward|autograd|optim|lstsq|pinv|fit|fit_transform|minimize)\b", code)))
            matrix_lines = [line.strip() for line in code.splitlines()
                            if not line.lstrip().startswith("#") and re.search(r"@|\b(?:matmul|einsum|dot)\s*\(", line)]
            answers = [p.read_text() for p in (run_folder / "session/turns").glob("*/answer.md")]
            chosen = candidate["run"] == search["selection"]["run"]
            row = {"arm": arm, "editor": candidate["editor"], "run": candidate["run"],
                   "selected": chosen, "selection_score": candidate["score"],
                   "diagnostic_evaluation_score": float(correct.mean()),
                   "per_branch": [float(correct[routes == d].mean()) for d in range(10)],
                   "turns": state["turn"], "python_runs": state["runs"], "passes": state["passes"],
                   "stop": state["stop_reason"], "changed_entries": candidate["changed_entries"],
                   "potential_fit_flags_require_review": flags,
                   "kernel_matrix_operations_require_review": matrix_lines,
                   "replies_mentioning_peers": sum(bool(re.search(r"\bpeer\b", text, re.IGNORECASE)) for text in answers),
                   "folder": str(run_folder)}
            rows.append(row)
            if arm == "independent":
                solo_correct[candidate["editor"]] = correct.astype(float)
            if chosen:
                selected[arm] = correct.astype(float)
    delta = selected["peers"] - selected["independent"]
    ledgers = {arm: json.loads((folder / arm / "ledger.json").read_text()) for arm in plans}
    result = {"seed": seed, "evaluation_items": len(targets), "rows": rows,
              "selected_peer_minus_independent": float(delta.mean()),
              "paired_standard_error": float(delta.std(ddof=1) / np.sqrt(len(delta))),
              "cost_estimate": {arm: ledger["spent"] for arm, ledger in ledgers.items()},
              "note": "Individual candidate evaluation is exploratory; candidates were selected using selection data only."}
    result["peer_vs_each_solo"] = {}
    for editor, correct in solo_correct.items():
        differences = selected["peers"] - correct
        result["peer_vs_each_solo"][editor] = {
            "difference": float(differences.mean()),
            "paired_standard_error": float(differences.std(ddof=1) / np.sqrt(len(differences))),
            "wrong_to_right": int((differences > 0).sum()), "right_to_wrong": int((differences < 0).sum())}
    result["complete_solo_baselines"] = [r["editor"] for r in rows if r["arm"] == "independent" and r["stop"] == "finished"]
    (folder / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(report(args.folder), indent=2))
