"""Score frozen natural-task candidates after both arms finish; no paid calls."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from wsplab.runners.torch import TorchRunner

if __package__:
    from . import natural_cifar as task
else:
    import natural_cifar as task


def paired(a, b):
    difference = a.astype(float) - b.astype(float)
    return {"difference": float(difference.mean()),
            "paired_standard_error": float(difference.std(ddof=1) / np.sqrt(len(difference))),
            "wrong_to_right": int((difference > 0).sum()),
            "right_to_wrong": int((difference < 0).sum())}


def report(folder):
    folder = Path(folder)
    plans = {arm: json.loads((folder / arm / "plan.json").read_text()) for arm in ("peers", "independent")}
    for key in ["fixture_sha256", "task_sha256", "models", "providers", "caps", "effort", "max_output_tokens",
                "passes", "max_runs", "max_turns", "seed", "note", "description", "guard", "runner_sha256"]:
        if plans["peers"][key] != plans["independent"][key]:
            raise ValueError(f"unmatched arms: {key}")
    for arm in plans:
        summary = json.loads((folder / arm / "summary.json").read_text())
        if summary["status"] != "completed":
            raise ValueError(f"{arm} did not complete; no complete comparison")
    model, practice, evaluation = task.load_fixture(plans["peers"]["fixture"])
    runner = TorchRunner(task.build)
    _, selection = practice.split(.5, plans["peers"]["seed"])
    baseline_selection = selection.score(runner.call(model), selection.ids()).mean
    baseline = evaluation.outputs(runner.call(model), evaluation.ids()).argmax(1) == evaluation.targets
    rows, chosen, solo, costs = [], {}, {}, {}
    for arm in plans:
        search = json.loads((folder / arm / "run/search.json").read_text())
        if len(search["candidates"]) != len(plans[arm]["providers"]):
            raise ValueError(f"{arm} is missing candidates")
        for candidate in search["candidates"]:
            run = Path(candidate["folder"])
            correct = evaluation.outputs(runner.call(run / "result/model.safetensors"), evaluation.ids()).argmax(1)
            correct = correct == evaluation.targets
            np.save(run / "external_correct.npy", correct)
            state = json.loads((run / "session/state.json").read_text())
            calls = [json.loads(line) for line in (run / "session/tool_calls.jsonl").read_text().splitlines()]
            code = "\n\n".join(f"# Tool record {i}\n" + c.get("code", "") for i, c in enumerate(calls))
            (run / "audit-code.py").write_text(code)
            selected = candidate["run"] == search["selection"]["run"]
            accepted = candidate["score"] > baseline_selection
            rows.append({"arm": arm, "editor": candidate["editor"], "selected": selected,
                         "selection_score": candidate["score"], "evaluation_score": float(correct.mean()),
                         "change_from_baseline": paired(correct, baseline),
                         "operational_fallback_to_original": not accepted,
                         "stop": state["stop_reason"], "turns": state["turn"], "passes": state["passes"],
                         "python_runs": state["runs"], "changed_entries": candidate["changed_entries"],
                         "audit_flags_require_manual_review": sorted(set(re.findall(
                             r"\b(?:backward|autograd|optim|lstsq|pinv|fit|fit_transform|minimize|conv1d|conv2d|conv3d)\b",
                             code))),
                         "matrix_lines_require_manual_review": [line.strip() for line in code.splitlines()
                             if not line.lstrip().startswith("#") and re.search(r"@|\b(?:matmul|einsum|dot)\s*\(", line)],
                         "folder": str(run)})
            operational = correct if accepted else baseline
            if selected:
                chosen[arm] = operational
            if arm == "independent":
                solo[candidate["editor"]] = {"raw": correct, "operational": operational}
        ledger = json.loads((folder / arm / "ledger.json").read_text())
        costs[arm] = {"providers": ledger["spent"], "total": sum(ledger["spent"].values()),
                      "api_errors": sum(bool(c["error"]) for c in ledger["calls"]),
                      "estimated_usage_calls": sum(c["usage_was_estimated"] for c in ledger["calls"])}
    result = {"fixture": plans["peers"]["fixture"], "items": len(baseline),
              "baseline": float(baseline.mean()), "baseline_selection": baseline_selection,
              "rows": rows, "operational_scores": {a: float(c.mean()) for a, c in chosen.items()},
              "peers_minus_independent": paired(chosen["peers"], chosen["independent"]),
              "peers_minus_solos": {e: {k: paired(chosen["peers"], c) for k, c in views.items()}
                                    for e, views in solo.items()},
              "costs": costs, "audit_status": "manual_review_required",
              "note": "One fixed attempt per arm. External candidate scores are diagnostic only. "
                      "Operational fallback uses selection score, never external evaluation."}
    (folder / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(report(args.folder), indent=2))
