"""Read-only progress/report watcher for unpaid MGSM controls."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def read_rows(path):
    if not path.exists():
        return []
    content = path.read_bytes()
    # A concurrent append can stop in the middle of a UTF-8 character, not just
    # in the middle of JSON. Decode only complete newline-terminated records.
    lines = content.split(b"\n")[:-1]
    return [json.loads(line.decode("utf-8")) for line in lines if line]


def report(root, destination):
    result = {"at": time.time(), "paid_api_calls": 0, "is_council_result": False,
              "external_evaluation_used": False, "arms": {},
              "qualification": "manual_comprehension_review_still_required"}
    for arm, folder in [("cpu_feasibility", "controls-cpu-8"), ("gpu_full_pilot", "controls-gpu-32")]:
        base = root / folder
        metadata = read_json(base / "execution.json")
        status = read_json(base / "status.json") or {"status": "not_started"}
        data = {"status": status, "conditions": {}}
        result["arms"][arm] = data
        if not metadata:
            continue
        data["groups"] = metadata["groups"]
        data["max_output_tokens"] = metadata["max_output_tokens"]
        raw = {}
        for name in metadata["order"]:
            rows = read_rows(base / f"{name}.jsonl")
            raw[name] = rows
            values = {"responses": len(rows), "languages": {}, "difficulty_rule": "not_assessed"}
            for lang in ("en", "bn", "sw"):
                subset = [r for r in rows if r["language"] == lang]
                if not subset:
                    continue
                stat = {"n": len(subset), "truncated": sum(r["truncated"] for r in subset)}
                if name != "translate":
                    stat.update(correct=sum(r["strict_correct"] for r in subset),
                                accuracy=sum(r["strict_correct"] for r in subset) / len(subset))
                values["languages"][lang] = stat
            if arm == "gpu_full_pilot" and name != "translate" and len(rows) == 96:
                langs = values["languages"]
                if all(langs.get(l, {}).get("n") == 32 for l in ("en", "bn", "sw")):
                    target = (langs["bn"]["accuracy"] + langs["sw"]["accuracy"]) / 2
                    gap = langs["en"]["accuracy"] - target
                    truncation = sum(r["truncated"] for r in rows) / 96
                    values.update(target_accuracy=target, english_minus_target=gap,
                                  overall_truncation=truncation,
                                  difficulty_rule="passes_numeric_rule" if (
                                      langs["en"]["accuracy"] >= .5 and target <= .4 and
                                      gap >= .2 and truncation <= .1) else "fails_numeric_rule")
            data["conditions"][name] = values
        if arm == "gpu_full_pilot" and len(raw.get("nonthinking", [])) == 96:
            baseline = {(r["group"], r["language"]): r["strict_correct"] for r in raw["nonthinking"]}
            for name, rows in raw.items():
                if name == "translate" or len(rows) != 96:
                    continue
                pairs = {(r["group"], r["language"]): r["strict_correct"] for r in rows}
                if set(pairs) != set(baseline):
                    raise ValueError("Unmatched full-pilot groups")
                groups = sorted({g for g, _ in pairs})
                changes = np.array([np.mean([int(pairs[g, l]) - int(baseline[g, l])
                                             for l in ("bn", "sw")]) for g in groups])
                data["conditions"][name]["target_change_vs_nonthinking"] = {
                    "difference": float(changes.mean()),
                    "standard_error_clustered_by_problem": float(changes.std(ddof=1) / np.sqrt(len(groups))),
                    "independent_problem_groups": len(groups), "exploratory_not_multiplicity_adjusted": True}
    completed = all(a["status"]["status"] == "completed" for a in result["arms"].values())
    result["status"] = "controls_complete_manual_review_pending" if completed else "in_progress_or_incomplete"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# MGSM control results", "", f"Status: **{result['status']}**.", "",
             "This is the frozen pilot, not a council experiment or external test. No paid API calls.", "",
             "Difficulty-rule checks are screening criteria, not proof of editability. Translation/comprehension review remains necessary.", ""]
    for arm, data in result["arms"].items():
        lines += [f"## {arm}", "", f"Process status: `{data['status']['status']}`.", "",
                  "| Condition | English | Bengali | Swahili | Truncated / responses | Numeric difficulty rule |",
                  "|---|---:|---:|---:|---:|---|"]
        for name, values in data["conditions"].items():
            cells = []
            for lang in ("en", "bn", "sw"):
                stat = values["languages"].get(lang, {})
                cells.append(f"{stat['correct']}/{stat['n']}" if "correct" in stat else
                             f"{stat['n']} translations" if "n" in stat else "pending")
            truncated = sum(s["truncated"] for s in values["languages"].values())
            lines.append(f"| {name} | {' | '.join(cells)} | {truncated}/{values['responses']} | {values['difficulty_rule']} |")
        lines += [""]
    lines += ["CPU feasibility uses short outputs and FP32; full GPU controls use BF16 and the longer allowance.",
              "Compare controls within their backend. Translations are not assigned arithmetic scores.", "",
              "See [the frozen controls plan](natural-mgsm-controls-plan.md) and the adjacent JSON for per-condition diagnostics.", ""]
    destination.with_suffix(".md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--destination", type=Path, default=Path("experiments/06_mgsm/natural-mgsm-controls-results"))
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    deadline = time.monotonic() + 40 * 3600
    terminal = {"completed", "failed", "stopped_time_limit"}
    while True:
        data = report(args.root, args.destination)
        (args.root / "controls-monitor.json").write_text(json.dumps(
            {"pid": os.getpid(), "at": time.time(), "status": data["status"]}, indent=2) + "\n")
        if not args.watch or all(a["status"]["status"] in terminal for a in data["arms"].values()):
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(30)
