"""Observe both paid trial processes; produce diagnostics only after both finish."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

if __package__:
    from .natural_report import report
else:
    from natural_report import report


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def watch(folder):
    processes = json.loads((folder / "processes.json").read_text())
    deadline = time.monotonic() + 12 * 3600
    while time.monotonic() < deadline:
        snapshot = {"at": time.time(), "arms": {}, "status": "running"}
        for arm, process in processes.items():
            try:
                os.kill(process["pid"], 0)
                alive = True
            except ProcessLookupError:
                alive = False
            p = folder / arm
            summary = json.loads((p / "summary.json").read_text()) if (p / "summary.json").exists() else None
            ledger = json.loads((p / "ledger.json").read_text()) if (p / "ledger.json").exists() else {}
            states = {}
            for state_path in sorted((p / "run").glob("run_*/session/state.json")):
                try:
                    state = json.loads(state_path.read_text())
                except json.JSONDecodeError:
                    continue  # Session writes are not atomic; retry on the next snapshot.
                states[state_path.parent.parent.name] = {k: state.get(k) for k in
                    ["turn", "runs", "passes", "done", "stop_reason"]}
            snapshot["arms"][arm] = {"pid": process["pid"], "alive": alive, "states": states,
                                     "spent": ledger.get("spent"), "summary_status": summary and summary["status"]}
        arms = list(snapshot["arms"].values())
        if all(a["summary_status"] == "completed" for a in arms):
            snapshot["status"] = "scoring_diagnostics"
            write(folder / "status.json", snapshot)
            try:
                result = report(folder)
                snapshot.update(status="complete_audit_pending", operational_scores=result["operational_scores"],
                                peers_minus_independent=result["peers_minus_independent"])
            except Exception as exc:
                snapshot.update(status="diagnostic_error", error_type=type(exc).__name__)
                write(folder / "status.json", snapshot)
                raise
        elif any(a["summary_status"] == "failed" or (not a["alive"] and not a["summary_status"]) for a in arms):
            snapshot["status"] = "failed_arm_needs_review"
        write(folder / "status.json", snapshot)
        with (folder / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(snapshot) + "\n")
        if snapshot["status"] != "running":
            return
        time.sleep(60)
    write(folder / "status.json", {"at": time.time(), "status": "watch_timeout", "note": "Paid jobs untouched."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    watch(args.folder.resolve())
