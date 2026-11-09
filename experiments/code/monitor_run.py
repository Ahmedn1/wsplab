"""Read-only watcher for an existing run; never restarts it or makes API calls."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None  # Writers may be between truncate and write.


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except (OSError, IndexError):
        return None


def snapshot(folder, pid, identity, stale_seconds=3600):
    state = read_json(folder / "session/state.json")
    report = read_json(folder / "report.json")
    alive = identity is not None and process_identity(pid) == identity
    files = [folder / "session/state.json", folder / "session/tool_calls.jsonl"]
    files += list((folder / "work").glob("gen_*.jsonl"))
    existing = [p for p in files if p.exists()]
    latest = max(existing, key=lambda p: p.stat().st_mtime) if existing else None
    age = time.time() - latest.stat().st_mtime if latest else None
    if report and "verdict" in report:
        status = "finished"
    elif not alive:
        status = "process_exited_without_report"
    elif state and state.get("done"):
        status = "final_evaluation"
    elif age is not None and age > stale_seconds:
        status = "needs_inspection"  # A long generation/API call is not proof of a hang.
    else:
        status = "running"
    return {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": pid, "process_alive": alive, "status": status, "state": state,
            "last_activity": str(latest) if latest else None,
            "activity_age_seconds": round(age) if age is not None else None,
            "report": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--hours", type=float, default=48)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0 or args.hours <= 0:
        parser.error("interval and hours must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    identity = process_identity(args.pid)
    deadline = time.monotonic() + args.hours * 3600
    while True:
        result = snapshot(args.folder, args.pid, identity)
        if time.monotonic() >= deadline:
            result["status"] = "monitor_expired"
        temp = args.out / f"status.{os.getpid()}.tmp"
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(args.out / "status.json")
        with (args.out / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        if args.once or result["status"] in {"finished", "process_exited_without_report", "monitor_expired"}:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
