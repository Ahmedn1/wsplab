"""Run the comparison, three at a time, stopping new jobs at the spending limit."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
LIMIT = 6.5
ROOT = Path.home() / ".wsplab/live/ablation"
ROOT.mkdir(parents=True, exist_ok=True)
ARMS = ["gpt-6-sol", "gpt-6.1-sol", "gpt-6.1-terra", "council", "weights"]
jobs = [(a, r) for r in range(3) for a in ARMS]
lock = threading.Lock(); total = [0.0]
def run(job):
    arm, rep = job
    with lock:
        if total[0] >= LIMIT:
            print("skipped", job, flush=True); return
    folder = ROOT / f"{arm}_{rep}"
    done = subprocess.run([sys.executable, "ablate_job.py", arm, str(rep), str(folder)], capture_output=True, text=True, env=os.environ)
    lines = [l for l in done.stdout.splitlines() if l.startswith("{")]
    with lock:
        if lines:
            row = json.loads(lines[-1]); total[0] += row["usd"]
            with open(ROOT / "results.jsonl", "a") as f: f.write(lines[-1] + "\n")
            print(job, row["verdict"], round(row["final"] or 0, 4), row["usd"], "total", round(total[0], 2), flush=True)
        else:
            print("FAILED", job, done.stderr[-900:], flush=True)
with ThreadPoolExecutor(3) as pool:
    list(pool.map(run, jobs))
print("done, total", round(total[0], 2))
