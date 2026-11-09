"""Decided in advance: a team run of >= 70% is a breakthrough, the claim needs 2 of 5 and a clean audit. Gemini alone is re-measured
at the same limit, because its earlier runs were cut short by the 15-run limit."""
import json, os, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path.home() / ".wsplab/live/team"; ROOT.mkdir(parents=True, exist_ok=True)
LIMIT = 12.0
TEAM = "anthropic:claude-opus-5-5,gemini:gemini-4-argon,openai:gpt-6.1-terra"
total=[0.0]; lock=threading.Lock()
jobs=[("team",s,[TEAM],f"team_{s}") for s in (1,2,3,4,5)]+[("gemini:gemini-4-argon",s,[],f"gemini_{s}") for s in (1,2,3)]
jobs=[jobs[0],jobs[5],jobs[1],jobs[6],jobs[2],jobs[7],jobs[3],jobs[4]]
def run(job):
    arm, seed, extra, name = job
    with lock:
        if total[0] >= LIMIT:
            print("skipped", name, flush=True); return
    env={**os.environ,"HARD_RUNS":"40","HARD_CAP":"3"}
    d = subprocess.run([sys.executable, "hard_job.py", arm, str(seed), str(ROOT / name), *extra], capture_output=True, text=True, env=env)
    lines = [l for l in d.stdout.splitlines() if l.startswith("{")]
    with lock:
        if lines:
            open(ROOT/"results.jsonl","a").write(lines[-1]+"\n"); r=json.loads(lines[-1]); total[0]+=r["usd"]
            print(name, r["verdict"], 'final', round(r["final"] or 0,4), 'usd', r["usd"], 'turns', r["turns"], 'runs', r["runs"], 'passes', r["passes"], 'flags', r["fit_flags"], r["stop"], 'total', round(total[0],2), flush=True)
        else: print("FAILED", name, d.stderr[-500:], flush=True)
with ThreadPoolExecutor(3) as p: list(p.map(run, jobs))
print("done", round(total[0],2))
