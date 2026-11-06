import json, os, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path.home() / ".wsplab/live/hard"; ROOT.mkdir(parents=True, exist_ok=True)
LIMIT = 12.0
total=[0.0]; lock=threading.Lock()
M = ["anthropic:claude-opus-5-5", "gemini:gemini-4-argon", "openai:gpt-6.1-terra"]
def rot(r): return ",".join(M[r:]+M[:r])
jobs=[]
for r in range(3):
    if r: jobs += [(M[0],r,[],f"claude_{r}"),(M[2],r,[],f"terra_{r}")]
    jobs.append((M[1],r,[],f"geminib_{r}"))
    jobs.append(("council",r,[rot(r)],f"council_{r}"))
    jobs.append(("council-noself",r,[rot(r),"--exclude-self"],f"noself_{r}"))
def run(job):
    arm, rep, extra, name = job
    with lock:
        if total[0] >= LIMIT:
            print("skipped", name, flush=True); return
    d = subprocess.run([sys.executable, "hard_job.py", arm, str(rep), str(ROOT / name), *extra], capture_output=True, text=True, env=os.environ)
    lines = [l for l in d.stdout.splitlines() if l.startswith("{")]
    with lock:
        if lines:
            open(ROOT/"results.jsonl","a").write(lines[-1]+"\n"); r=json.loads(lines[-1]); total[0]+=r["usd"]
            print(name, r["verdict"], 'final', round(r["final"] or 0,4), 'usd', r["usd"], 'passes', r["passes"], 'flags', r["fit_flags"], r["stop"], 'total', round(total[0],2), flush=True)
        else: print("FAILED", name, d.stderr[-500:], flush=True)
with ThreadPoolExecutor(3) as p: list(p.map(run, jobs))
print("done", round(total[0],2))
