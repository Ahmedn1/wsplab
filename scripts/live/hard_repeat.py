"""Does the 85.7% repeat? Same council, new seeds. Decided in advance: >= 70% is a breakthrough, and the claim needs 2 of 5."""
import json, os, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path.home() / ".wsplab/live/hard_repeat"; ROOT.mkdir(parents=True, exist_ok=True)
LIMIT = 10.0
MEMBERS = "gemini:gemini-4-argon,openai:gpt-6.1-terra,anthropic:claude-opus-5-5"
total=[0.0]; lock=threading.Lock()
def run(seed):
    with lock:
        if total[0] >= LIMIT:
            print("skipped", seed, flush=True); return
    d = subprocess.run([sys.executable, "hard_job.py", "council", str(seed), str(ROOT / f"seed_{seed}"), MEMBERS], capture_output=True, text=True, env=os.environ)
    lines = [l for l in d.stdout.splitlines() if l.startswith("{")]
    with lock:
        if lines:
            open(ROOT/"results.jsonl","a").write(lines[-1]+"\n"); r=json.loads(lines[-1]); total[0]+=r["usd"]
            print(seed, r["verdict"], 'final', round(r["final"] or 0,4), 'usd', r["usd"], 'runs', r["runs"], 'rounds', r.get("rounds"), [w.split(':')[0] for w in r.get("winners",[])], 'flags', r["fit_flags"], r["stop"], 'total', round(total[0],2), flush=True)
        else: print("FAILED", seed, d.stderr[-600:], flush=True)
with ThreadPoolExecutor(3) as p: list(p.map(run, [101,102,103,104,105]))
print("done", round(total[0],2))
