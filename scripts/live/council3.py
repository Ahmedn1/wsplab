"""The cross-provider council, with each provider chairing once."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path.home() / ".wsplab/live/ablation2"; ROOT.mkdir(parents=True, exist_ok=True)
M = ["anthropic:claude-opus-5-5", "gemini:gemini-4-argon", "openai:gpt-6.1-terra"]
def run(rep):
    members = M[rep:] + M[:rep]
    folder = ROOT / f"council3_{rep}"
    d = subprocess.run([sys.executable, "ablate_job.py", "council", str(rep), str(folder), ",".join(members)], capture_output=True, text=True, env=os.environ)
    lines = [l for l in d.stdout.splitlines() if l.startswith("{")]
    if lines:
        open(ROOT/"results.jsonl","a").write(lines[-1]+"\n"); r=json.loads(lines[-1]); print(rep,'chair',members[0],r["verdict"],round(r["final"] or 0,4),'bias',round(r["bias1"],2),'usd',r["usd"],'winners',r.get("winners"),flush=True)
    else: print("FAILED",rep,d.stderr[-700:],flush=True)
with ThreadPoolExecutor(3) as p: list(p.map(run, range(3)))
