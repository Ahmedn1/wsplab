import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path.home() / ".wsplab/live/ablation2"; ROOT.mkdir(parents=True, exist_ok=True)
arm = sys.argv[1]
def run(rep):
    folder = ROOT / f"{arm.replace(':','_')}_{rep}"
    d = subprocess.run([sys.executable, "ablate_job.py", arm, str(rep), str(folder)], capture_output=True, text=True, env=os.environ)
    lines = [l for l in d.stdout.splitlines() if l.startswith("{")]
    if lines:
        open(ROOT/"results.jsonl","a").write(lines[-1]+"\n"); r=json.loads(lines[-1]); print(rep,r["verdict"],round(r["final"] or 0,4),'bias',round(r["bias1"],2),'rot',r["rotation_fixed"],'usd',r["usd"],'turns',r["turns"],flush=True)
    else: print("FAILED",rep,d.stderr[-600:],flush=True)
with ThreadPoolExecutor(3) as p: list(p.map(run, range(3)))
