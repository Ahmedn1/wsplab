import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path.home() / ".wsplab/live/hard"; ROOT.mkdir(parents=True, exist_ok=True)
def run(job):
    arm, rep, extra, name = job
    d = subprocess.run([sys.executable, "hard_job.py", arm, str(rep), str(ROOT / name), *extra], capture_output=True, text=True, env=os.environ)
    lines = [l for l in d.stdout.splitlines() if l.startswith("{")]
    if lines:
        open(ROOT/"results.jsonl","a").write(lines[-1]+"\n"); r=json.loads(lines[-1])
        print(name, r["verdict"], 'base', round(r["baseline"],4), 'final', round(r["final"] or 0,4), 'clean', round(r["clean"],4), 'usd', r["usd"], 'turns', r["turns"], 'passes', r["passes"], 'flags', r["fit_flags"], r["stop"], flush=True)
    else: print("FAILED", name, d.stderr[-700:], flush=True)
jobs=[("anthropic:claude-opus-5-5",0,[],"claude_0"),("gemini:gemini-4-argon",0,[],"gemini_0"),("openai:gpt-6.1-terra",0,[],"terra_0")]
with ThreadPoolExecutor(3) as p: list(p.map(run, jobs))
