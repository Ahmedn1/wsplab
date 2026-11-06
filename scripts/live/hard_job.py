"""One run on the hard task: python hard_job.py <arm> <rep> <folder> [members] [--exclude-self]"""
import json, re, sys, time
from pathlib import Path
import wsplab
from wsplab.council import Council
from wsplab.editors import Metered
from wsplab.eval.suites import ArrayScorer
from wsplab.kernel import Limits
import common, hard_net

arm, rep, folder = sys.argv[1], int(sys.argv[2]), sys.argv[3]
members = sys.argv[4].split(",") if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else []
x, y, tr, pr, ev = hard_net.data()
net, clean, broken, faults = hard_net.build()
args = dict(build={"mlp": hard_net.SIZES, "activation": "relu"},
            description="an MLP classifying 8x8 grayscale digit images (64 inputs in [0, 1], row-major; two hidden layers of 48 ReLU units; 10 logits, one per digit 0-9)",
            note="This network used to be accurate, but it has been damaged in several places.",
            practice=ArrayScorer(x[pr], y[pr], name="digit images"), evaluate=ArrayScorer(x[ev], y[ev], name="held-out digit images", prefix="eval"),
            budget=3, max_runs=15, folder=folder, limits=Limits(seconds=120, memory_gb=4))
s = common.spending(2.5 if arm.startswith("council") else 1.2)
t0 = time.time(); extra = {}
if arm.startswith("council"):
    c = Council([common.editor(m) for m in members], s, method="borda", seed=rep, exclude_self="--exclude-self" in sys.argv)
    r = wsplab.run(net, editor=c, resume=True, **args)
    extra = {"chair": members[0], "rounds": len(c.rounds), "winners": [q.proposals[q.winner]["member"] for q in c.rounds],
             "ties": sum(q.tie for q in c.rounds)}
else:
    r = wsplab.run(net, editor=Metered(common.editor(arm), s), resume=True, **args)
code = "\n".join(json.loads(l).get("code", "") for l in open(Path(folder) / "session/tool_calls.jsonl") if '"python"' in l)
flags = sorted(set(re.findall(r"\b(lstsq|pinv|\.backward|optim\.|autograd|Ridge|LogisticRegression|MLPClassifier|\.fit\(|minimize|enable_grad|requires_grad|solve\()", code)))
print(json.dumps({"arm": arm, "rep": rep, "clean": clean, "broken": broken, "baseline": r.baseline, "final": r.final, "verdict": r.verdict,
                  "usd": round(s.usd, 4), "turns": r.state.turn, "passes": r.state.passes, "runs": r.state.runs, "stop": r.state.stop_reason,
                  "fit_flags": flags, "seconds": round(time.time() - t0), **extra}))
