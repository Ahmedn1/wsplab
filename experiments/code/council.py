"""A 3-member council on a harder fault: a trained digit classifier with a 3-way rotation of output rows and one biased class."""
import json, sys
import numpy as np, torch
from sklearn.datasets import load_digits
import wsplab
from wsplab import patch as P
from legacy_council import Council
from wsplab.eval.suites import ArrayScorer
from wsplab.kernel import Limits
from wsplab.runners.torch import mlp
import common

x, y = load_digits(return_X_y=True)
x = (x / 16).astype(np.float32); y = y.astype(np.int64)
order = np.random.default_rng(1).permutation(len(x)); x, y = x[order], y[order]
tr, pr, ev = slice(0, 900), slice(900, 1300), slice(1300, None)
torch.manual_seed(1)
net = mlp([64, 32, 10], "relu")
opt = torch.optim.Adam(net.parameters(), 1e-2)
for _ in range(400):
    opt.zero_grad(); torch.nn.functional.cross_entropy(net(torch.tensor(x[tr])), torch.tensor(y[tr])).backward(); opt.step()
clean = (net(torch.tensor(x[ev])).argmax(1).numpy() == y[ev]).mean()
with torch.no_grad():
    w, b = net.layers[2].weight, net.layers[2].bias
    w[[4, 7, 9]] = w[[7, 9, 4]].clone(); b[[4, 7, 9]] = b[[7, 9, 4]].clone()
    b[1] += 6.0
cap, folder = float(sys.argv[1]), sys.argv[2]
s = common.spending(cap)
members = [common.editor(m) for m in ("gpt-6-sol", "gpt-6.1-sol", "gpt-6.1-terra")]
council = Council(members, s, method="borda", seed=3)
r = wsplab.run(net, build={"mlp": [64, 32, 10], "activation": "relu"},
               description="an MLP classifying 8x8 grayscale digit images (64 inputs in [0, 1], then 32 ReLU units, then 10 logits, one per digit 0-9)",
               note="This model gets many digits wrong.", practice=ArrayScorer(x[pr], y[pr], name="digit images"),
               evaluate=ArrayScorer(x[ev], y[ev], name="held-out digit images", prefix="eval"),
               editor=council, budget=4, max_runs=30, folder=folder, limits=Limits(seconds=120, memory_gb=4))
rounds = [{"votes": [round(t, 1) for t in q.totals], "winner_member": q.proposals[q.winner]["member"], "tie": q.tie, "proposals": len(q.proposals)} for q in council.rounds]
print(json.dumps({"clean": float(clean), "verdict": r.verdict, "baseline": r.baseline, "final": r.final, "reasons": r.reasons,
                  "turns": r.state.turn, "passes": r.state.passes, "stop": r.state.stop_reason, "run_usd": round(s.usd, 4),
                  "rounds": rounds, "evidence": P.Patch.load(r.patch).meta.get("council")}, indent=1))
print(r.answer)
