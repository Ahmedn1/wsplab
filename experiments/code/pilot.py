"""A really trained digit classifier with two output rows swapped. The editor is told only that it confuses some digits."""
import json, os, sys
import numpy as np, torch
from sklearn.datasets import load_digits
import wsplab
from wsplab.editors import Metered
from wsplab.eval.suites import ArrayScorer
from wsplab.kernel import Limits
from wsplab.runners.torch import mlp
import common

x, y = load_digits(return_X_y=True)
x = (x / 16).astype(np.float32); y = y.astype(np.int64)
order = np.random.default_rng(0).permutation(len(x)); x, y = x[order], y[order]
tr, pr, ev = slice(0, 900), slice(900, 1300), slice(1300, None)
torch.manual_seed(0)
net = mlp([64, 32, 10], "relu")
opt = torch.optim.Adam(net.parameters(), 1e-2)
for _ in range(400):
    opt.zero_grad(); torch.nn.functional.cross_entropy(net(torch.tensor(x[tr])), torch.tensor(y[tr])).backward(); opt.step()
clean = (net(torch.tensor(x[ev])).argmax(1).numpy() == y[ev]).mean()
with torch.no_grad():
    w, b = net.layers[2].weight, net.layers[2].bias
    w[[3, 8]] = w[[8, 3]].clone(); b[[3, 8]] = b[[8, 3]].clone()
model = sys.argv[1]; cap = float(sys.argv[2]); folder = sys.argv[3]
s = common.spending(cap)
r = wsplab.run(net, build={"mlp": [64, 32, 10], "activation": "relu"},
               description="an MLP classifying 8x8 grayscale digit images (64 inputs in [0, 1], then 32 ReLU units, then 10 logits, one per digit 0-9)",
               note="This model confuses some digits.", practice=ArrayScorer(x[pr], y[pr], name="digit images"),
               evaluate=ArrayScorer(x[ev], y[ev], name="held-out digit images", prefix="eval"),
               editor=Metered(common.editor(model), s), budget=3, max_runs=30, folder=folder, limits=Limits(seconds=120, memory_gb=4))
print(json.dumps({"clean": float(clean), "verdict": r.verdict, "baseline": r.baseline, "final": r.final, "reasons": r.reasons,
                  "turns": r.state.turn, "passes": r.state.passes, "runs": r.state.runs, "stop": r.state.stop_reason, "run_usd": round(s.usd, 4), "tokens": s.tokens}, indent=1))
print(r.answer)
