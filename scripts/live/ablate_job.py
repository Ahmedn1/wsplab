"""One run of the council comparison: python ablate_job.py <arm> <repeat> <folder>"""
import json, sys, time
import numpy as np, torch
from sklearn.datasets import load_digits
import wsplab
from wsplab.council import Council
from wsplab.editors import Metered
from wsplab.eval.suites import ArrayScorer
from wsplab.io import Checkpoint
from wsplab.kernel import Limits
from wsplab.runners.torch import mlp
import common

MODELS = ("gpt-6-sol", "gpt-6.1-sol", "gpt-6.1-terra")
CAPS = {"council": 1.2, "weights": 1.6}
arm, rep, folder = sys.argv[1], int(sys.argv[2]), sys.argv[3]

x, y = load_digits(return_X_y=True)
x = (x / 16).astype(np.float32); y = y.astype(np.int64)
order = np.random.default_rng(1).permutation(len(x)); x, y = x[order], y[order]
tr, pr, ev = slice(0, 900), slice(900, 1300), slice(1300, None)
torch.manual_seed(1)
net = mlp([64, 32, 10], "relu")
opt = torch.optim.Adam(net.parameters(), 1e-2)
for _ in range(400):
    opt.zero_grad(); torch.nn.functional.cross_entropy(net(torch.tensor(x[tr])), torch.tensor(y[tr])).backward(); opt.step()
clean_acc = float((net(torch.tensor(x[ev])).argmax(1).numpy() == y[ev]).mean())
clean = {k: v.clone().numpy() for k, v in net.state_dict().items()}
with torch.no_grad():
    w, b = net.layers[2].weight, net.layers[2].bias
    w[[4, 7, 9]] = w[[7, 9, 4]].clone(); b[[4, 7, 9]] = b[[7, 9, 4]].clone()
    b[1] += 6.0

common_args = dict(build={"mlp": [64, 32, 10], "activation": "relu"},
                   description="an MLP classifying 8x8 grayscale digit images (64 inputs in [0, 1], then 32 ReLU units, then 10 logits, one per digit 0-9)",
                   note="This model gets many digits wrong.", practice=ArrayScorer(x[pr], y[pr], name="digit images"),
                   evaluate=ArrayScorer(x[ev], y[ev], name="held-out digit images", prefix="eval"), budget=4, max_runs=30,
                   folder=folder, limits=Limits(seconds=120, memory_gb=4))
s = common.spending(CAPS.get(arm, 0.8))
t0 = time.time()
extra = {}
if arm == "council":
    council = Council([common.editor(m) for m in MODELS], s, method="borda", seed=rep)
    r = wsplab.run(net, editor=council, **common_args)
    chair_won = [q.proposals[q.winner]["member"] == "openai:gpt-6-sol" for q in council.rounds]
    bias_proposed = [any("b[1]" in p["text"] or "[1]" in p["text"] and "bias" in p["text"] for p in q.proposals) for q in council.rounds]
    extra = {"rounds": len(council.rounds), "chair_won": chair_won, "ties": sum(q.tie for q in council.rounds), "bias_mentioned_in_proposal": bias_proposed}
elif arm == "weights":
    out = wsplab.search(net, editors=[Metered(common.editor(m), s) for m in MODELS], selection="vote", spending=s, seed=rep, **common_args)
    r = out.result
    extra = {"success_rate": out.success_rate, "winner_run": out.selection.get("run"), "tie": out.selection.get("tie"),
             "candidates": [{"editor": c["editor"], "score": c["score"], "gain": c["gain"]} for c in out.candidates]}
else:
    r = wsplab.run(net, editor=Metered(common.editor(arm), s), **common_args)
e = Checkpoint(r.weights)
bias1, rows = float(e.float("layers.2.bias")[1]), e.float("layers.2.weight")
row_error = float(np.abs(rows[[4, 7, 9]] - clean["layers.2.weight"][[4, 7, 9]]).max())
row = {"arm": arm, "rep": rep, "clean": clean_acc, "baseline": r.baseline, "final": r.final, "verdict": r.verdict, "usd": round(s.usd, 4),
       "tokens": s.tokens, "turns": getattr(r.state, "turn", None), "bias1": bias1, "bias_fixed": abs(bias1 - float(clean["layers.2.bias"][1])) < 1.0,
       "rotation_fixed": row_error < 0.05, "seconds": round(time.time() - t0), **extra}
print(json.dumps(row))
