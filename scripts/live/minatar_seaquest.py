"""MinAtar Seaquest: improve the policy network GPT-6 Astra wrote in "Sparks of Weight-Space Programming" (Experiment 5).

The network (600-32-32-6 tanh, `networks/seaquest/hand.npz` in RikiyaT/lm_understand-logs @ 4d09ba6d) scored 53.1 per
episode in the paper; PPO with the same architecture reached 88.9 after 500M steps, so the architecture has room.

Scoring follows the paper: MinAtar 1.0.15 Seaquest, the paper's 6-channel input, actions sampled from the softmax of the
logits, sticky actions 0.1 (MinAtar's own wrapper), episodes cut at 10,000 steps. MinAtar is GPL-3.0: it is a dependency
of this experiment that you install yourself (`pip install minatar==1.0.15 --no-deps`), never shipped with wsplab.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wsplab.eval.suites import Scores
from wsplab.eval.stats import paired
from wsplab.passes.base import PassFailed, PassInvalid, PassResult
from wsplab.runners.torch import TorchRunner, state_to_checkpoint

DATA = Path(os.path.expanduser("~/.wsplab/data/minatar_paper"))
HAND_SHA256 = None  # filled in by prepare() into the manifest
NAMES = {"W1": "actor/dense1/kernel", "b1": "actor/dense1/bias", "W2": "actor/dense2/kernel", "b2": "actor/dense2/bias",
         "W3": "actor/logits/kernel", "b3": "actor/logits/bias"}
MAX_STEPS = 10_000
STICKY = 0.1
PRACTICE_SEEDS = range(0, 2000)            # the editors' work half and the selection half come from these
EVAL_SEEDS = range(1_000_000, 1_002_000)   # never shown to an editor
MIN_EPISODES, MAX_EPISODES, RECORD = 100, 1000, 4
END = {"timeout": 0, "hit": 1, "oxygen": 2, "surfaced_empty": 3}
# The paper's channel mapping: (its channel) <- (MinAtar's channels)
MAPPING = [("sub_front", "sub_back"), ("enemy_fish", "enemy_sub", "enemy_bullet"), ("enemy_fish", "enemy_sub", "diver"),
           ("friendly_bullet",), ("trail", "sub_back"), ("oxygen_guage", "diver_guage")]


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class Policy(nn.Module):
    """h1 = tanh(x @ W1 + b1); h2 = tanh(h1 @ W2 + b2); logits = h2 @ W3 + b3. The paper's names and orientation."""

    def __init__(self):
        super().__init__()
        self.W1, self.b1 = nn.Parameter(torch.zeros(600, 32)), nn.Parameter(torch.zeros(32))
        self.W2, self.b2 = nn.Parameter(torch.zeros(32, 32)), nn.Parameter(torch.zeros(32))
        self.W3, self.b3 = nn.Parameter(torch.zeros(32, 6)), nn.Parameter(torch.zeros(6))
        self.h1, self.h2, self.logits = nn.Tanh(), nn.Tanh(), nn.Identity()

    def forward(self, x):
        x = x.float()
        h = self.h1(x @ self.W1 + self.b1)
        h = self.h2(h @ self.W2 + self.b2)
        return self.logits(h @ self.W3 + self.b3)


def build():
    return Policy()


def observe(env) -> np.ndarray:
    """The paper's input: 10x10 grid, 6 channels, index (row*10+col)*6+channel."""
    state = env.state()
    ch = env.env.channels
    out = np.zeros((10, 10, 6), dtype=np.uint8)
    for k, sources in enumerate(MAPPING):
        out[:, :, k] = np.any(np.stack([state[:, :, ch[s]] for s in sources]), axis=0)
    return out.reshape(600)


def play(call, seeds, record=0, hooks=None):
    """Plays one episode per seed in lockstep, batching the network over live episodes. The game's randomness and the
    action sampling are both seeded by the episode's seed, so two networks meet the same start.
    Returns per-episode summaries and, for the first `record` episodes, step records."""
    from minatar import Environment

    envs, rngs = [], []
    for s in seeds:
        e = Environment("seaquest", sticky_action_prob=STICKY)
        e.seed(int(s))
        e.reset()
        envs.append(e)
        rngs.append(np.random.default_rng([int(s), 7]))
    n = len(envs)
    ret = np.zeros(n)
    length = np.zeros(n, dtype=np.int64)
    end = np.zeros(n, dtype=np.int64)
    surfaced = np.zeros(n, dtype=np.int64)
    max_divers = np.zeros(n, dtype=np.int64)
    actions = np.zeros((n, 6), dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    steps = {i: {"obs": [], "action": [], "reward": [], "logits": []} for i in range(min(record, n))}
    while alive.any():
        idx = np.flatnonzero(alive)
        obs = np.stack([observe(envs[i]) for i in idx])
        logits = np.asarray(call(obs), dtype=np.float64)
        z = logits - logits.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        for j, i in enumerate(idx):
            a = int(rngs[i].choice(6, p=p[j]))
            oxygen = envs[i].env.oxygen
            r, term = envs[i].act(a)
            ret[i] += r
            length[i] += 1
            actions[i, a] += 1
            surfaced[i] += envs[i].env.oxygen > oxygen  # surfacing with divers refills the oxygen
            max_divers[i] = max(max_divers[i], envs[i].env.diver_count)
            if i in steps:
                steps[i]["obs"].append(obs[j])
                steps[i]["action"].append(a)
                steps[i]["reward"].append(r)
                steps[i]["logits"].append(logits[j])
            if term:
                g = envs[i].env
                end[i] = END["oxygen"] if g.oxygen <= 0 else (
                    END["surfaced_empty"] if g.sub_y == 0 and g.diver_count == 0 else END["hit"])
                alive[i] = False
            elif length[i] >= MAX_STEPS:
                end[i] = END["timeout"]
                alive[i] = False
    summary = np.column_stack([ret, length, end, surfaced, max_divers, actions]).astype(np.float64)
    return summary, steps


class SeaquestScorer:
    """Items are episodes, `episode_<seed>`; an item's score is its return."""

    def __init__(self, seeds, name="Seaquest episodes"):
        self.seeds = [int(s) for s in seeds]
        self.name = name

    def ids(self):
        return [f"episode_{s}" for s in self.seeds]

    @staticmethod
    def seeds_of(ids):
        return [int(i.rsplit("_", 1)[1]) for i in ids]

    def summaries(self, call, ids, record=0):
        return play(call, self.seeds_of(ids), record)

    def score(self, call, ids):
        summary, _ = self.summaries(call, ids)
        return Scores({i: [float(v)] for i, v in zip(ids, summary[:, 0])}, {"metric": "episode_return"})

    def describe(self):
        return (f"Practice episodes: {len(self.seeds)} seeds, ids episode_<seed>, listed in /practice/seeds.json. "
                "An item's score is the episode's return. None of them is in the evaluation.")

    def export(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "seeds.json").write_text(json.dumps(self.seeds))

    def fingerprints(self):
        return {str(s).encode() for s in self.seeds}

    def split(self, fraction, seed=0):
        order = np.random.default_rng(seed).permutation(len(self.seeds))
        cut = int(len(order) * fraction)
        return (SeaquestScorer([self.seeds[i] for i in sorted(order[:cut])], self.name),
                SeaquestScorer([self.seeds[i] for i in sorted(order[cut:])], self.name))


SUMMARY_COLUMNS = ["return", "length", "end (0 timeout, 1 hit, 2 oxygen ran out, 3 surfaced with no diver)",
                   "times surfaced with divers", "most divers carried", "count of action 0 no-op", "count of action 1 left",
                   "count of action 2 up", "count of action 3 right", "count of action 4 down", "count of action 5 fire"]


class SeaquestForward:
    """The ```forward tool for episodes. Same request as the generic tool: {"model", "items", "activations"}.
    Writes scores.json (episode returns), outputs.npy (one summary row per episode, columns in SUMMARY_COLUMNS) and,
    for the first RECORD episodes, steps_<k>.npz with each step's input, action, reward and logits, plus
    act_<module>.npy per requested module (rows = those steps, in order)."""

    fence = "forward"

    def __init__(self, runner, scorer, *, work, base, edited, edit_digest):
        self.runner, self.scorer = runner, scorer
        self.work, self.base, self.edited, self.edit_digest = Path(work), Path(base), edited, edit_digest
        self._seen = set()

    def _request(self, body):
        try:
            req = json.loads(body)
        except json.JSONDecodeError as e:
            raise PassInvalid(f"JSONDecodeError: {e}") from e
        if not isinstance(req, dict) or req.get("model") not in ("original", "edited", "both"):
            raise PassInvalid('"model" must be "original", "edited" or "both"')
        known = set(self.scorer.ids())
        ids = req.get("items") or self.scorer.ids()
        unknown = [i for i in ids if i not in known]
        if unknown:
            raise PassInvalid(f"unknown episode ids: {unknown[:5]}")
        if not MIN_EPISODES <= len(ids) <= MAX_EPISODES:
            raise PassInvalid(f"a pass plays between {MIN_EPISODES} and {MAX_EPISODES} episodes; asked for {len(ids)}")
        names = req.get("activations") or []
        modules = {"h1", "h2", "logits"}
        if set(names) - modules:
            raise PassInvalid(f"activations can be recorded for {sorted(modules)}")
        key = hashlib.sha1(f"{json.dumps(req, sort_keys=True)}|{self.edit_digest()}".encode()).hexdigest()
        if key in self._seen:
            raise PassInvalid("this request is identical to an earlier one with unchanged edits")
        self._seen.add(key)
        return req, list(ids), names

    def _run(self, model, ids, names, folder):
        weights = self.base if model == "original" else self.edited()
        folder.mkdir(parents=True, exist_ok=True)
        try:
            call = self.runner.call(weights)
            summary, steps = self.scorer.summaries(call, ids, RECORD)
            for k, rec in steps.items():
                np.savez(folder / f"steps_{k}.npz", episode=ids[k], obs=np.array(rec["obs"], dtype=np.uint8),
                         action=np.array(rec["action"]), reward=np.array(rec["reward"]),
                         logits=np.array(rec["logits"], dtype=np.float32))
            if names and steps:
                obs = np.concatenate([np.array(steps[k]["obs"]) for k in sorted(steps)])
                got = self.runner.capture(weights, obs, names)
                for name, arr in got.items():
                    np.save(folder / f"act_{name}.npy", arr)
        except Exception as e:  # noqa: BLE001 - a failed run must not cost a pass
            raise PassFailed(f"{type(e).__name__}: {e}") from e
        np.save(folder / "outputs.npy", summary)
        scores = {i: [float(v)] for i, v in zip(ids, summary[:, 0])}
        (folder / "scores.json").write_text(json.dumps(scores))
        (folder / "columns.json").write_text(json.dumps(SUMMARY_COLUMNS))
        return summary, scores

    def run(self, body, request_number):
        req, ids, names = self._request(body)
        started = time.time()
        k = request_number

        def describe(summary):
            ends = np.bincount(summary[:, 2].astype(int), minlength=4)
            return {"mean_return": round(float(summary[:, 0].mean()), 3),
                    "mean_length": round(float(summary[:, 1].mean()), 1),
                    "ends": {name: int(ends[c]) for name, c in END.items()}}

        if req["model"] != "both":
            folder = self.work / f"fwd_{k}"
            summary, _ = self._run(req["model"], ids, names, folder)
            out = {"ok": True, "model": req["model"], "episodes": len(ids), **describe(summary)}
            where = f"/work/{folder.name}/"
        else:
            a, sa = self._run("original", ids, names, self.work / f"fwd_{k}_original")
            b, sb = self._run("edited", ids, names, self.work / f"fwd_{k}_edited")
            p = paired([sa[i] for i in ids], [sb[i] for i in ids])
            out = {"ok": True, "model": "both", "episodes": len(ids), "original": describe(a), "edited": describe(b),
                   "paired": {"return_difference": round(p.difference, 4), "standard_error": round(p.standard_error, 4),
                              "episodes_better": p.wrong_to_right, "episodes_worse": p.right_to_wrong}}
            where = f"/work/fwd_{k}_original/ and /work/fwd_{k}_edited/"
        out["s"] = round(time.time() - started, 1)
        return PassResult("run", where, out)


def tools(work, base, store, edited, scorer, runner):
    return {"forward": SeaquestForward(runner, scorer, work=work, base=base, edited=edited, edit_digest=store.digest)}


def prepare(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    hand = DATA / "networks/seaquest/hand.npz"
    z = np.load(hand, allow_pickle=False)
    state = {k: torch.from_numpy(np.ascontiguousarray(z[v])) for k, v in NAMES.items()}
    net = build()
    net.load_state_dict(state, strict=True)
    state_to_checkpoint(state, folder / "model.safetensors")
    manifest = {"source": "RikiyaT/lm_understand-logs@4d09ba6d exp5_write_from_scratch/networks/seaquest/hand.npz",
                "hand_sha256": digest(hand), "model_sha256": digest(folder / "model.safetensors"),
                "practice_seeds": [PRACTICE_SEEDS.start, PRACTICE_SEEDS.stop],
                "eval_seeds": [EVAL_SEEDS.start, EVAL_SEEDS.stop], "minatar": "1.0.15", "sticky": STICKY,
                "max_steps": MAX_STEPS}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(folder, episodes=1000):
    """Reproduce the paper's 53.1 (1,000 episodes; their seeds are unknown, so agreement is within sampling error)."""
    folder = Path(folder)
    runner = TorchRunner(build, "cpu")
    t = time.time()
    summary, _ = play(runner.call(folder / "model.safetensors"), range(5_000_000, 5_000_000 + episodes))
    paper = json.loads((DATA / "eval/minatar_eval.json").read_text())["seaquest"]["hand"]
    out = {"episodes": episodes, "mean": float(summary[:, 0].mean()),
           "se": float(summary[:, 0].std(ddof=1) / np.sqrt(episodes)), "paper": paper,
           "mean_length": float(summary[:, 1].mean()),
           "ends": np.bincount(summary[:, 2].astype(int), minlength=4).tolist(), "seconds": round(time.time() - t, 1)}
    (folder / "baseline.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def description():
    """The game, its source, the input and output exactly as in Astra's prompt, then this task's network and scoring."""
    lines = (DATA / "runs/seaquest/message.txt").read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("THE GAME"))
    stop = next(i for i, ln in enumerate(lines) if ln.startswith("THE NETWORK"))
    game = "\n".join(lines[start:stop]).strip()
    return (
        "a policy network for MinAtar Seaquest. It was written by hand, without training, by another model in a "
        "single reply; your job is to make it score higher by editing its weights.\n\n" + game + "\n\n"
        "THE NETWORK: a multilayer perceptron with 600 inputs, two hidden layers of 32 units with tanh, and 6 outputs "
        "(20,486 parameters): h1 = tanh(x @ W1 + b1), W1 600x32; h2 = tanh(h1 @ W2 + b2), W2 32x32; logits = h2 @ W3 + "
        "b3, W3 32x6. Tensor names: W1, b1, W2, b2, W3, b3. Row i of W1 holds the weights from input i; column j of a "
        "weight matrix belongs to unit j of the next layer. Modules for activations: h1, h2, logits.\n\n"
        "SCORING: the mean return per episode over fresh episodes never shown to you, each action sampled from the "
        "softmax of the logits, sticky actions 0.1, episodes cut at 10,000 steps. The same architecture trained by "
        "reinforcement learning reaches much higher scores, so there is room to improve.\n\n"
        "THE FORWARD TOOL FOR THIS NETWORK: practice items are episodes (ids episode_<seed>). A pass plays "
        f"{MIN_EPISODES} to {MAX_EPISODES} episodes (default: all practice episodes). Besides scores.json, outputs.npy "
        "has one row per episode with columns " + json.dumps(SUMMARY_COLUMNS) + " (also in columns.json). For the first "
        f"{RECORD} episodes of a request, steps_<k>.npz holds every step's input (600 numbers), action, reward and "
        "logits, and act_<module>.npy holds the requested modules' activations over those steps, in order.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.folder) if args.verify else prepare(args.folder), indent=2))
