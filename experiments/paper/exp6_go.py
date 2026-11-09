"""Experiment 6: write a 1.5M-weight Go network from scratch. Write mode through the kernel, with the experiment's own pass
tools, ```games and ```positions, plugged in through `tools`. Everything else is the normal path.

The authors did not release their opponents, so only "random" works out of the box, and scores will not match the paper.
Give your own as `opponents={"weak": fn, "strong": fn}`, each mapping (inputs, legal mask) to an action.

    pip install pgx
    python exp6_go.py anthropic:<model> 10
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from _common import exact_prompt
from wsplab.passes import PassFailed, PassInvalid, PassResult

DEPTH, WIDTH, INPUTS, OUTPUTS = 6, 512, 243, 82
MAX_GAMES, MAX_POSITIONS = 128, 256


class Layer(nn.Module):
    def __init__(self, a: int, b: int):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(a, b))
        self.b = nn.Parameter(torch.zeros(b))

    def forward(self, x):
        return x @ self.W + self.b


class GoNet(nn.Module):
    """h_1 = relu(x @ L1.W + L1.b), through L6, then logits = h_6 @ out.W + out.b: the prompt's network and tensor names."""

    def __init__(self, depth: int = DEPTH, width: int = WIDTH):
        super().__init__()
        self.depth = depth
        for k in range(1, depth + 1):
            self.add_module(f"L{k}", Layer(INPUTS if k == 1 else width, width))
        self.add_module("out", Layer(width, OUTPUTS))

    def forward(self, x):
        for k in range(1, self.depth + 1):
            x = torch.relu(getattr(self, f"L{k}")(x))
        return getattr(self, "out")(x)


def encode(observation) -> np.ndarray:
    """pgx's observation as the prompt's 243 inputs: input (row * 9 + column) * 3 + plane."""
    o = np.asarray(observation)
    planes = np.stack([o[..., 0], o[..., 1], np.full((9, 9), o[0, 0, -1])], -1)
    return planes.reshape(-1).astype(np.float32)


def play(policy, opponent, games: int, seed: int = 0) -> list[dict]:
    """Games with the network as Black in half of them. Moves are sampled from the legal part of the softmax."""
    import jax
    import pgx

    env = pgx.make("go_9x9")
    step = jax.jit(env.step)
    rng = np.random.default_rng(seed)
    records = []
    for g in range(games):
        state = env.init(jax.random.PRNGKey(seed * 100_003 + g))
        network = g % 2
        moves = []
        while not (bool(state.terminated) or bool(state.truncated)):
            legal = np.asarray(state.legal_action_mask)
            x = encode(state.observation)
            if int(state.current_player) == network:
                logits = np.asarray(policy(x[None]), dtype=np.float64)[0]
                logits = np.where(legal, logits - logits[legal].max(), -np.inf)
                p = np.exp(logits)
                action = int(rng.choice(OUTPUTS, p=p / p.sum()))
            else:
                action = int(opponent(x, legal, rng))
            moves.append(action)
            state = step(state, action)
        reward = float(np.asarray(state.rewards)[network])
        records.append({"win": reward > 0, "network_player": network, "moves": moves})
    return records


def random_opponent(_x, legal, rng) -> int:
    return int(rng.choice(np.flatnonzero(legal)))


class GamesTool:
    """```games: up to 128 games against practice opponents."""

    fence = "games"

    def __init__(self, runner, work: Path, edited, opponents: dict | None = None):
        self.runner, self.work, self.edited = runner, Path(work), edited
        self.opponents = {"random": random_opponent, **(opponents or {})}

    def run(self, body: str, request_number: int) -> PassResult:
        try:
            games = json.loads(body)["games"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise PassInvalid(f"{type(e).__name__}: {e}") from e
        unknown = sorted(set(games) - set(self.opponents))
        if unknown:
            raise PassFailed(f"ValueError: this setup has no opponent {unknown}: the authors' were not released")
        if sum(games.values()) > MAX_GAMES or any(n < 0 or n % 2 for n in games.values()):
            raise PassInvalid(f"even counts, at most {MAX_GAMES} games in total")
        started = time.time()
        call = self.runner.call(self.edited())
        folder = self.work / f"pass_{request_number}"
        folder.mkdir(parents=True, exist_ok=True)
        summary: dict = {}
        saved = {}
        for name, n in games.items():
            records = play(call, self.opponents[name], n, seed=request_number)
            saved[name] = records
            wins = sum(r["win"] for r in records)
            summary[name] = {"games": n, "wins": wins, "score": wins / n if n else 0.0,
                             "game_length": float(np.mean([len(r["moves"]) for r in records])) if records else 0.0}
        (folder / "games.json").write_text(json.dumps(saved))
        (folder / "summary.json").write_text(json.dumps(summary, indent=1))
        summary["s"] = round(time.time() - started, 1)
        return PassResult("games", f"/work/{folder.name}/", summary)


class PositionsTool:
    """```positions: the network's top moves on up to 256 positions given as lists of moves from the empty board."""

    fence = "positions"

    def __init__(self, runner, work: Path, edited):
        self.runner, self.work, self.edited = runner, Path(work), edited

    def run(self, body: str, request_number: int) -> PassResult:
        import jax
        import pgx

        try:
            positions = json.loads(body)["positions"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise PassInvalid(f"{type(e).__name__}: {e}") from e
        if len(positions) > MAX_POSITIONS:
            raise PassInvalid(f"at most {MAX_POSITIONS} positions")
        env = pgx.make("go_9x9")
        step = jax.jit(env.step)
        rows, masks = [], []
        for p in positions:
            if "moves" not in p:
                raise PassInvalid('this example reads positions as {"moves": [...]} only')
            state = env.init(jax.random.PRNGKey(0))
            for move in p["moves"]:
                state = step(state, 81 if move == "pass" else int(move))
            rows.append(encode(state.observation))
            masks.append(np.asarray(state.legal_action_mask))
        x, legal = np.stack(rows), np.stack(masks)
        logits = np.asarray(self.runner.call(self.edited())(x))
        folder = self.work / f"pass_{request_number}"
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "inputs.npy", x)
        np.save(folder / "logits.npy", logits)
        np.save(folder / "legal.npy", legal)
        top = np.argsort(np.where(legal, logits, -np.inf), axis=1)[:, ::-1][:, :3]
        return PassResult("positions", f"/work/{folder.name}/", {"top_moves": top.tolist()})


PRELUDE = """
import numpy as np
SHAPES = {}
for _k in range(1, 7):
    SHAPES[f"L{_k}.W"] = (243 if _k == 1 else 512, 512)
    SHAPES[f"L{_k}.b"] = (512,)
SHAPES["out.W"], SHAPES["out.b"] = (512, 82), (82,)
PASS = 81
def inp(row, col, plane):
    return (row * 9 + col) * 3 + plane
def mv(row, col):
    return row * 9 + col
_load, _save, _names = load, save_edit, NAMES
NAMES = list(SHAPES)
def load(name):
    return _load(name).numpy()
def save(name, a):
    return _save(name, np.asarray(a, np.float32))
saved = edited
"""


def config(budget: int = 10, opponents: dict | None = None) -> dict:
    def tools(work, base, store, edited):
        from wsplab.runners.torch import TorchRunner

        runner = TorchRunner(GoNet)
        return {"games": GamesTool(runner, work, edited, opponents), "positions": PositionsTool(runner, work, edited)}

    return dict(build=GoNet, description="a Go policy", mode="write", budget=budget, max_runs=120, tools=tools, no_eval=True,
                prelude=PRELUDE, prompt=exact_prompt(f"exp6_go_from_scratch/runs/budget_{budget}/message.txt"))


if __name__ == "__main__":
    import wsplab

    result = wsplab.run(GoNet(), editor=sys.argv[1], **config(int(sys.argv[2])))
    print(result.verdict, result.patch)
