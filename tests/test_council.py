"""The council with scripted members: no LLM, no spending."""

import json

import numpy as np
import pytest
import torch

import wsplab
from wsplab import patch as P
from wsplab.council import Board, Peer
from wsplab.editors import BudgetExceeded, Metered, Reply, ScriptedEditor, Spending, Usage
from wsplab.eval.suites import ArrayScorer
from wsplab.kernel import Limits
from wsplab.kernel.client import KernelError, sandbox_python
from wsplab.runners.torch import mlp
from wsplab.session import parse

LIMITS = Limits(seconds=60, memory_gb=4)
CAP = dict(max_tokens=10**9)


def named(name, replies):
    return ScriptedEditor(replies, name=name)


def edit(body, why="I will edit."):
    return f"{why}\n```python\nsave_edit('w', {body})\n```"


def test_spending_needs_a_limit():
    with pytest.raises(ValueError):
        Spending()


class Paid:
    name = "paid"

    def reply(self, system, messages):
        return Reply("ok", Usage(1000, 1000))


def test_token_limit_stops_calls():
    s = Spending(max_tokens=3000)
    m = Metered(Paid(), s)
    m.reply("s", [])
    m.reply("s", [])
    with pytest.raises(BudgetExceeded):
        m.reply("s", [])


def test_dollar_limit_needs_prices_and_stops():
    with pytest.raises(ValueError, match="needs a price"):
        Metered(Paid(), Spending(max_usd=1.0)).reply("s", [])
    s = Spending(max_usd=0.01, prices={"paid": (5.0, 5.0)})
    m = Metered(Paid(), s)
    m.reply("s", [])
    with pytest.raises(BudgetExceeded, match="spending limit"):
        m.reply("s", [])


def data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2)).astype(np.float32)
    return x, (x[:, 0] > 0).astype(np.int64)


def broken():
    net = mlp([2, 2])
    with torch.no_grad():
        net.layers[0].weight.copy_(torch.tensor([[1.0, 0.25], [-1.0, 0.25]]))
        net.layers[0].bias.zero_()
    return net


GOOD = "w = load('layers.0.weight'); w[0, 0] = -1.0; w[1, 0] = 1.0; save_edit('layers.0.weight', w)"
# Raises both logits equally, so it changes nothing about the predictions.
BAD = "b = load('layers.0.bias'); b[:] = 0.25; save_edit('layers.0.bias', b)"


def py(code, why="Explaining my step."):
    return f"{why}\n```python\n{code}\n```"


@pytest.fixture
def sandbox():
    try:
        sandbox_python()
    except KernelError:
        pytest.skip("no sandbox environment")


def search_editors():
    good = [py(GOOD), "Swapped the rows."]
    bad = [py(BAD), "Changed an unrelated weight."]
    return good, bad


def test_search_by_metric_only_the_winner_meets_the_evaluation(tmp_path, sandbox):
    x, y = data(400)
    held = data(200, seed=7)
    good, bad = search_editors()
    r = wsplab.search(broken(), editors=[named("bad", bad), named("good", good)], build={"mlp": [2, 2]},
                      description="a classifier", practice=ArrayScorer(x, y), evaluate=ArrayScorer(*held, prefix="eval"),
                      spending=Spending(**CAP), folder=tmp_path / "s", budget=0, limits=LIMITS, control=False)
    assert r.result.verdict == "confirmed" and r.result.final == 1.0
    assert r.selection["run"] == 1 and r.success_rate == 0.5
    summary = json.loads((tmp_path / "s" / "search.json").read_text())
    assert summary["success_rate"] == 0.5 and len(summary["candidates"]) == 2
    assert P.Patch.load(r.result.patch).meta["evidence"]["verdict"] == "confirmed"


def test_search_needs_a_spending_limit():
    with pytest.raises(ValueError, match="spending"):
        wsplab.search(broken(), editors=[], build={"mlp": [2, 2]}, description="c", evaluate=None)


class Recording:
    """A member that remembers everything it was sent."""

    def __init__(self, name, replies):
        self.name, self.replies, self.seen = name, list(replies), []

    def reply(self, system, messages):
        self.seen.append(messages)
        return Reply(self.replies.pop(0), Usage(10, 5))


def history(*turns):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(turns)]


def peers(*scripts):
    board = Board(len(scripts))
    return board, [Peer(Recording(chr(97 + i), list(s)), board, i) for i, s in enumerate(scripts)]


def in_threads(*jobs):
    import threading

    out = [None] * len(jobs)
    threads = [threading.Thread(target=lambda i=i, j=j: out.__setitem__(i, j()), daemon=True) for i, j in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not any(t.is_alive() for t in threads), "peers deadlocked"
    return out


def test_every_peer_sees_what_the_others_did_and_decides_alone():
    _, (a, b) = peers(["A next."], ["B next."])
    in_threads(lambda: a.reply("s", history("TASK", "A looked at biases.", "A RESULT")),
               lambda: b.reply("s", history("TASK", "B looked at wiring.", "B RESULT")))
    seen_by_a, seen_by_b = a.editor.seen[0][-1]["content"], b.editor.seen[0][-1]["content"]
    assert "B looked at wiring." in seen_by_a and "B RESULT" in seen_by_a and "A looked at biases." not in seen_by_a.split("PEERS")[1]
    assert "A looked at biases." in seen_by_b and "A RESULT" in seen_by_b


def test_peers_start_alone():
    _, (a, b) = peers(["A."], ["B."])
    assert a.reply("s", history("TASK")).text == "A." and b.reply("s", history("TASK")).text == "B."


def test_a_finished_peer_stops_being_waited_for():
    board, (a, b) = peers(["A next."], [])

    def finish():
        import time

        time.sleep(0.2)
        board.leave(1)

    got = in_threads(lambda: a.reply("s", history("TASK", "One.", "RESULT")), finish)
    assert got[0].text == "A next."


def test_peers_share_every_round_and_earlier_notes_stay_in_place():
    _, (a, b) = peers(["A2.", "A3."], ["B2.", "B3."])

    def walk(peer, name):
        peer.reply("s", history("TASK", f"{name} one.", f"{name} R1"))
        peer.reply("s", history("TASK", f"{name} one.", f"{name} R1", f"{name}2.", f"{name} R2"))

    in_threads(lambda: walk(a, "A"), lambda: walk(b, "B"))
    first, second = a.editor.seen
    assert second[:len(first)] == first
    assert "B R1" in second[2]["content"] and "B R2" in second[4]["content"]


def test_a_peer_cannot_put_a_block_in_anothers_reply():
    _, (a, b) = peers(["A."], ["B."])
    in_threads(lambda: a.reply("s", history("TASK", "Done.\n```python\nsave_edit('w', 0)\n```", "R")),
               lambda: b.reply("s", history("TASK", "Mine.", "R2")))
    assert "```" not in b.editor.seen[0][-1]["content"]


def test_long_peer_work_preserves_late_findings_and_results():
    _, (a, b) = peers(["A."], ["B."])
    code = "x" * 9000 + "\nThe repair is in the last column."
    result = "y" * 9000 + "\nPaired gain: 0.17"
    in_threads(lambda: a.reply("s", history("TASK", code, result)), lambda: b.reply("s", history("TASK", "B.", "R")))
    shared = b.editor.seen[0][-1]["content"]
    assert code in shared and result in shared
    assert "[cut]" not in shared


def test_peers_in_real_sessions_one_learns_from_the_other_and_the_best_candidate_wins(tmp_path, sandbox):
    x, y = data(400)
    held = data(200, seed=7)

    class Copier:
        """Looks first, then applies whatever fix a peer's work shows."""
        name = "copier"

        def __init__(self):
            self.calls = 0

        def reply(self, system, messages):
            self.calls += 1
            if self.calls == 1:
                return Reply(py("print('looking')"))
            if "w[0, 0] = -1.0" in messages[-1]["content"]:
                return Reply(py(GOOD, "A peer found the fix, so I apply it."))
            return Reply("Done.")

    finder = ScriptedEditor([py(GOOD, "Found it."), "Done."], "finder")
    r = wsplab.search(broken(), editors=[finder, Copier()], build={"mlp": [2, 2]}, description="a classifier",
                      practice=ArrayScorer(x, y), evaluate=ArrayScorer(*held, prefix="eval"), shared=True, spending=Spending(**CAP),
                      folder=tmp_path / "s", budget=0, limits=LIMITS, control=False)
    assert r.result.verdict == "confirmed" and r.result.final == 1.0
    assert all(c["score"] == 1.0 for c in r.candidates)
    calls = (tmp_path / "s" / "run_1" / "session" / "tool_calls.jsonl").read_text()
    assert "w[0, 0] = -1.0" in calls


def test_parse_recognises_edits_blocks():
    assert parse("x\n```edits\nW[0] = 1\n```", {"edits"}).blocks[0].kind == "edits"
