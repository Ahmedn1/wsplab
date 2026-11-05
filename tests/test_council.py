"""The council with scripted members: no LLM, no spending."""

import json

import numpy as np
import pytest
import torch

import wsplab
from wsplab import patch as P
from wsplab.council import Ballot, Council, edits_weights, read_ballot, score, vote, winners
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


def test_edit_detection():
    assert edits_weights("save_edit('w', x)")
    assert edits_weights("drop_edit('w')")
    assert edits_weights("f = save_edit\nf('w', x)")
    assert edits_weights("M.layers[3].mlp.neuron[1].out.zero()")
    assert edits_weights('M.module["a"].unit[0].weights.scale(2)')
    assert not edits_weights("w = load('w')\nprint(w.mean())")
    assert not edits_weights("M.layers[3].mlp.neuron[1].out.read()")
    assert not edits_weights("some.other.zero()")
    assert not edits_weights("def (")


def test_read_ballot():
    labels = {"A": 2, "B": 0, "C": 1}
    assert read_ballot("Reasons.\nRANKING: B > A > C", labels) == [0, 2, 1]
    assert read_ballot("ranking: c, a", labels) == [1, 2]
    assert read_ballot("first thoughts\nRANKING: A\nactually\nRANKING: C > A", labels) == [1, 2]
    assert read_ballot("RANKING: B > B > Z", labels) == [0]
    assert read_ballot("I cannot decide.", labels) == []


def ballots(*orders, weights=None):
    return [Ballot(f"v{i}", list(o), (weights or {}).get(i, 1.0)) for i, o in enumerate(orders)]


def test_borda():
    assert score(ballots([0, 1, 2], [1, 0, 2], [1, 2, 0]), 3, "borda") == [3.0, 5.0, 1.0]


def test_plurality():
    assert score(ballots([0, 1], [0, 1], [1, 0]), 2, "plurality") == [2.0, 1.0]


def test_approval_counts_everything_listed():
    assert score(ballots([0, 1], [1], [1, 2]), 3, "approval") == [1.0, 3.0, 1.0]


def test_weights_and_abstentions():
    assert score(ballots([0], [1], [], weights={0: 3.0}), 2, "plurality") == [3.0, 1.0, ][:2]


def test_unknown_method_and_winners():
    with pytest.raises(ValueError):
        score([], 1, "dictator")
    assert winners([1.0, 3.0, 3.0]) == [1, 2]


def council(chair_first, a_ballot, b_ballot, b_proposal="edit('x + 2', 'Mine.')", c=None, **kw):
    """Three members. Each script: [proposal, ballot]."""
    members = [named("a", [chair_first, a_ballot]), named("b", [eval(b_proposal), b_ballot]),
               named("c", c or [edit("x + 3", "Third."), "RANKING: A > B > C"])]
    return Council(members, Spending(**CAP), **kw), members


def test_non_edit_replies_skip_the_vote():
    a, b = named("a", ["Just reading.\n```python\nprint(1)\n```"]), named("b", [])
    c = Council([a, b], Spending(**CAP))
    assert "print(1)" in c.reply("s", [{"role": "user", "content": "go"}]).text
    assert b.seen == [] and c.rounds == []


def test_the_winner_of_the_vote_is_what_runs():
    c, members = council(edit("x + 1", "Chair."), "RANKING: A > B > C", "RANKING: A > B > C",
                         c=[edit("x + 3", "Third."), "RANKING: A > B > C"], seed=1)
    reply = c.reply("s", [{"role": "user", "content": "go"}])
    r = c.rounds[0]
    assert r.proposals[r.winner]["text"] == reply.text
    assert len(r.proposals) == 3 and len(r.ballots) == 3


def test_ballots_hide_authors_and_shuffle_per_voter():
    members = [named("alice-model", [edit("x + 1", "Alpha."), "RANKING: A"]),
               named("bob-model", [edit("x + 2", "Beta."), "RANKING: A"]),
               named("carol-model", [edit("x + 3", "Gamma."), "RANKING: A"])]
    c = Council(members, Spending(**CAP), seed=3)
    c.reply("s", [{"role": "user", "content": "go"}])
    seen = [m.seen[-1][-1]["content"] for m in members]
    for text in seen:
        for name in ("alice-model", "bob-model", "carol-model"):
            assert name not in text
    orders = [tuple(x for x in ("Alpha", "Beta", "Gamma") if x in t) and
              tuple(sorted(("Alpha", "Beta", "Gamma"), key=t.index)) for t in seen]
    assert len(set(orders)) > 1


def test_votes_are_mapped_back_through_each_voters_own_order():
    """Everyone votes for the proposal that says Beta, whatever letter it has on their ballot."""

    class Voter:
        def __init__(self, name, proposal):
            self.name, self.proposal, self.calls = name, proposal, 0

        def reply(self, system, messages):
            self.calls += 1
            if self.calls == 1:
                return Reply(self.proposal)
            text = messages[-1]["content"]
            sections = text.split("=== PROPOSAL ")[1:]
            label = next(s[0] for s in sections if "Beta" in s)
            return Reply(f"RANKING: {label}")

    members = [Voter("a", edit("x + 1", "Alpha.")), Voter("b", edit("x + 2", "Beta.")), Voter("c", edit("x + 3", "Gamma."))]
    c = Council(members, Spending(**CAP), seed=5)
    assert "Beta" in c.reply("s", [{"role": "user", "content": "go"}]).text


def test_a_tie_goes_to_the_chairs_ranking():
    class Fixed:
        def __init__(self, name, proposal, favourite):
            self.name, self.proposal, self.favourite, self.calls = name, proposal, favourite, 0

        def reply(self, system, messages):
            self.calls += 1
            if self.calls == 1:
                return Reply(self.proposal)
            sections = messages[-1]["content"].split("=== PROPOSAL ")[1:]
            return Reply("RANKING: " + next(s[0] for s in sections if self.favourite in s))

    members = [Fixed("a", edit("x + 1", "Alpha."), "Beta"), Fixed("b", edit("x + 2", "Beta."), "Alpha")]
    c = Council(members, Spending(**CAP), method="plurality")
    assert "Beta" in c.reply("s", [{"role": "user", "content": "go"}]).text
    assert c.rounds[0].tie


def test_a_member_who_proposes_no_edit_has_no_candidate_but_still_votes():
    members = [named("a", [edit("x + 1"), "RANKING: A"]), named("b", ["I would not edit yet.", "RANKING: A"])]
    c = Council(members, Spending(**CAP))
    c.reply("s", [{"role": "user", "content": "go"}])
    assert len(c.rounds[0].proposals) == 1 and len(c.rounds[0].ballots) == 2


def test_duplicate_models_are_told_apart():
    c = Council([named("same", []), named("same", [])], Spending(**CAP))
    assert [m.name for m in c.members] == ["same#0", "same#1"]


def test_a_council_needs_two_members_and_a_real_method():
    with pytest.raises(ValueError, match="two members"):
        Council([named("a", [])], Spending(**CAP))
    with pytest.raises(ValueError, match="method"):
        Council([named("a", []), named("b", [])], Spending(**CAP), method="dice")


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


def test_council_in_a_real_session_applies_the_winning_edit(tmp_path, sandbox):
    x, y = data()
    pick_good = "Fixing rows looks right.\nRANKING: {}"

    class Member:
        def __init__(self, name, replies):
            self.name, self.replies = name, list(replies)

        def reply(self, system, messages):
            r = self.replies.pop(0)
            if r == "VOTE":
                sections = messages[-1]["content"].split("=== PROPOSAL ")[1:]
                return Reply(pick_good.format(next(s[0] for s in sections if "w[0, 0] = -1.0" in s)))
            return Reply(r)

    chair = Member("chair", [py("print(NAMES)"), py(BAD), "VOTE", "Done."])
    other = Member("other", [py(GOOD), "VOTE"])
    third = Member("third", [py(GOOD), "VOTE"])
    council_ = Council([chair, other, third], Spending(**CAP), seed=2)
    r = wsplab.run(broken(), build={"mlp": [2, 2]}, description="a classifier", evaluate=ArrayScorer(x, y), budget=0,
                   editor=council_, folder=tmp_path / "r", limits=LIMITS, control=False)
    assert r.verdict == "confirmed" and r.final == 1.0
    record = [json.loads(line) for line in (r.folder / "session" / "council.jsonl").read_text().splitlines()]
    assert len(record) == 1 and len(record[0]["proposals"]) == 3
    evidence = P.Patch.load(r.patch).meta["council"]
    assert evidence["rounds"] == 1 and evidence["members"] == ["chair", "other", "third"]


def test_an_edit_hidden_from_the_static_check_is_refused_by_the_kernel(tmp_path, sandbox):
    x, y = data()
    hidden = "code = 'save' + '_edit'\nglobals()[code]('layers.0.weight', load('layers.0.weight') + 1)"
    assert not edits_weights(hidden)
    chair = named("chair", [py(hidden), "Done."])
    r = wsplab.run(broken(), build={"mlp": [2, 2]}, description="a classifier", evaluate=ArrayScorer(x, y), budget=0,
                   editor=Council([chair, named("b", [])], Spending(**CAP)), folder=tmp_path / "r", limits=LIMITS)
    assert "not approved" in (r.folder / "session" / "tool_calls.jsonl").read_text()
    assert r.verdict == "no_fix_found"


def test_one_editor_is_unchanged_by_the_council_code(tmp_path, sandbox):
    x, y = data()
    r = wsplab.run(broken(), build={"mlp": [2, 2]}, description="a classifier", evaluate=ArrayScorer(x, y), budget=0,
                   editor=ScriptedEditor([py(GOOD), "Done."]), folder=tmp_path / "r", limits=LIMITS, control=False)
    assert r.verdict == "confirmed" and "council" not in P.Patch.load(r.patch).meta


def test_budget_exhausted_ends_the_session_cleanly(tmp_path, sandbox):
    x, y = data()

    class Expensive(ScriptedEditor):
        def reply(self, system, messages):
            r = super().reply(system, messages)
            return Reply(r.text, Usage(50, 50))

    members = [Expensive([py(GOOD), "RANKING: A"], "a"), Expensive([py(GOOD), "RANKING: A"], "b")]
    r = wsplab.run(broken(), build={"mlp": [2, 2]}, description="a classifier", evaluate=ArrayScorer(x, y), budget=0,
                   editor=Council(members, Spending(max_tokens=100)), folder=tmp_path / "r", limits=LIMITS, control=False)
    assert "token limit" in r.state.stop_reason


def test_inline_replies_go_through_the_vote_too(tmp_path, sandbox):
    x, y = data()
    good = "Swap.\n```edits\nlayers.0.weight[0, 0] = -1\nlayers.0.weight[1, 0] = 1\n```"
    bad = "Break.\n```edits\nlayers.0.weight[0, 1] = 9\n```"

    class Member:
        def __init__(self, name, proposal):
            self.name, self.proposal, self.n = name, proposal, 0

        def reply(self, system, messages):
            self.n += 1
            if self.n == 1:
                return Reply(self.proposal)
            sections = messages[-1]["content"].split("=== PROPOSAL ")[1:]
            return Reply("RANKING: " + next(s[0] for s in sections if "Swap" in s))

    r = wsplab.run(broken(), build={"mlp": [2, 2]}, description="a classifier", evaluate=ArrayScorer(x, y), budget=0,
                   access="inline", editor=Council([Member("a", bad), Member("b", good), Member("c", good)], Spending(**CAP)),
                   folder=tmp_path / "r", limits=LIMITS, control=False)
    assert r.final == 1.0


def test_vote_on_finished_candidates_breaks_ties_by_measurement():
    class Split:
        def __init__(self, name, favourite):
            self.name, self.favourite = name, favourite

        def reply(self, system, messages):
            sections = messages[-1]["content"].split("=== PROPOSAL ")[1:]
            return Reply("RANKING: " + next(s[0] for s in sections if self.favourite in s))

    out = vote(["first", "second"], [Split("a", "first"), Split("b", "second")], system="s", messages=[],
               spending=Spending(**CAP), method="plurality", measured=[0.1, 0.9])
    assert out["tie"] and out["winner"] == 1


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


def test_search_with_a_vote_on_candidates(tmp_path, sandbox):
    x, y = data(400)
    held = data(200, seed=7)

    class Candidate(ScriptedEditor):
        def reply(self, system, messages):
            content = messages[-1]["content"]
            if "=== PROPOSAL" in content:
                sections = content.split("=== PROPOSAL ")[1:]
                return Reply("RANKING: " + next(s[0] for s in sections if "Swapped the rows" in s))
            return super().reply(system, messages)

    good, bad = search_editors()
    r = wsplab.search(broken(), editors=[Candidate(list(bad), "bad"), Candidate(list(good), "good")], build={"mlp": [2, 2]},
                      description="a classifier", practice=ArrayScorer(x, y), evaluate=ArrayScorer(*held, prefix="eval"),
                      selection="vote", spending=Spending(**CAP), folder=tmp_path / "s", budget=0, limits=LIMITS, control=False)
    assert r.selection["method"] == "vote" and r.selection["run"] == 1 and r.result.verdict == "confirmed"


def test_parse_recognises_edits_blocks():
    assert parse("x\n```edits\nW[0] = 1\n```", {"edits"}).blocks[0].kind == "edits"
