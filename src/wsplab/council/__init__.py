"""A council of LLMs: several editors vote on what happens to the weights, instead of one deciding.

A Council is an Editor, so nothing else changes. A chair drives the analysis alone. When a reply would change weights,
every member proposes its own reply from the same transcript, members rank the anonymised proposals, and only the winner
runs. The kernel refuses edits from any block that did not win a vote, so an edit cannot slip past. A vote never overrides
measurement: the winner still has to pass the evaluation gate.
"""

from __future__ import annotations

import ast
import json
import random
import string
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..editors.base import Editor, Reply, Usage
from ..editors.cost import Metered, Spending
from ..session.protocol import parse
from .tally import METHODS, Ballot, read_ballot, score, winners

__all__ = ["BALLOT", "Ballot", "Council", "METHODS", "edits_weights", "read_ballot", "score", "vote", "winners"]

EDIT_CALLS = {"save_edit", "drop_edit"}
EDIT_VERBS = {"write", "scale", "zero", "add", "project_out"}

BALLOT = """You are one of several editors deciding together. Above is the session so far. Each of you proposed the next reply independently, and they are listed below in random order, without names. Only one will be run. Judge each by whether the evidence supports it, not by how it is written.

{proposals}

Reply with your reasons, then end with one line ranking every proposal, best first, for example:
RANKING: {example}"""
APPROVAL = """You are one of several editors deciding together. Above is the session so far. Each of you proposed the next reply independently, and they are listed below in random order, without names. Only one will be run. Judge each by whether the evidence supports it, not by how it is written.

{proposals}

Reply with your reasons, then end with one line listing every proposal you approve of, for example:
APPROVE: {example}"""


def edits_weights(code: str) -> bool:
    """Whether a Python block visibly changes weights. The kernel's guard catches anything this misses."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in EDIT_CALLS:
            return True
        if isinstance(node, ast.Attribute) and node.attr in EDIT_VERBS and _rooted_at_m(node):
            return True
    return False


def _rooted_at_m(node: ast.AST) -> bool:
    while True:
        if isinstance(node, ast.Attribute | ast.Subscript | ast.Call):
            node = node.value if not isinstance(node, ast.Call) else node.func
        else:
            return isinstance(node, ast.Name) and node.id == "M"


def _edit_block(text: str) -> str | None:
    """The code or edits a reply would apply to the weights, or None if it changes nothing."""
    blocks = parse(text, {"python", "edits"}).blocks
    if len(blocks) != 1:
        return None
    block = blocks[0]
    if block.kind == "edits" or edits_weights(block.body):
        return block.body
    return None


@dataclass
class Round:
    turn: int
    proposals: list[dict]
    ballots: list[dict]
    totals: list[float]
    winner: int
    tie: bool


@dataclass
class Council:
    """`members[chair]` drives the session. `spending` is required: a round costs about two calls per member."""

    members: list[Editor]
    spending: Spending
    method: str = "borda"
    chair: int = 0
    weights: dict[str, float] = field(default_factory=dict)
    seed: int = 0
    folder: Path | None = None

    def __post_init__(self):
        if len(self.members) < 2:
            raise ValueError("a council needs at least two members")
        if self.method not in METHODS:
            raise ValueError(f"unknown voting method {self.method!r}: use one of {METHODS}")
        names = [m.name for m in self.members]
        if len(set(names)) != len(names):
            self.members = [_Renamed(m, f"{m.name}#{i}") for i, m in enumerate(self.members)]
        self.metered = [Metered(m, self.spending) for m in self.members]
        self.name = "council:" + "+".join(m.name for m in self.members)
        self.rng = random.Random(self.seed)
        self.rounds: list[Round] = []
        self._approved: set[str] = set()

    # The Editor interface ------------------------------------------------------------------------------------------
    def attach(self, session) -> None:
        self.folder = session.folder

    def approves(self, code: str) -> bool:
        return code in self._approved

    def reply(self, system: str, messages: list[dict]) -> Reply:
        first = self.metered[self.chair].reply(system, messages)
        if _edit_block(first.text) is None:
            return first
        return self._round(system, messages, first)

    def evidence(self) -> dict:
        return {"members": [m.name for m in self.members], "chair": self.members[self.chair].name, "scope": "scripts",
                "method": self.method, "rounds": len(self.rounds), "tokens": self.spending.tokens,
                "split_decisions": sum(r.tie for r in self.rounds)}

    # A round ---------------------------------------------------------------------------------------------------------
    def _in_parallel(self, fn, items) -> list:
        with ThreadPoolExecutor(max(1, len(items))) as pool:
            return list(pool.map(fn, items))

    def _round(self, system: str, messages: list[dict], first: Reply) -> Reply:
        others = [i for i in range(len(self.metered)) if i != self.chair]
        replies = dict(zip(others, self._in_parallel(lambda i: self.metered[i].reply(system, messages), others)))
        replies[self.chair] = first
        proposals = [(i, r) for i, r in sorted(replies.items()) if _edit_block(r.text) is not None]
        ballots = self._in_parallel(lambda i: self._ballot(i, system, messages, proposals), range(len(self.metered)))
        totals = score(ballots, len(proposals), self.method)
        leaders = winners(totals)
        winner = leaders[0] if len(leaders) == 1 else self._chair_choice(leaders, ballots, proposals)
        record = Round(len(messages), [{"member": self.members[i].name, "text": r.text} for i, r in proposals],
                       [{"voter": b.voter, "order": b.order, "reason": b.reason} for b in ballots], totals, winner,
                       len(leaders) > 1)
        self.rounds.append(record)
        self._log(record)
        chosen = proposals[winner][1]
        self._approved.add(_edit_block(chosen.text))
        usage = Usage()
        for _, r in proposals:
            usage.add(r.usage)
        return Reply(chosen.text, usage)

    def _chair_choice(self, leaders: list[int], ballots: list[Ballot], proposals) -> int:
        """Ties go to the chair's own ranking, then to the chair's proposal, then to the first."""
        for c in ballots[self.chair].order:
            if c in leaders:
                return c
        own = [k for k, (i, _) in enumerate(proposals) if i == self.chair and k in leaders]
        return own[0] if own else leaders[0]

    def _ballot(self, member: int, system: str, messages: list[dict], proposals) -> Ballot:
        """Each voter sees the proposals in its own random order, with no names, so neither position nor authorship helps."""
        order = list(range(len(proposals)))
        self.rng.shuffle(order)
        labels = {string.ascii_uppercase[k]: c for k, c in enumerate(order)}
        listed = "\n\n".join(f"=== PROPOSAL {label} ===\n{proposals[c][1].text}" for label, c in labels.items())
        template = APPROVAL if self.method == "approval" else BALLOT
        joiner = ", " if self.method == "approval" else " > "
        ask = template.format(proposals=listed, example=joiner.join(labels))
        reply = self.metered[member].reply(system, [*messages, {"role": "user", "content": ask}])
        name = self.members[member].name
        return Ballot(name, read_ballot(reply.text, labels), self.weights.get(name, 1.0), reply.text)

    def _log(self, record: Round) -> None:
        if self.folder is None:
            return
        with open(Path(self.folder) / "council.jsonl", "a") as f:
            f.write(json.dumps(record.__dict__) + "\n")


class _Renamed:
    """Two members may be the same model; the record must still tell them apart."""

    def __init__(self, editor, name: str):
        self.editor = editor
        self.name = name

    def reply(self, system: str, messages: list[dict]) -> Reply:
        return self.editor.reply(system, messages)


def vote(proposals: list[str], members: list[Editor], *, system: str, messages: list[dict], spending: Spending,
         method: str = "borda", seed: int = 0, measured: list[float] | None = None) -> dict:
    """Vote on finished candidates, each described by text. Ties go to the best measured score when there is one."""
    council = Council(list(members), spending, method, seed=seed)
    fake = [(i, Reply(p)) for i, p in enumerate(proposals)]
    ballots = council._in_parallel(lambda i: council._ballot(i, system, messages, fake), range(len(council.metered)))
    totals = score(ballots, len(proposals), method)
    leaders = winners(totals)
    winner = max(leaders, key=lambda i: measured[i]) if measured and len(leaders) > 1 else leaders[0]
    return {"winner": winner, "totals": totals, "tie": len(leaders) > 1,
            "ballots": [{"voter": b.voter, "order": b.order, "reason": b.reason} for b in ballots]}
