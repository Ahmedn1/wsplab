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
from ..editors.cost import BudgetExceeded, Metered, Spending
from ..session.protocol import parse
from .peers import Board, Peer
from .tally import METHODS, Ballot, read_ballot, score, winners

__all__ = ["BALLOT", "Ballot", "Board", "Council", "METHODS", "Peer", "Team", "edits_weights", "read_ballot", "score", "vote", "winners"]

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
    """`members[chair]` drives the session. `spending` is required: a round costs about two calls per member.

    `exclude_self` leaves a member's own proposal off its ballot: voters tend to rank their own first even when it is anonymous."""

    members: list[Editor]
    spending: Spending
    method: str = "borda"
    chair: int = 0
    weights: dict[str, float] = field(default_factory=dict)
    seed: int = 0
    folder: Path | None = None
    exclude_self: bool = False

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
        order = [c for c, (author, _) in enumerate(proposals) if not (self.exclude_self and author == member)]
        name = self.members[member].name
        if not order:
            return Ballot(name, [], self.weights.get(name, 1.0), "")
        self.rng.shuffle(order)
        labels = {string.ascii_uppercase[k]: c for k, c in enumerate(order)}
        listed = "\n\n".join(f"=== PROPOSAL {label} ===\n{proposals[c][1].text}" for label, c in labels.items())
        template = APPROVAL if self.method == "approval" else BALLOT
        joiner = ", " if self.method == "approval" else " > "
        ask = template.format(proposals=listed, example=joiner.join(labels))
        conversation = [*messages, {"role": "user", "content": ask}]
        reply = self.metered[member].reply(system, conversation)
        order = read_ballot(reply.text, labels)
        reason = reply.text
        if not order:
            # A voter can reason at length and forget the last line. That is a vote worth one more short call, not an abstention.
            again = conversation + [{"role": "assistant", "content": reply.text},
                                    {"role": "user", "content": f"End with the ranking line, for example:\nRANKING: {joiner.join(labels)}"}]
            reply = self.metered[member].reply(system, again)
            order = read_ballot(reply.text, labels)
            reason += "\n\n" + reply.text
        return Ballot(name, order, self.weights.get(name, 1.0), reason)

    def _log(self, record: Round) -> None:
        if self.folder is None:
            return
        with open(Path(self.folder) / "council.jsonl", "a") as f:
            f.write(json.dumps(record.__dict__) + "\n")


TEAM_ASK = """You are working on this together with teammates, and one of you, the lead, writes the next step. Before they do, share what you found. In at most {words} words: what the latest result shows that the team has not already established, what it leaves unexplained, and the one thing most worth trying or checking next. Do not repeat what the team already agrees on: add something new, or argue for a competing explanation. A null result is worth sharing. Say so if you think an edit is ready to save. Plain text only: no code blocks."""
TEAM_NOTES = """

FINDINGS FROM YOUR TEAMMATES, who read the same result:
{notes}
Use what helps. You decide the next step."""


@dataclass
class Team:
    """A cooperative council. After every result each teammate shares a short note of findings, and the lead writes the next step
    with those notes in view. Notes stay attached to the result they answer, so everyone sees the whole discussion from then on.

    There is no vote: insights meet at every step instead of only at an edit. Teammates are asked for what is new or contested, since
    on a live run they simply echoed one another. A Team is an Editor, so the session, kernel and patch
    code do not change, and the transcript holds the lead's replies alone. `spending` is required: a turn costs a call per member.
    """

    members: list[Editor]
    spending: Spending
    lead: int = 0
    words: int = 150
    folder: Path | None = None

    def __post_init__(self):
        if len(self.members) < 2:
            raise ValueError("a team needs at least two members")
        names = [m.name for m in self.members]
        if len(set(names)) != len(names):
            self.members = [_Renamed(m, f"{m.name}#{i}") for i, m in enumerate(self.members)]
        self.metered = [Metered(m, self.spending) for m in self.members]
        self.name = "team:" + "+".join(m.name for m in self.members)
        self.notes: dict[str, str] = {}
        self.rounds = 0
        self.lost_notes = 0

    def attach(self, session) -> None:
        self.folder = session.folder

    def evidence(self) -> dict:
        return {"members": [m.name for m in self.members], "lead": self.members[self.lead].name, "scope": "cooperative",
                "rounds": self.rounds, "lost_notes": self.lost_notes, "tokens": self.spending.tokens}

    def _with_notes(self, messages: list[dict]) -> list[dict]:
        """The conversation with each result followed by the team's notes on it. Only ever appends to what was sent before, so
        prompt caches keep working."""
        out = []
        for m in messages:
            note = self.notes.get(m["content"]) if m["role"] == "user" else None
            out.append({**m, "content": m["content"] + TEAM_NOTES.format(notes=note)} if note else m)
        return out

    def _note(self, member: int, system: str, conversation: list[dict]) -> str | None:
        ask = {"role": "user", "content": TEAM_ASK.format(words=self.words)}
        try:
            return self.metered[member].reply(system, [*conversation, ask]).text.replace("```", "'''").strip()
        except BudgetExceeded:
            raise
        except Exception:  # noqa: BLE001 - one teammate failing must not stop the others
            self.lost_notes += 1
            return None

    def reply(self, system: str, messages: list[dict]) -> Reply:
        if len(messages) > 1 and messages[-1]["content"] not in self.notes:  # the first message is the task: nothing to report yet
            conversation = self._with_notes(messages)
            others = [i for i in range(len(self.metered)) if i != self.lead]
            with ThreadPoolExecutor(len(others)) as pool:
                found = list(pool.map(lambda i: (i, self._note(i, system, conversation)), others))
            lines = [f"Teammate {n}: {text}" for n, (_, text) in enumerate(found, 1) if text]
            if lines:
                self.notes[messages[-1]["content"]] = "\n".join(lines) + "\n"
                self.rounds += 1
                self._log(found)
        return self.metered[self.lead].reply(system, self._with_notes(messages))

    def _log(self, found: list) -> None:
        if self.folder is None:
            return
        with open(Path(self.folder) / "team.jsonl", "a") as f:
            f.write(json.dumps({"round": self.rounds, "notes": {self.members[i].name: text for i, text in found}}) + "\n")


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
