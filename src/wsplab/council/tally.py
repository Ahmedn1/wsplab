"""Counting votes. Pure functions over ballots, so the rules are easy to test and to read."""

from __future__ import annotations

import re
from dataclasses import dataclass

METHODS = ("borda", "plurality", "approval")
_LINE = re.compile(r"^\s*(?:RANKING|APPROVE)\s*:\s*(.+)$", re.M | re.I)


@dataclass
class Ballot:
    """Candidates in order of preference, by index. An empty ballot is an abstention."""

    voter: str
    order: list[int]
    weight: float = 1.0
    reason: str = ""


def read_ballot(text: str, labels: dict[str, int]) -> list[int]:
    """Candidate indices from a reply's last RANKING or APPROVE line. Unknown labels are dropped and repeats ignored."""
    found = _LINE.findall(text)
    if not found:
        return []
    order: list[int] = []
    for token in re.findall(r"[A-Za-z]+", found[-1]):
        index = labels.get(token.upper())
        if index is not None and index not in order:
            order.append(index)
    return order


def score(ballots: list[Ballot], candidates: int, method: str) -> list[float]:
    if method not in METHODS:
        raise ValueError(f"unknown voting method {method!r}: use one of {METHODS}")
    totals = [0.0] * candidates
    for ballot in ballots:
        if method == "plurality":
            if ballot.order:
                totals[ballot.order[0]] += ballot.weight
        elif method == "approval":
            for c in ballot.order:
                totals[c] += ballot.weight
        else:
            for rank, c in enumerate(ballot.order):
                totals[c] += ballot.weight * (candidates - 1 - rank)
    return totals


def winners(totals: list[float]) -> list[int]:
    best = max(totals)
    return [i for i, t in enumerate(totals) if t == best]
