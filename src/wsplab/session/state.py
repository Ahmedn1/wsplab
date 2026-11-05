"""A session's record, in the shape of the paper's state.json."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class State:
    budget: int = 0
    max_runs: int = 300
    turn: int = 0
    runs: int = 0
    passes: int = 0
    requests: int = 0
    done: bool = False
    nudges: int = 0
    handoffs: int = 0
    since_handoff: int = 0
    flags: int = 0
    flag_streak: int = 0
    unexplained: int = 0
    unexplained_streak: int = 0
    notices: list[str] = field(default_factory=list)
    stop_reason: str | None = None

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=1) + "\n")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> State:
        return cls(**json.loads(path.read_text()))
