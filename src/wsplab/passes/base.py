"""Pass tools: the budgeted ways an editor can run the network. Each owns one fence, such as ```forward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class PassInvalid(Exception):
    """The request itself is wrong. It does not use up a pass."""


class PassFailed(Exception):
    """The request was fine but running it failed. It does not use up a pass."""


@dataclass
class PassResult:
    kind: str
    where: str
    summary: dict | str


@runtime_checkable
class PassTool(Protocol):
    fence: str

    def run(self, body: str, request_number: int) -> PassResult:
        """Run one request. `request_number` counts every request, so result files never collide."""
        ...
