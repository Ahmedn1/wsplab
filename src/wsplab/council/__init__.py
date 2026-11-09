"""Councils of editors, optional: several LLMs edit the same network as peers.

Each peer runs a whole session of its own (its own kernel, edits and budget) and, after every step, sees what the others
did and what came back. There is no lead and no vote: each peer decides its own next step, and the best candidate on a
held-out selection set wins. Run one through `wsplab.search(..., shared=True)`; one editor stays the default.
"""

from .peers import Board, Peer

__all__ = ["Board", "Peer"]
