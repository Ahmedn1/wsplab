"""Peers: each model works on its own copy and decides for itself, and everyone sees everyone's work after every step.

There is no lead and no vote. Each peer runs a whole session of its own, with its own kernel, edits and budget. Peers move in rounds:
after a step, a peer waits for the others, then sees what each of them did and what came back, and decides its own next step. It can
adopt, test or contradict anything it sees. A peer's reply is already its findings, so sharing costs no extra calls.
"""

from __future__ import annotations

import threading

from ..editors.base import Editor, Reply

REPLY_CHARS = 2500
RESULT_CHARS = 1500
PEER_NOTES = """

WHAT YOUR PEERS DID LAST. Each works independently on their own copy of the network, with their own edits and budget, and sees this too.
{notes}
Nothing here is applied to your copy. Adopt, test or contradict any of it: your next step is your own decision."""


class Board:
    """Where peers meet between steps. A round closes once every peer still working has posted."""

    def __init__(self, peers: int):
        self.active = set(range(peers))
        self.posts: dict[int, dict] = {}
        self.closed: dict[int, dict] = {}
        self.generation = 0
        self.condition = threading.Condition()

    def _close_if_complete(self) -> None:
        if self.posts and set(self.posts) >= self.active:
            self.closed = self.posts
            self.posts = {}
            self.generation += 1
            self.condition.notify_all()

    def exchange(self, peer: int, post: dict) -> dict[int, dict]:
        """Post this peer's step, wait for the round to close, and return what the others posted."""
        with self.condition:
            generation = self.generation
            self.posts[peer] = post
            self._close_if_complete()
            while self.generation == generation:
                self.condition.wait()
            return {k: v for k, v in self.closed.items() if k != peer}

    def leave(self, peer: int) -> None:
        """A peer that finishes or fails stops being waited for."""
        with self.condition:
            self.active.discard(peer)
            self.posts.pop(peer, None)
            self._close_if_complete()


def _cut(text: str, limit: int) -> str:
    text = text.replace("```", "\'\'\'")
    return text if len(text) <= limit else text[:limit] + "\n[cut]"


class Peer:
    """One member, as an Editor for its own session."""

    def __init__(self, editor: Editor, board: Board, index: int):
        self.editor = editor
        self.board = board
        self.index = index
        self.name = editor.name
        self.notes: dict[int, str] = {}
        self.rounds = 0

    def _with_notes(self, messages: list[dict]) -> list[dict]:
        return [{**m, "content": m["content"] + PEER_NOTES.format(notes=self.notes[i])} if i in self.notes else m
                for i, m in enumerate(messages)]

    def reply(self, system: str, messages: list[dict]) -> Reply:
        if len(messages) > 1:
            others = self.board.exchange(self.index, {"reply": messages[-2]["content"], "result": messages[-1]["content"]})
            if others:
                self.rounds += 1
                self.notes[len(messages) - 1] = "\n".join(
                    f"Peer {chr(65 + k)}:\n{_cut(p['reply'], REPLY_CHARS)}\nWhat came back:\n{_cut(p['result'], RESULT_CHARS)}\n"
                    for k, p in sorted(others.items()))
        return self.editor.reply(system, self._with_notes(messages))
