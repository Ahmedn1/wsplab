"""The session loop: send a message, read one block, run it, report back, until the budget is spent.

Reconstructed from the published prompts and logs (exp8_math/runs/p50c, exp8_math/gemma_followup/runs/g50b): the
authors did not release their driver. It writes the same layout as their logs: turns/tNNN/{prompt.txt,answer.md},
tool_calls.jsonl, state.json.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .. import log
from ..editors.base import Editor, EditorRefused, Usage
from ..editors.cost import BudgetExceeded
from ..passes.base import PassFailed, PassInvalid, PassTool
from ..prompts import SYSTEM
from .messages import V5, Messages
from .protocol import Block, parse
from .state import State

NOTES_CHARS = 8000


def _spending(editor):
    """The spending limit an editor is metered against, if any: council members wrap their editor in a few layers."""
    for _ in range(5):
        if editor is None:
            return None
        if getattr(editor, "spending", None) is not None:
            return editor.spending
        editor = getattr(editor, "editor", None) or getattr(editor, "inner", None)
    return None


def _short_result(summary) -> str:
    """A forward pass's result in a few words: the scores, never the files or per-item details."""
    if isinstance(summary, str):
        return summary.strip().splitlines()[0][:120] if summary.strip() else ""

    def score(value):
        if isinstance(value, dict):
            value = value.get("accuracy", value.get("score"))
        return f"{value:.4g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    parts = []
    for key in ("model", "items", "score", "accuracy", "original", "edited"):
        if key in summary:
            shown = summary[key] if key in ("model", "items") else score(summary[key])
            if shown is not None:
                parts.append(f"{key} {shown}")
    pair = summary.get("paired")
    if isinstance(pair, dict):
        difference = next((v for k, v in pair.items() if k.endswith("difference")), None)
        if isinstance(difference, (int, float)):
            error = pair.get("standard_error")
            parts.append(f"difference {difference:+.4g}" + (f" ± {error:.2g}" if isinstance(error, (int, float)) else ""))
    return ", ".join(parts)


class Kernel(Protocol):
    def run(self, code: str) -> str: ...
    def edited(self) -> list[str]: ...
    def files(self) -> list[str]: ...
    def notes(self) -> str: ...
    def edits_used(self) -> tuple[list[str], str | None]:
        """The tensors currently edited, and a short hash of their values."""
        ...


@dataclass
class Policy:
    """explain: "enforce" sends unexplained blocks back, at most `explain_retries` times in a row; "remind" adds a
    reminder to the result of an unexplained block; "off" does neither. `require_all_passes` is the paper's rule that a final
    answer does not end the session while passes are left. `warn_unsaved` warns the editor this many Python runs before the
    limit if it has saved no edit: live editors have spent every run on analysis and been left unable to apply what they found."""

    explain: str = "enforce"
    explain_retries: int = 2
    report_edits: bool = True
    handoff_every: int = 40
    max_flags: int = 3
    max_nudges: int = 5
    require_all_passes: bool = True
    warn_unsaved: int = 0
    max_turns: int | None = None


@dataclass
class Result:
    final_answer: str | None
    state: State
    usage: Usage = field(default_factory=Usage)


class Stop(Exception):
    """Ends a session early, with a reason."""


class Session:
    def __init__(self, task: str, editor: Editor, kernel: Kernel, tools: dict[str, PassTool] | None = None, *,
                 budget: int = 0, max_runs: int = 300, folder: str | Path, messages: Messages = V5,
                 policy: Policy | None = None, system: str = SYSTEM,
                 before_turn: Callable[[Session], None] | None = None,
                 after_reply: Callable[[Session], None] | None = None, resume: bool = False):
        self.task = task
        self.editor = editor
        self.kernel = kernel
        self.tools = tools or {}
        self.messages = messages
        self.policy = policy or Policy()
        self.system = system
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.state = State(budget=budget, max_runs=max_runs)
        self.history: list[dict] = []
        self.usage = Usage()
        self.before_turn = before_turn
        self.after_reply = after_reply
        self._prefix: list[str] = []
        self._kinds = {"python", *self.tools}
        self._resumed_message: str | None = None
        self._truncated = False
        self._did = ""  # what this turn's reply led to, for the progress line
        self._asked = (0.0, Usage())  # this turn's editor time and tokens
        if resume and (self.folder / "state.json").exists() and (self.folder / "next.json").exists():
            self._resume()
        if hasattr(editor, "attach"):
            editor.attach(self)

    # Things that can happen to a running session -------------------------------------------------------------------
    def notice(self, text: str, name: str | None = None) -> None:
        """Deliver a note to the editor with the next message."""
        self._prefix.append(self.messages.note.format(text=text) + "\n\n")
        if name:
            self.state.notices.append(name)

    def kernel_restarted(self) -> None:
        self._prefix.append(self.messages.kernel_restarted + "\n")

    def new_rule(self, text: str) -> None:
        self._prefix.append(self.messages.new_rule.format(text=text) + "\n\n")

    def _resume(self) -> None:
        """Continue from the last completed turn. The conversation and /work survive a crash, the kernel's variables do
        not, and the editor is told so, as it was in the paper's own restarts."""
        saved = json.loads((self.folder / "next.json").read_text())
        self.state = State.load(self.folder / "state.json")
        self.history = saved["history"]
        self._resumed_message = saved["message"]
        if not self.state.done:
            self.kernel_restarted()

    def _checkpoint(self, message: str) -> None:
        tmp = self.folder / "next.tmp"
        tmp.write_text(json.dumps({"message": message, "history": self.history}))
        tmp.replace(self.folder / "next.json")
        self.state.save(self.folder / "state.json")

    # The loop --------------------------------------------------------------------------------------------------------
    def run(self) -> Result:
        if self.state.done:
            # A crash after the session ended: its result stands, and asking the editor again would only cost money.
            answer = self.folder / "final_answer.md"
            return Result(answer.read_text() if answer.exists() else None, self.state, self.usage)
        message = self._resumed_message or self.task
        final = None
        while True:
            self.state.turn += 1
            if self.before_turn:
                self.before_turn(self)
            if self.policy.max_turns is not None and self.state.turn > self.policy.max_turns:
                self.state.stop_reason = "max_turns"
                break
            message = self._dress(message)
            try:
                reply = self._ask(message)
            except EditorRefused:
                self.state.flags += 1
                self.state.flag_streak += 1
                if self.state.flag_streak > self.policy.max_flags:
                    self.state.stop_reason = "refused"
                    break
                self.state.turn -= 1
                message = self.messages.refused
                continue
            except (Stop, BudgetExceeded) as e:
                self.state.stop_reason = str(e)
                break
            self.state.flag_streak = 0
            self.state.since_handoff += 1
            if self.after_reply:
                self.after_reply(self)
            outcome = self._handle(reply)
            self._report_turn(outcome is None)
            if outcome is not None:
                self._checkpoint(outcome)
            if outcome is None:
                final = reply
                self.state.done = True
                self.state.stop_reason = self.state.stop_reason or "finished"
                break
            message = outcome
        self.state.save(self.folder / "state.json")
        if final is not None:
            (self.folder / "final_answer.md").write_text(final)
        self._report_end(final)
        return Result(final, self.state, self.usage)

    def _report_turn(self, final: bool) -> None:
        s = self.state
        seconds, usage = self._asked
        did = "final answer" if final and not self._did else self._did
        self._did = ""
        log.info(f"turn {s.turn}: {did} | passes {s.passes}/{s.budget}, Python runs {s.runs}/{s.max_runs} | editor "
                 f"{log.duration(seconds)}, {log.tokens(usage.input_tokens)} in / {log.tokens(usage.output_tokens)} out tokens")

    def _report_end(self, final: str | None) -> None:
        s, u = self.state, self.usage
        log.info(f"editing session over after {log.plural(s.turn, 'turn')} ({s.stop_reason}): {s.passes}/{s.budget} passes, "
                 f"{log.plural(s.runs, 'Python run')}, {log.tokens(u.input_tokens)} input and {log.tokens(u.output_tokens)} output tokens "
                 f"in all; transcript in {self.folder}")
        spending = _spending(self.editor)
        if spending is not None:
            log.info(f"spent so far: ${spending.usd:.2f}" + (f" of ${spending.max_usd:.2f}" if spending.max_usd else ""))
        if final and final.strip():
            first = final.strip().splitlines()[0]
            log.info(f"the editor's answer begins: {first[:200]}{'...' if len(first) > 200 else ''}")

    def _dress(self, message: str) -> str:
        """Add what belongs around this turn's message: a handoff, then pending notes."""
        prefix = "".join(self._prefix)
        self._prefix.clear()
        message = prefix + message
        if self.state.since_handoff >= self.policy.handoff_every:
            message = self._handoff(message)
        return message

    def _handoff(self, last: str) -> str:
        s = self.state
        s.handoffs += 1
        s.since_handoff = 0
        self.history.clear()
        body = self.messages.handoff.format(
            replies=s.turn - 1, passes=s.passes, budget=s.budget, runs=s.runs, limit=s.max_runs,
            edited=self.kernel.edited(), files=self.kernel.files(), notes=self.kernel.notes()[-NOTES_CHARS:],
            notes_chars=NOTES_CHARS, last=last)
        return f"{self.task}\n\n{body}"

    def _ask(self, message: str) -> str:
        turn = self.folder / "turns" / f"t{self.state.turn:03d}"
        turn.mkdir(parents=True, exist_ok=True)
        (turn / "prompt.txt").write_text(message)
        self.history.append({"role": "user", "content": message})
        t0 = time.time()
        try:
            reply = self.editor.reply(self.system, list(self.history))
        except EditorRefused:
            self.history.pop()
            raise
        self.usage.add(reply.usage)
        self._asked = (time.time() - t0, reply.usage)
        self._truncated = reply.truncated
        self.history.append({"role": "assistant", "content": reply.text})
        (turn / "answer.md").write_text(reply.text)
        (turn / "run.json").write_text(json.dumps({"editor": self.editor.name, "seconds": round(time.time() - t0, 2),
                                                   "usage": reply.usage.__dict__}, indent=1) + "\n")
        return reply.text

    def _handle(self, reply: str) -> str | None:
        """The next message, or None when the reply is the final answer."""
        if self._truncated:
            self._did = "the reply hit the output limit, so its block was not run"
            return self.messages.truncated
        parsed = parse(reply, self._kinds)
        if not parsed.blocks:
            return self._no_block()
        self.state.nudge_streak = 0
        if len(parsed.blocks) > 1:
            self._did = f"{len(parsed.blocks)} blocks in one reply, none run"
            return self.messages.too_many_blocks.format(n=len(parsed.blocks))
        if self.policy.explain == "enforce" and not parsed.explanation:
            self.state.unexplained += 1
            if self.state.unexplained_streak < self.policy.explain_retries:
                self.state.unexplained_streak += 1
                self._did = "a block without an explanation, not run"
                return self.messages.no_explanation.format(explain=self.messages.explain)
        self.state.unexplained_streak = 0
        out = self._run_block(parsed.blocks[0])
        if self.policy.explain == "remind" and not parsed.explanation:
            out += "\n" + self.messages.reminder
        return out

    def _no_block(self) -> str | None:
        left = self.state.budget - self.state.passes
        if left > 0 and self.tools and self.policy.require_all_passes:
            self.state.nudges += 1
            self.state.nudge_streak += 1
            if self.state.nudge_streak > self.policy.max_nudges:
                self.state.stop_reason = "no_block"  # An editor that will not act must not spend the budget saying so.
                self._did = "no block again: the session ends"
                return None
            self._did = f"no block; reminded that {left} passes are left"
            return self.messages.nudge.format(left=left)
        return None

    def _run_block(self, block: Block) -> str:
        if block.kind == "python":
            return self._run_python(block.body)
        return self._run_pass(block)

    def _log(self, **entry) -> None:
        with open(self.folder / "tool_calls.jsonl", "a") as f:
            f.write(json.dumps({"t": time.time(), **entry}) + "\n")

    def _run_python(self, code: str) -> str:
        s = self.state
        if s.runs >= s.max_runs:
            self._did = "Python refused: no runs left"
            return self.messages.runs_exhausted.format(limit=s.max_runs)
        s.runs += 1
        t0 = time.time()
        approves = getattr(self.editor, "approves", None)
        out = self.kernel.run(code) if approves is None else self.kernel.run(code, allow_edits=approves(code))
        self._log(kind="python", s=round(time.time() - t0, 2), code=code, out=out)
        failed = "Traceback (most recent call last)" in out
        self._did = f"Python run {s.runs}{' (raised an error)' if failed else ''}, {log.duration(time.time() - t0)}"
        message = self.messages.python_output.format(k=s.runs, limit=s.max_runs, out=out)
        left = s.max_runs - s.runs
        if 0 < left <= self.policy.warn_unsaved and not self.kernel.edited():
            message += "\n" + self.messages.no_edit_yet.format(left=left)
        return message

    def _run_pass(self, block: Block) -> str:
        s = self.state
        if s.passes >= s.budget:
            self._did = "forward pass refused: none left"
            return self.messages.passes_exhausted
        s.requests += 1
        tool = self.tools[block.kind]
        t0 = time.time()
        try:
            with log.activity(f"forward pass {s.passes + 1}/{s.budget}"):
                result = tool.run(block.body, s.requests)
        except PassInvalid as e:
            s.requests -= 1
            self._log(kind=block.kind, request=block.body, error=str(e))
            self._did = f"invalid forward request, not counted: {str(e)[:120]}"
            return self.messages.pass_invalid.format(error=e)
        except PassFailed as e:
            self._log(kind=block.kind, request=block.body, error=str(e))
            self._did = f"forward pass failed, not counted: {str(e)[:120]}"
            return self.messages.pass_failed.format(error=e)
        s.passes += 1
        shown = _short_result(result.summary)
        self._did = f"forward pass {s.passes}/{s.budget}{': ' + shown if shown else ''}, {log.duration(time.time() - t0)}"
        summary = result.summary if isinstance(result.summary, str) else json.dumps(result.summary, indent=1)
        out = self.messages.pass_done.format(k=s.passes, budget=s.budget, kind=result.kind, where=result.where,
                                             summary=summary, left=s.budget - s.passes)
        digest = None
        if self.policy.report_edits:
            names, digest = self.kernel.edits_used()
            out += "\n" + self.messages.pass_edits.format(names=", ".join(names) or "none", digest=digest or "none")
            if s.passes == s.budget:
                out += "\n" + self.messages.all_passes_used
        self._log(kind=block.kind, request=block.body, result=summary, edit_hash=digest)
        return out
