"""Replay the paper's logged runs through the driver: every message must match the logged prompt byte for byte.

Only the outputs of code and forward passes are read from the logs. The wrappers, counters, handoffs, explanation
checks, notices and refusal handling are all produced by wsplab.
"""

import concurrent.futures as cf
import json
import re
from pathlib import Path

import pytest

from wsplab import upstream
from wsplab.editors import EditorRefused, Reply
from wsplab.passes import PassFailed, PassInvalid, PassResult
from wsplab.prompts import PromptBuilder
from wsplab.session import V3, V5, Policy, Session

P50C = "exp8_math/runs/p50c"
G50B = "exp8_math/gemma_followup/runs/g50b"
PY = re.compile(r"OUTPUT of Python run \d+ of at most \d+:\n```\n(.*)\n```", re.S)
FWD = re.compile(r"Forward pass \d+ of \d+ done \((\w+)\); results in (.*?)\. Summary:\n```\n(.*)\n```\n"
                 r"Forward passes left: \d+\.(?:\nEdits used|\Z|\n\()", re.S)
FAILED = re.compile(r"The forward pass failed and was not counted: (.*)", re.S)
INVALID = re.compile(r"The forward request is not valid \((.*)\)\. Not counted\.", re.S)
HANDOFF = re.compile(r"Tensors currently edited: (\[.*?\])\. Files in /work: (\[.*?\])\.\nYOUR NOTEBOOK \(/work/notes\.md, "
                     r"last 8,000 characters\):\n```\n(.*?)\n```\nContinue from here\.", re.S)


EDITS = re.compile(r"Edits used in this pass: (.*) \(edit hash (\w+)\)\.")


def fetch_run(run, turns):
    paths = [f"{run}/turns/t{t:03d}/{f}" for t in range(1, turns + 1) for f in ("prompt.txt", "answer.md")]
    with cf.ThreadPoolExecutor(16) as pool:
        got = list(pool.map(upstream.fetch, paths))
    prompts = {t: got[2 * (t - 1)].read_text() for t in range(1, turns + 1)}
    answers = {t: got[2 * (t - 1) + 1].read_text() for t in range(1, turns + 1)}
    return prompts, answers


class Log:
    def __init__(self, prompts, answers):
        self.prompts, self.answers = prompts, answers
        self.session = None

    def result_text(self, turn):
        """What the logged prompt of `turn` reports as the last result, without any handoff around it."""
        text = self.prompts.get(turn, "")
        if "\n\nLast result:\n" in text:
            text = text.split("\n\nLast result:\n", 1)[1]
        return text


class ReplayEditor:
    name = "replay"

    def __init__(self, log, refused_turns=()):
        self.log = log
        self.refused = set(refused_turns)
        self.mismatches = []
        self.compared = 0

    def reply(self, system, messages):
        turn = self.log.session.state.turn
        if turn in self.refused:
            self.refused.discard(turn)
            raise EditorRefused("content filter")
        got, want = messages[-1]["content"], self.log.prompts[turn]
        self.compared += 1
        if got != want:
            at = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            self.mismatches.append((turn, at, got[max(0, at - 80):at + 120], want[max(0, at - 80):at + 120]))
        assert system == "Answer the user's request."
        return Reply(self.log.answers[turn])


class ReplayKernel:
    def __init__(self, log):
        self.log = log

    def run(self, code):
        m = PY.search(self.log.result_text(self.log.session.state.turn + 1))
        return m.group(1) if m else "(not delivered)"

    def edits_used(self):
        m = EDITS.search(self.log.result_text(self.log.session.state.turn + 1))
        names = [] if m.group(1) == "none" else m.group(1).split(", ")
        return names, None if m.group(2) == "none" else m.group(2)

    def _handoff(self):
        return HANDOFF.search(self.log.prompts[self.log.session.state.turn])

    def edited(self):
        return eval(self._handoff().group(1))

    def files(self):
        return eval(self._handoff().group(2))

    def notes(self):
        return self._handoff().group(3)


class ReplayForward:
    fence = "forward"

    def __init__(self, log):
        self.log = log

    def run(self, body, request_number):
        text = self.log.result_text(self.log.session.state.turn + 1)
        if m := FWD.search(text):
            return PassResult(m.group(1), m.group(2), m.group(3))
        if m := INVALID.search(text):
            raise PassInvalid(m.group(1))
        if m := FAILED.search(text):
            raise PassFailed(m.group(1))
        raise AssertionError(f"no pass result in the prompt after turn {self.log.session.state.turn}: {text[:200]}")


def replay(prompts, answers, tmp_path, *, budget, messages, policy, refused=(), before=None, after=None):
    log = Log(prompts, answers)
    editor = ReplayEditor(log, refused)
    session = Session(prompts[1], editor, ReplayKernel(log), {"forward": ReplayForward(log)}, budget=budget,
                      folder=tmp_path, messages=messages, policy=policy, before_turn=before, after_reply=after)
    log.session = session
    result = session.run()
    return editor, result


def notice(path):
    return upstream.fetch(f"exp8_math/notices/{path}").read_text()


@pytest.mark.network
def test_p50c_replays_byte_for_byte(tmp_path):
    prompts, answers = fetch_run(P50C, 158)
    message = upstream.fetch(f"{P50C}/message.txt").read_text()

    def after(s):
        if s.state.turn == 28:
            s.policy.explain = "remind"

    def before(s):
        if s.state.turn == 29:
            s.task = message
            s.new_rule(V3.explain)
        elif s.state.turn == 31:
            s.kernel_restarted()
            s.policy.explain = "enforce"
        elif s.state.turn == 154:
            s.kernel_restarted()
            s.policy.report_edits = True
        elif s.state.turn == 46:
            s.notice(notice("001_pass_size.txt").strip(), "001_pass_size.txt")
        elif s.state.turn == 51:
            s.notice(notice("002_tool_facts.txt").strip(), "002_tool_facts.txt")

    editor, result = replay(prompts, answers, tmp_path, budget=50, messages=V3, policy=Policy(explain="off", report_edits=False),
                            refused=[6], before=before, after=after)
    assert editor.mismatches == []
    assert editor.compared == 158
    want = json.loads(upstream.fetch(f"{P50C}/state.json").read_text())
    for key in ("turn", "runs", "passes", "requests", "handoffs", "since_handoff", "flags", "unexplained", "nudges"):
        assert getattr(result.state, key) == want[key], key
    assert result.final_answer == answers[158]


@pytest.mark.network
def test_g50b_replays_byte_for_byte(tmp_path):
    prompts, answers = fetch_run(G50B, 117)

    def before(s):
        if s.state.turn == 25:
            s.notice(notice("004_failure_counts.txt").strip(), "004_failure_counts.txt")

    editor, result = replay(prompts, answers, tmp_path, budget=50, messages=V5, policy=Policy(), before=before)
    assert editor.mismatches == []
    assert editor.compared == 117
    want = json.loads(upstream.fetch(f"{G50B}/state.json").read_text())
    for key in ("turn", "runs", "passes", "requests", "handoffs", "since_handoff", "unexplained", "nudges"):
        assert getattr(result.state, key) == want[key], key


@pytest.mark.network
def test_base_template_renders_the_authors_prompt_exactly():
    slots = json.loads((Path(__file__).parent / "gemma_v5_slots.json").read_text())
    built = PromptBuilder("lm_edit", slots=slots).build()
    assert built == upstream.fetch(f"{G50B}/turns/t001/prompt.txt").read_text()
