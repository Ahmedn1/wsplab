"""Driver behaviour the logged runs never exercised."""

import pytest

from wsplab.editors import EditorRefused, Reply, ScriptedEditor
from wsplab.passes import PassResult
from wsplab.prompts import PromptBuilder, render
from wsplab.session import Policy, Session, parse


class FakeKernel:
    def __init__(self):
        self.ran = []

    def run(self, code):
        self.ran.append(code)
        return "ok"

    def edited(self):
        return []

    def files(self):
        return ["notes.md"]

    def notes(self):
        return "my notes"

    def edits_used(self):
        return [], None


class FakeForward:
    fence = "forward"

    def run(self, body, request_number):
        return PassResult("generate", f"/work/gen_{request_number}.jsonl", {"ok": True})


def block(kind="python", body="1 + 1", why="Because."):
    return f"{why}\n```{kind}\n{body}\n```"


def session(tmp_path, replies, **kw):
    kernel = FakeKernel()
    editor = ScriptedEditor(replies)
    kw.setdefault("budget", 0)
    tools = {"forward": FakeForward()} if kw["budget"] else {}
    return Session("TASK", editor, kernel, tools, folder=tmp_path, **kw), editor, kernel


def test_parse_ignores_fences_of_other_kinds():
    p = parse("Look:\n```text\nnot code\n```\nNow:\n```python\nx = 1\n```\n", {"python"})
    assert [b.body for b in p.blocks] == ["x = 1"]


def test_parse_final_answer_has_no_blocks():
    assert parse("All done, here is why.", {"python"}).blocks == []


def test_final_answer_ends_the_session(tmp_path):
    s, _, kernel = session(tmp_path, [block(), "Finished."])
    r = s.run()
    assert r.final_answer == "Finished." and r.state.done and kernel.ran == ["1 + 1"]
    assert (tmp_path / "turns/t001/prompt.txt").read_text() == "TASK"
    assert (tmp_path / "final_answer.md").read_text() == "Finished."


def test_final_answer_with_passes_left_gets_a_nudge(tmp_path):
    s, editor, _ = session(tmp_path, ["I give up.", block("forward", "{}"), "Done."], budget=1)
    r = s.run()
    assert r.state.nudges == 1 and r.final_answer == "Done."
    assert "Forward passes left: 1" in editor.seen[1][-1]["content"]


def test_two_blocks_are_refused_and_nothing_runs(tmp_path):
    s, editor, kernel = session(tmp_path, ["Both.\n```python\n1\n```\n```python\n2\n```", "Done."])
    s.run()
    assert kernel.ran == [] and "2 blocks" in editor.seen[1][-1]["content"]


def test_unexplained_blocks_rejected_twice_then_run(tmp_path):
    bare = "```python\n1\n```"
    s, _, kernel = session(tmp_path, [bare, bare, bare, "Done."])
    r = s.run()
    assert len(kernel.ran) == 1 and r.state.unexplained == 3


def test_run_limit(tmp_path):
    s, editor, kernel = session(tmp_path, [block(), block(), "Done."], max_runs=1)
    s.run()
    assert len(kernel.ran) == 1 and "No Python runs are left" in editor.seen[2][-1]["content"]


def test_pass_budget_is_enforced(tmp_path):
    s, editor, _ = session(tmp_path, [block("forward", "{}"), block("forward", "{}"), "Done."], budget=1)
    r = s.run()
    assert r.state.passes == 1 and "No forward passes are left" in editor.seen[2][-1]["content"]


def test_handoff_restarts_the_conversation_with_the_notebook(tmp_path):
    s, editor, _ = session(tmp_path, [block(), block(), "Done."], policy=Policy(handoff_every=2))
    s.run()
    last = editor.seen[2]
    assert len(last) == 1
    assert last[0]["content"].startswith("TASK\n\nTHIS IS A RESTARTED CONVERSATION")
    assert "my notes" in last[0]["content"] and "Last result:\nOUTPUT of Python run 2" in last[0]["content"]


def test_refusal_is_recovered_and_gives_up_after_too_many(tmp_path):
    class Refuser:
        name = "refuser"

        def __init__(self, times):
            self.times = times

        def reply(self, system, messages):
            if self.times:
                self.times -= 1
                raise EditorRefused("filtered")
            return Reply("Done.")

    s = Session("TASK", Refuser(1), FakeKernel(), folder=tmp_path / "a")
    assert s.run().final_answer == "Done."
    s = Session("TASK", Refuser(99), FakeKernel(), folder=tmp_path / "b", policy=Policy(max_flags=2))
    assert s.run().state.stop_reason == "refused"


def test_max_turns_stops(tmp_path):
    s, _, _ = session(tmp_path, [block()] * 5, policy=Policy(max_turns=2))
    assert s.run().state.stop_reason == "max_turns"


def test_prompt_needs_every_slot():
    with pytest.raises(ValueError, match="no value"):
        render("Hello {{who}}", {})


def test_note_is_appended_and_is_not_a_custom_prompt():
    b = PromptBuilder(template="Base text.", note="This model really sucks at math")
    assert b.build().endswith("NOTE FROM THE USER. This model really sucks at math\n")


def test_user_can_replace_a_section_or_everything():
    text = "TASK. a\n\nMETHOD.\none\n\nRULES. two"
    b = PromptBuilder(template=text).replace_section("METHOD.", "METHOD.\nmine")
    assert b.build() == "TASK. a\n\nMETHOD.\nmine\n\nRULES. two"
    assert b.fidelity == "custom"
    assert PromptBuilder(template="Mine entirely.").build() == "Mine entirely."
