"""The provider adapters with their network calls stubbed. Real calls are for live runs."""

import inspect
from types import SimpleNamespace as NS

import pytest

from wsplab.editors import EditorRefused, Metered, Reply, ScriptedEditor, Spending, get_editor
from wsplab.editors.claude import AnthropicEditor
from wsplab.editors.llm import ChatEditor, OpenAIEditor
from wsplab.session import Session


class Stream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


def claude(stop_reason="end_turn", text="Hello", **usage):
    numbers = dict(input_tokens=10, output_tokens=5, cache_creation_input_tokens=100, cache_read_input_tokens=1000)
    return NS(stop_reason=stop_reason, stop_details=None, usage=NS(**{**numbers, **usage}),
              content=[NS(type="thinking", thinking=""), NS(type="text", text=text)])


def stubbed_claude(message, monkeypatch):
    pytest.importorskip("anthropic")
    editor = AnthropicEditor("claude-opus-5-5", api_key="test")
    seen = {}

    def stream(**kw):
        seen.update(kw)
        return Stream(message)

    monkeypatch.setattr(editor.client.messages, "stream", stream)
    return editor, seen


def test_claude_request_sets_effort_and_caching(monkeypatch):
    editor, seen = stubbed_claude(claude(), monkeypatch)
    r = editor.reply("Answer the user's request.", [{"role": "user", "content": "hi"}])
    assert r.text == "Hello" and not r.truncated
    assert seen["output_config"] == {"effort": "high"} and seen["cache_control"] == {"type": "ephemeral"}
    assert seen["system"] == "Answer the user's request." and seen["max_tokens"] == 32768


def test_claude_parameters_exist_in_the_installed_sdk():
    pytest.importorskip("anthropic")
    params = inspect.signature(AnthropicEditor("m", api_key="k").client.messages.stream).parameters
    assert {"output_config", "cache_control", "system", "max_tokens"} <= set(params)


def test_claude_usage_counts_cached_input_towards_the_limit(monkeypatch):
    editor, _ = stubbed_claude(claude(), monkeypatch)
    usage = editor.reply("s", [{"role": "user", "content": "x"}]).usage
    assert usage.input_tokens == 1110 and usage.cached_tokens == 1000 and usage.output_tokens == 5


def test_claude_refusal_and_truncation(monkeypatch):
    editor, _ = stubbed_claude(claude("refusal"), monkeypatch)
    with pytest.raises(EditorRefused):
        editor.reply("s", [{"role": "user", "content": "x"}])
    editor, _ = stubbed_claude(claude("max_tokens"), monkeypatch)
    assert editor.reply("s", [{"role": "user", "content": "x"}]).truncated


def openai_reply(finish="stop"):
    return NS(choices=[NS(finish_reason=finish, message=NS(content="Hello"))], usage=NS(prompt_tokens=7, completion_tokens=3))


def stubbed_openai(reply, monkeypatch):
    pytest.importorskip("openai")
    editor = ChatEditor("m", api_key="k")
    seen = {}

    def create(**kw):
        seen.update(kw)
        return reply

    monkeypatch.setattr(editor.client.chat.completions, "create", create)
    return editor, seen


def response(*texts, reason=None, cached=2):
    items = [NS(type="reasoning")] + [NS(type="message", content=[NS(type="output_text", text=t)]) for t in texts]
    return NS(output=items, incomplete_details=NS(reason=reason) if reason else None,
              usage=NS(input_tokens=10, output_tokens=4, input_tokens_details=NS(cached_tokens=cached)))


def stubbed_responses(reply, monkeypatch):
    pytest.importorskip("openai")
    editor = OpenAIEditor("m", api_key="k")
    seen = {}

    def create(**kw):
        seen.update(kw)
        return reply

    monkeypatch.setattr(editor.client.responses, "create", create)
    return editor, seen


def test_commentary_and_final_answer_are_joined_so_the_block_starts_a_line(monkeypatch):
    """Found on a live run: a reply arrives as a commentary message, then a final answer holding the block."""
    from wsplab.session import parse

    editor, seen = stubbed_responses(response("I will look at the weights.", "```python\nprint(1)\n```"), monkeypatch)
    r = editor.reply("sys", [{"role": "user", "content": "x"}])
    parsed = parse(r.text, {"python"})
    assert parsed.explanation == "I will look at the weights." and parsed.blocks[0].body == "print(1)"
    assert seen["instructions"] == "sys" and seen["reasoning"] == {"effort": "high"}
    assert seen["max_output_tokens"] == 32768
    assert r.usage.input_tokens == 10 and r.usage.cached_tokens == 2


def test_responses_truncation_and_refusal(monkeypatch):
    editor, _ = stubbed_responses(response("cut", reason="max_output_tokens"), monkeypatch)
    assert editor.reply("s", []).truncated
    editor, _ = stubbed_responses(response("x", reason="content_filter"), monkeypatch)
    with pytest.raises(EditorRefused):
        editor.reply("s", [])
    refused = NS(output=[NS(type="message", content=[NS(type="refusal", refusal="no")])], incomplete_details=None, usage=None)
    editor, _ = stubbed_responses(refused, monkeypatch)
    with pytest.raises(EditorRefused):
        editor.reply("s", [])


def test_openai_request_and_usage(monkeypatch):
    editor, seen = stubbed_openai(openai_reply(), monkeypatch)
    r = editor.reply("sys", [{"role": "user", "content": "x"}])
    assert seen["messages"][0] == {"role": "system", "content": "sys"}
    assert r.text == "Hello" and r.usage.input_tokens == 7


def test_openai_filter_and_truncation(monkeypatch):
    editor, _ = stubbed_openai(openai_reply("content_filter"), monkeypatch)
    with pytest.raises(EditorRefused):
        editor.reply("s", [])
    editor, _ = stubbed_openai(openai_reply("length"), monkeypatch)
    assert editor.reply("s", []).truncated


class Kernel:
    def __init__(self):
        self.ran = []

    def run(self, code):
        self.ran.append(code)
        return "ok"

    def edited(self):
        return []

    def files(self):
        return []

    def notes(self):
        return ""

    def edits_used(self):
        return [], None


def test_a_truncated_reply_never_runs_its_half_written_block(tmp_path):
    class Cut(ScriptedEditor):
        def reply(self, system, messages):
            r = super().reply(system, messages)
            return Reply(r.text, truncated=not self.seen[1:])

    kernel = Kernel()
    editor = Cut(["Explaining.\n```python\nsave_edit('w', load('w') *", "Done."])
    Session("TASK", editor, kernel, folder=tmp_path).run()
    assert kernel.ran == []
    assert "cut off" in editor.seen[1][-1]["content"]


def test_cached_input_at_its_own_price():
    class Paid:
        name = "paid"

        def reply(self, system, messages):
            from wsplab.editors import Usage

            return Reply("x", Usage(1_000_000, 1_000_000, 900_000))

    spending = Spending(max_usd=100.0, prices={"paid": (10.0, 50.0, 1.0)})
    Metered(Paid(), spending).reply("s", [])
    assert spending.usd == pytest.approx(1.0 + 0.9 + 50.0)


def test_openai_reports_cached_tokens(monkeypatch):
    reply = openai_reply()
    reply.usage.prompt_tokens_details = NS(cached_tokens=5)
    editor, _ = stubbed_openai(reply, monkeypatch)
    assert editor.reply("s", []).usage.cached_tokens == 5


def test_cached_input_counts_against_a_spending_limit():
    class Paid:
        name = "paid"

        def reply(self, system, messages):
            from wsplab.editors import Usage

            return Reply("x", Usage(1_000_000, 0, 999_000))

    spending = Spending(max_usd=100.0, prices={"paid": (4.0, 20.0)})
    Metered(Paid(), spending).reply("s", [])
    assert spending.usd == 4.0


def test_editor_specs():
    with pytest.raises(ValueError, match="provider"):
        get_editor("nope:model")
    with pytest.raises(ValueError, match="give `editor`"):
        get_editor(None)
