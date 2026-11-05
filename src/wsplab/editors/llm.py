"""LLM editors. Plain chat completion, since the protocol needs no tool calling: any chat model can edit weights."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from .base import EditorRefused, Reply, ScriptedEditor, Usage


class OpenAIEditor:
    """Any OpenAI-compatible server: `base_url` covers OpenRouter, vLLM, Ollama and others."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, **params):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "none"))
        self.model = model
        self.params = params
        self.name = f"openai:{model}"

    def reply(self, system: str, messages: list[dict]) -> Reply:
        from openai import BadRequestError

        try:
            r = self.client.chat.completions.create(model=self.model, **self.params,
                                                    messages=[{"role": "system", "content": system}, *messages])
        except BadRequestError as e:
            if "content" in str(e).lower() and ("policy" in str(e).lower() or "filter" in str(e).lower()):
                raise EditorRefused(str(e)) from e
            raise
        if r.choices[0].finish_reason == "content_filter":
            raise EditorRefused("content filter")
        usage = r.usage
        return Reply(r.choices[0].message.content or "",
                     Usage(usage.prompt_tokens, usage.completion_tokens) if usage else Usage())


class AnthropicEditor:
    def __init__(self, model: str, max_tokens: int = 32000, api_key: str | None = None, **params):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.params = params
        self.name = f"anthropic:{model}"

    def reply(self, system: str, messages: list[dict]) -> Reply:
        with self.client.messages.stream(model=self.model, max_tokens=self.max_tokens, system=system, messages=messages,
                                         **self.params) as stream:
            r = stream.get_final_message()
        if r.stop_reason == "refusal":
            raise EditorRefused("refused by the model")
        text = "".join(b.text for b in r.content if b.type == "text")
        return Reply(text, Usage(r.usage.input_tokens, r.usage.output_tokens,
                                 getattr(r.usage, "cache_read_input_tokens", 0) or 0))


class CodexEditor:
    """The paper's route: the Codex CLI, one call per turn, with the conversation resumed. Pass its no-tools configuration
    through `args`; each call is sent as one prompt on stdin."""

    def __init__(self, model: str, effort: str = "high", binary: str = "codex", args: tuple = ()):
        self.model = model
        self.effort = effort
        self.binary = binary
        self.args = tuple(args)
        self.name = f"codex:{model}"
        self.thread: str | None = None
        self.seen = 0

    def reply(self, system: str, messages: list[dict]) -> Reply:
        if len(messages) <= self.seen:
            self.thread = None
        self.seen = len(messages)
        command = [self.binary, "exec"]
        if self.thread:
            command += ["resume", self.thread]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "last.txt"
            command += ["--json", "-m", self.model, "-c", f"model_reasoning_effort={self.effort}",
                        "--output-last-message", str(out), *self.args, "-"]
            done = subprocess.run(command, input=messages[-1]["content"], capture_output=True, text=True)
            for line in done.stdout.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.thread = event.get("thread_id", self.thread)
                if event.get("type") == "error" and "flagged" in str(event.get("message", "")):
                    raise EditorRefused(event["message"])
            if done.returncode:
                raise RuntimeError(f"codex failed: {done.stderr[-2000:]}")
            return Reply(out.read_text())


def get_editor(spec):
    """An editor from a spec: an object with .reply, "openai:<model>", "anthropic:<model>", "codex:<model>", or
    "script:<file.json>" with a list of replies."""
    if hasattr(spec, "reply"):
        return spec
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError("give `editor`: for example anthropic:<model>, openai:<model> or an Editor object")
    provider, _, rest = spec.partition(":")
    if provider == "openai":
        return OpenAIEditor(rest)
    if provider == "anthropic":
        return AnthropicEditor(rest)
    if provider == "codex":
        return CodexEditor(rest)
    if provider == "script":
        return ScriptedEditor(json.loads(Path(rest).read_text()))
    raise ValueError(f"unknown editor provider {provider!r}")
