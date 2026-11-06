"""LLM editors. Plain chat completion, since the protocol needs no tool calling: any chat model can edit weights."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from .base import EditorRefused, Reply, ScriptedEditor, Usage
from .claude import AnthropicEditor


class OpenAIEditor:
    """OpenAI's own models, through the Responses API. Recent models answer in several messages: a commentary, then a final
    answer holding the block. Chat Completions shows only part of that, so a reply can arrive with no block at all.

    `effort` sets the reasoning effort. Each message is joined on its own line, so a block still starts one."""

    def __init__(self, model: str, *, effort: str | None = "high", api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 3600, max_retries: int = 5, max_output_tokens: int = 32768, **params):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY"), timeout=timeout,
                             max_retries=max_retries)
        self.model = model
        # Spending is checked before a call, so this bounds what one call can cost. A reply has been seen to reason for 128,000
        # tokens and stop with nothing to show for it.
        self.params = {"max_output_tokens": max_output_tokens, **params, **({"reasoning": {"effort": effort}} if effort else {})}
        self.name = f"openai:{model}"

    def reply(self, system: str, messages: list[dict]) -> Reply:
        from openai import BadRequestError

        try:
            r = self.client.responses.create(model=self.model, instructions=system, input=messages, **self.params)
        except BadRequestError as e:
            if "policy" in str(e).lower() or "flagged" in str(e).lower():
                raise EditorRefused(str(e)) from e
            raise
        parts: list[str] = []
        for item in r.output:
            if item.type != "message":
                continue
            for block in item.content:
                if block.type == "refusal":
                    raise EditorRefused(block.refusal)
                if block.type == "output_text":
                    parts.append(block.text)
        reason = getattr(getattr(r, "incomplete_details", None), "reason", None)
        if reason == "content_filter":
            raise EditorRefused("content filter")
        u = r.usage
        details = getattr(u, "input_tokens_details", None)
        usage = Usage(u.input_tokens, u.output_tokens, getattr(details, "cached_tokens", 0) or 0) if u else Usage()
        return Reply("\n\n".join(p.strip("\n") for p in parts), usage, truncated=reason == "max_output_tokens")


class ChatEditor:
    """Any OpenAI-compatible Chat Completions server: `base_url` covers OpenRouter, vLLM, Ollama and others."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, timeout: float = 3600,
                 max_retries: int = 5, **params):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "none"), timeout=timeout,
                             max_retries=max_retries)
        self.model = model
        self.params = params
        self.name = f"openai-chat:{model}"

    def reply(self, system: str, messages: list[dict]) -> Reply:
        from openai import BadRequestError

        try:
            r = self.client.chat.completions.create(model=self.model, **self.params,
                                                    messages=[{"role": "system", "content": system}, *messages])
        except BadRequestError as e:
            if "content" in str(e).lower() and ("policy" in str(e).lower() or "filter" in str(e).lower()):
                raise EditorRefused(str(e)) from e
            raise
        finish = r.choices[0].finish_reason
        if finish == "content_filter":
            raise EditorRefused("content filter")
        u = r.usage
        details = getattr(u, "prompt_tokens_details", None)
        # Some servers leave thinking out of completion_tokens but bill it as output, and count it in total_tokens.
        output = max(u.completion_tokens, (getattr(u, "total_tokens", 0) or 0) - u.prompt_tokens) if u else 0
        usage = Usage(u.prompt_tokens, output, getattr(details, "cached_tokens", 0) or 0) if u else Usage()
        return Reply(r.choices[0].message.content or "", usage, truncated=finish == "length")


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
    """An editor from a spec: an object with .reply, "openai:<model>", "openai-chat:<model>" for a compatible server (set
    OPENAI_BASE_URL), "anthropic:<model>", "codex:<model>", or "script:<file.json>" with a list of replies."""
    if hasattr(spec, "reply"):
        return spec
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError("give `editor`: for example anthropic:<model>, openai:<model> or an Editor object")
    provider, _, rest = spec.partition(":")
    if provider == "openai":
        return OpenAIEditor(rest)
    if provider == "openai-chat":
        return ChatEditor(rest)
    if provider == "anthropic":
        return AnthropicEditor(rest)
    if provider == "codex":
        return CodexEditor(rest)
    if provider == "script":
        return ScriptedEditor(json.loads(Path(rest).read_text()))
    raise ValueError(f"unknown editor provider {provider!r}")
