"""Claude as the editor, through the Anthropic SDK. Plain text replies, no tool use: the session protocol needs none.

Streaming avoids HTTP timeouts on long replies. A session's history only ever grows, so prompt caching pays for itself from
the second turn. Effort is set explicitly because editing weights is hard reasoning: the paper ran at its top effort levels,
while some models default lower.

Server-side refusal fallbacks are off on purpose: they would quietly hand a turn to a different model, and a patch records
which editor made it. Refusals are handled by the session instead, with the paper's own recovery message.
"""

from __future__ import annotations

from .base import EditorRefused, Reply, Usage


class AnthropicEditor:
    def __init__(self, model: str, *, effort: str = "high", max_tokens: int = 64000, api_key: str | None = None,
                 max_retries: int = 5, **params):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key, max_retries=max_retries)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.params = params
        self.name = f"anthropic:{model}"

    def reply(self, system: str, messages: list[dict]) -> Reply:
        with self.client.messages.stream(model=self.model, max_tokens=self.max_tokens, system=system, messages=messages,
                                         output_config={"effort": self.effort}, cache_control={"type": "ephemeral"},
                                         **self.params) as stream:
            r = stream.get_final_message()
        if r.stop_reason == "refusal":
            raise EditorRefused(str(getattr(r, "stop_details", None) or "refused"))
        text = "".join(b.text for b in r.content if b.type == "text")
        u = r.usage
        written = getattr(u, "cache_creation_input_tokens", 0) or 0
        cached = getattr(u, "cache_read_input_tokens", 0) or 0
        # input_tokens covers only what was neither read from nor written to the cache, so a spending limit needs all three.
        return Reply(text, Usage(u.input_tokens + written + cached, u.output_tokens, cached),
                     truncated=r.stop_reason == "max_tokens")
