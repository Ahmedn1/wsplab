"""The task prompt. One base template per network kind and mode, adapted from the paper's prompts by replacing
task-specific words with {{slots}}; the user's own text is appended as a note.

Control, from least to most:
    PromptBuilder("lm_edit", slots={...}, note="This model really sucks at math")
    builder.replace_section("METHOD.", "METHOD.\\nYour own method...")
    PromptBuilder(template="your whole prompt, with or without {{slots}}")
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from importlib import resources

# The paper's system message: harness/instructions.md, used unchanged for every run.
SYSTEM = "Answer the user's request."

DEFAULT_TASK_SENTENCE = ("Make it better by editing its weights directly: find out what goes wrong inside the network "
                         "when it fails and repair that with targeted changes.")
_SLOT = re.compile(r"\{\{(\w+)\}\}")


def load_template(name: str) -> str:
    return resources.files("wsplab.prompts").joinpath("templates", f"{name}.txt").read_text(encoding="utf-8")


def render(template: str, slots: dict) -> str:
    """Fill {{slots}}. A slot with no value is an error: a prompt must never go out with a hole in it."""
    missing = sorted({m.group(1) for m in _SLOT.finditer(template)} - set(slots))
    if missing:
        raise ValueError(f"prompt slots with no value: {missing}")
    return _SLOT.sub(lambda m: str(slots[m.group(1)]), template)


@dataclass
class PromptBuilder:
    base: str | None = None
    slots: dict = field(default_factory=dict)
    note: str | None = None
    template: str | None = None
    sections: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if (self.base is None) == (self.template is None):
            raise ValueError("give exactly one of base (a built-in template name) and template (your own text)")

    @property
    def fidelity(self) -> str:
        """paper: the base template; custom: the user changed its text. A note or slot values change neither."""
        return "custom" if self.template is not None or self.sections else "paper"

    def replace_section(self, heading: str, text: str) -> PromptBuilder:
        """Replace one section of the template, named by the heading it starts with, such as "METHOD."."""
        self.sections[heading] = text
        return self

    def _text(self) -> str:
        text = self.template if self.template is not None else load_template(self.base)
        for heading, new in self.sections.items():
            parts = re.split(r"\n\n(?=[A-Z][A-Z ]+[.:])", text)
            hits = [i for i, p in enumerate(parts) if p.startswith(heading)]
            if len(hits) != 1:
                raise ValueError(f"no single section starts with {heading!r}")
            parts[hits[0]] = new
            text = "\n\n".join(parts)
        return text

    def build(self) -> str:
        slots = {"task_sentence": DEFAULT_TASK_SENTENCE, **self.slots}
        text = render(self._text(), slots)
        if self.note:
            text = text.rstrip("\n") + f"\n\nNOTE FROM THE USER. {self.note.strip()}\n"
        return text

    def sha256(self) -> str:
        return hashlib.sha256(self.build().encode()).hexdigest()
