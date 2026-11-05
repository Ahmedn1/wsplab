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
DEFAULT_READ_SENTENCE = "What does this network compute, and how?"
_SLOT = re.compile(r"\{\{(\w+)\}\}")


def _read(*path: str) -> str:
    return resources.files("wsplab.prompts").joinpath(*path).read_text(encoding="utf-8")


def _part(name: str) -> str:
    return _read("parts", f"{name}.txt").rstrip("\n")


def split_sections(text: str) -> list[str]:
    return re.split(r"\n\n(?=[A-Z][A-Z ]+[.:])", text)


def _swap(text: str, replacements: dict[str, str | None]) -> str:
    """Replace the sections that start with each heading; None removes the section."""
    out = []
    for section in split_sections(text):
        heading = next((h for h in replacements if section.startswith(h)), None)
        if heading is None:
            out.append(section)
        elif replacements[heading] is not None:
            out.append(replacements[heading])
    return "\n\n".join(out)


def load_template(name: str, *, passes: bool = True) -> str:
    """A base template. Every variant is the paper's prompt with only the sections that must differ replaced, so EXPLAIN,
    RULES and the rest stay the same text everywhere: `lm_edit` is the paper's own, and `generic_*` or `*_read` swap the
    sections about tools or about what is being asked. `inline_*` have no tools, so they are their own short text.
    `passes=False` removes everything about forward passes, for a session that may not run the network."""
    if name in ("inline_edit", "inline_read"):
        return _part(name)
    base = _read("templates", "lm_edit.txt")
    kind, mode = name.split("_")
    swaps: dict[str, str | None] = {}
    if kind == "generic":
        swaps.update({"TOOLS.": _part("tools_generic"), "METHOD.": _part("method_generic")})
    if mode == "read":
        swaps.update({"TASK.": _part("task_read"), "EVALUATION.": None, "SCOPE.": None, "METHOD.": _part("method_read"),
                      "RULES.": _part("rules_read"), "HOW THE SESSION RUNS.": _part("session_read")})
    elif mode == "write":
        swaps.update({"TASK.": _part("task_write"), "SCOPE.": None, "HOW THE SESSION RUNS.": _part("session_write")})
    elif mode != "edit":
        raise ValueError(f"no template {name!r}")
    if not passes:
        swaps["METHOD."] = _part("method_read") if mode == "read" else _part("method_nopass")
        if mode != "read":
            swaps["HOW THE SESSION RUNS."] = _part("session_nopass")
    text = _swap(base, swaps) if swaps else base
    if not passes:
        text = re.sub(r"\n3\. A forward-pass tool.*?(?=\n\n[A-Z])", "", text, flags=re.S)
    if mode == "read":
        text = (text.replace(", save_edit(name, tensor), drop_edit(name), edited()", "")
                .replace("save_edit(name, tensor), drop_edit(name), edited(), ", "")
                .replace("summarize the path that led to your edits", "summarize the path that led to your answer"))
    return text


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
    passes: bool = True

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
        text = self.template if self.template is not None else load_template(self.base, passes=self.passes)
        for heading, new in self.sections.items():
            parts = split_sections(text)
            hits = [i for i, p in enumerate(parts) if p.startswith(heading)]
            if len(hits) != 1:
                raise ValueError(f"no single section starts with {heading!r}")
            parts[hits[0]] = new
            text = "\n\n".join(parts)
        return text

    def build(self) -> str:
        default = DEFAULT_READ_SENTENCE if self.base and self.base.endswith("_read") else DEFAULT_TASK_SENTENCE
        slots = {"task_sentence": default, **self.slots}
        template = self._text()
        if slots.get("out_of_scope") is None:
            template = template.replace("{{out_of_scope}} (save_edit refuses them), ", "")
            slots.pop("out_of_scope", None)
        text = render(template, slots)
        if self.note:
            text = text.rstrip("\n") + f"\n\nNOTE FROM THE USER. {self.note.strip()}\n"
        return text

    def sha256(self) -> str:
        return hashlib.sha256(self.build().encode()).hexdigest()
