"""Shared by the paper's experiments. Everything comes from the authors' release at a pinned commit, fetched on demand."""

from wsplab.prompts import PromptBuilder
from wsplab.upstream import fetch


def exact_prompt(path: str) -> PromptBuilder:
    """The authors' prompt, unchanged."""
    return PromptBuilder(template=fetch(path).read_text(), exact=True)


def section(answer: str, heading: str) -> str:
    """The first line of text under `## heading` or after `heading:` in a reply."""
    lines = answer.splitlines()
    for i, line in enumerate(lines):
        low = line.strip().lower().lstrip("#* ").rstrip("*")
        if low.startswith(heading.lower()):
            rest = line.split(":", 1)[1].strip(" *") if ":" in line else ""
            if rest:
                return rest
            for after in lines[i + 1:]:
                if after.strip():
                    return after.strip()
    return ""
