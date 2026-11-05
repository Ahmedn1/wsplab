"""Grading answers. MathGrader's extraction and checking come from exp8_math/code/mathqa.py and evaluate.py (MIT), whose
hendrycks/math helpers are MIT too; the rest is new. A grader takes a response and a data row and returns a Grade."""

from __future__ import annotations

import functools
import re
import threading
from dataclasses import dataclass
from typing import Callable, Protocol

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


@dataclass
class Grade:
    correct: bool
    pred: str | None = None

    @property
    def answered(self) -> bool:
        return self.pred is not None


class Grader(Protocol):
    def grade(self, text: str, row: dict) -> Grade: ...


def answer_text(text: str) -> str:
    """The text the extraction reads: a stray reasoning block cut off (up to the last '</think>')."""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def _box_at(text: str, idx: int) -> str | None:
    """The content of the box that starts at text[idx] ('\\boxed' or '\\fbox'), or None."""
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key):]
    r = rest.lstrip()
    if r.startswith("{"):
        depth = 0
        for k, ch in enumerate(r):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return r[1:k].strip() or None
        return None
    if key == "\\boxed" and rest.startswith(" "):
        return rest.strip().split("$")[0].split("\n")[0].strip() or None
    return None


def last_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _box_at(text, idx)


def extract(text: str) -> str | None:
    return last_boxed(answer_text(text))


def _timeout() -> int | None:
    """math-verify's timeouts use signal.alarm, which works only in the main thread."""
    return 5 if threading.current_thread() is threading.main_thread() else None


@functools.lru_cache(maxsize=1 << 16)
def _parsed(s: str, timeout: int | None):
    import math_verify

    return math_verify.parse("\\boxed{" + s + "}", parsing_timeout=timeout)


def is_correct(pred: str | None, gold: str | None) -> bool:
    """math-verify equivalence of two answers; False for no answer or an answer that cannot be parsed."""
    if pred is None or gold is None:
        return False
    import math_verify

    timeout = _timeout()
    g, p = _parsed(gold, timeout), _parsed(pred, timeout)
    if not g or not p:
        return False
    try:
        return bool(math_verify.verify(g, p, timeout_seconds=timeout))
    except Exception:  # noqa: BLE001 - an answer math-verify cannot compare is a wrong answer
        return False


def lenient_pred(text: str) -> str | None:
    """An answer given without \\boxed{}: the latest phrase after 'Final Answer', 'the answer is' or 'Answer:' that
    contains a number or a formula, else the last number."""
    found = []
    for pat in (r"[Ff]inal [Aa]nswer\s*(?:is)?\s*[:=]?\s*\$*([^\n$]+)", r"[Tt]he (?:final )?answer is\s*[:=]?\s*\$*([^\n$]+)",
                r"\b[Aa]nswer\s*[:=]\s*\$*([^\n$]+)"):
        found += [(m.start(1), m.group(1).strip().strip("*").rstrip(".").strip()) for m in re.finditer(pat, text)]
    for _, t in sorted(found, reverse=True):
        if re.search(r"[0-9\\]", t):
            return t
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


class MathGrader:
    """The last \\boxed{} answer, checked with math-verify. `lenient` also accepts an answer given without a box."""

    def __init__(self, lenient: bool = False):
        self.lenient = lenient

    def grade(self, text: str, row: dict) -> Grade:
        pred = extract(text)
        gold = row["answer"]
        if pred is not None or not self.lenient:
            return Grade(is_correct(pred, gold), pred)
        loose = lenient_pred(text)
        if loose is None:
            return Grade(False, None)
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", loose)
        good = is_correct(loose, gold) or bool(nums and is_correct(nums[-1].replace(",", ""), gold))
        return Grade(good, loose)


class ExactGrader:
    """The whole response equals the answer, ignoring case and surrounding space."""

    def grade(self, text: str, row: dict) -> Grade:
        pred = answer_text(text).strip()
        return Grade(pred.casefold() == str(row["answer"]).strip().casefold(), pred or None)


class RegexGrader:
    """The pattern's first group, or whole match, in the response equals the answer."""

    def __init__(self, pattern: str):
        self.pattern = re.compile(pattern)

    def grade(self, text: str, row: dict) -> Grade:
        matches = list(self.pattern.finditer(answer_text(text)))
        if not matches:
            return Grade(False, None)
        m = matches[-1]
        pred = m.group(1) if m.groups() else m.group(0)
        return Grade(pred.strip() == str(row["answer"]).strip(), pred)


class ChoiceGrader:
    """Multiple choice: the last 'answer is (X)' or lone letter, compared with the answer letter."""

    def __init__(self, letters: str = "ABCDEFGHIJ"):
        self.letters = letters
        self.stated = re.compile(rf"[Aa]nswer(?: is|:)?\s*\(?([{letters}])\)?\b")
        self.alone = re.compile(rf"(?:^|\s)\(?([{letters}])\)?[.\s]*$")

    def grade(self, text: str, row: dict) -> Grade:
        t = answer_text(text)
        found = self.stated.findall(t) or self.alone.findall(t)
        pred = found[-1] if found else None
        return Grade(pred == str(row["answer"]).strip(), pred)


class CallableGrader:
    """Your own function: f(text, row) -> bool, or a Grade."""

    def __init__(self, fn: Callable):
        self.fn = fn

    def grade(self, text: str, row: dict) -> Grade:
        out = self.fn(text, row)
        return out if isinstance(out, Grade) else Grade(bool(out), text)


def get_grader(grader) -> Grader:
    if not isinstance(grader, str):
        return CallableGrader(grader) if callable(grader) and not hasattr(grader, "grade") else grader
    named = {"math": MathGrader, "math-lenient": lambda: MathGrader(lenient=True), "exact": ExactGrader,
             "choice": ChoiceGrader}
    if grader in named:
        return named[grader]()
    if grader.startswith("regex:"):
        return RegexGrader(grader[len("regex:"):])
    raise ValueError(f"unknown grader {grader!r}; use one of {sorted(named)}, regex:<pattern>, or a function")
