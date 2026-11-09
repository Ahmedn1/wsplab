"""What a language model is practised and judged on. A suite is data: rows of {id, problem, answer}, a grader, sampling
settings and a few words for the prompt. Graded the same way for passes and for the final evaluation."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .graders import INSTRUCTION, Grader, get_grader
from .suites import Scores


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class LMSuite:
    """`practice` rows are shown to the editor; `evaluation` rows never are. `samples` is answers per evaluation item."""

    practice: list[dict]
    evaluation: list[dict]
    grader: Grader
    sampling: dict
    benchmark: str = "held-out questions"
    samples: int = 1
    user_message: Callable[[dict], str] = field(default=lambda row: row["problem"])
    protocol: str | None = None
    practice_text: str | None = None
    check_overlap: bool = True

    def __post_init__(self):
        if not self.check_overlap:
            return
        seen = {str(r["problem"]) for r in self.practice}
        overlap = [r["id"] for r in self.evaluation if str(r["problem"]) in seen]
        if overlap:
            raise ValueError(f"{len(overlap)} items are in both the practice data and the evaluation: {overlap[:5]}")

    def export(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "practice.jsonl", "w") as f:
            for row in self.practice:
                f.write(json.dumps(row) + "\n")

    def describe_practice(self) -> str:
        return self.practice_text or (f"Practice problems (none is in the evaluation), one JSON object per line with id, "
                                      f"problem and answer: /practice/practice.jsonl ({len(self.practice):,}).")

    def describe_protocol(self, thinking: bool) -> str:
        if self.protocol:
            return self.protocol
        s = self.sampling
        parts = [f"{k.replace('_', ' ')} {s[k]}" for k in ("temperature", "top_p", "top_k", "presence_penalty") if k in s]
        template = "the chat template with thinking enabled" if thinking else "the model's chat template"
        return f"{template}; sampling with {', '.join(parts)} and at most {s['max_tokens']:,} new tokens"


def custom_suite(practice: str, evaluation: str, *, grader="exact", sampling: dict | None = None, **options) -> LMSuite:
    """Your own data, as .jsonl files of {"id", "problem", "answer"}."""
    return LMSuite(read_jsonl(practice), read_jsonl(evaluation), get_grader(grader),
                   sampling or {"temperature": 0.7, "top_p": 0.8, "max_tokens": 2048}, **options)


MATH_PRACTICE = ("gsm8k_train.jsonl", "math_train.jsonl")


def math_suite(sampling: dict | None = None, *, samples: int = 4, limit: int | None = None, lenient: bool = False,
               practice_limit: int | None = None) -> LMSuite:
    """The paper's math task. Practice comes from its release; MATH-500 is downloaded, and never reaches the editor."""
    from datasets import load_dataset

    from ..upstream import fetch

    rows: list[dict] = []
    for name in MATH_PRACTICE:
        rows += read_jsonl(fetch(f"exp8_math/practice/{name}"))
    if practice_limit:
        rows = rows[:practice_limit]
    data = load_dataset("HuggingFaceH4/MATH-500", split="test")
    evaluation = [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"], "source": "math500"} for r in data]
    if limit:
        evaluation = evaluation[:limit]
    return LMSuite(rows, evaluation, get_grader("math-lenient" if lenient else "math"),
                   sampling or {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768},
                   benchmark="MATH-500", samples=samples, user_message=lambda row: f"{row['problem']}\n{INSTRUCTION}")


class LMEvaluation:
    """Scores a language model on a suite's evaluation rows, or on any rows given."""

    def __init__(self, runner, suite: LMSuite, rows: list[dict] | None = None, samples: int | None = None):
        self.runner = runner
        self.suite = suite
        self.rows = suite.evaluation if rows is None else rows
        self.samples = samples or suite.samples
        self.name = suite.benchmark

    def evaluate(self, weights: Path) -> Scores:
        from ..passes.forward import item_seed, summarize
        from ..runners.base import Prompt

        prompts = [Prompt(r["id"], self.runner.render(self.suite.user_message(r)), item_seed(r["id"])) for r in self.rows]
        outputs = self.runner.generate(weights, prompts, self.samples, self.suite.sampling)
        per_item: dict[str, list[float]] = {}
        records = []
        for row, samples in zip(self.rows, outputs):
            grades = [self.suite.grader.grade(g.text, row) for g in samples]
            per_item[row["id"]] = [float(x.correct) for x in grades]
            records += [{"correct": x.correct, "pred": x.pred, "finish": g.finish_reason, "tokens": g.tokens,
                         "source": row.get("source", "other")} for x, g in zip(grades, samples)]
        return Scores(per_item, summarize(records))
