"""The verdict on an edit, reached outside the editor. The prompts make an editor submit its best edit even when it
proved nothing, so what counts is measured here, on data the editor never saw."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from ..io.safetensors import Checkpoint
from ..patch.format import Patch
from .stats import Paired

CONFIRMED, NOT_SIGNIFICANT, REGRESSED, NO_FIX, UNMEASURED = ("confirmed", "not_significant", "regressed", "no_fix_found",
                                                             "unmeasured")


@dataclass
class Gate:
    verdict: str
    reasons: list[str] = field(default_factory=list)


def decide(main: Paired | None, *, changed_entries: int, control: Paired | None = None,
           regressions: dict[str, Paired] | None = None, z: float = 2.0) -> Gate:
    """`control` compares a random edit of the same size with the real edit: (random, edited)."""
    if changed_entries == 0:
        return Gate(NO_FIX, ["no weight differs from the base model after rounding"])
    if main is None:
        return Gate(UNMEASURED, ["no evaluation was run"])
    reasons = [f"edited - original = {main.difference:+.4f} +/- {main.standard_error:.4f} over {main.items} items"]
    worse = {k: p for k, p in (regressions or {}).items() if p.z() <= -z}
    if main.z() <= -z or worse:
        reasons += [f"{k} dropped by {p.difference:+.4f} +/- {p.standard_error:.4f}" for k, p in worse.items()]
        return Gate(REGRESSED, reasons)
    if main.z() < z:
        return Gate(NOT_SIGNIFICANT, reasons)
    if control is not None:
        reasons.append(f"edited - random control = {control.difference:+.4f} +/- {control.standard_error:.4f}")
        if control.z() < z:
            reasons.append("a random edit of the same size does about as well")
            return Gate(NOT_SIGNIFICANT, reasons)
    return Gate(CONFIRMED, reasons)


def random_control(base: Checkpoint, patch: Patch, edited: Checkpoint, seed: int = 0) -> dict[str, np.ndarray]:
    """An edit of the same size that means nothing: each tensor's changes moved to random places with random signs, so
    the number of entries touched and the size of each change are the same."""
    rng = np.random.default_rng(seed)
    out = {}
    for record in patch.tensors:
        name = record["name"]
        original = base.float(name)
        delta = (edited.float(name) - original).reshape(-1)
        shuffled = rng.permutation(delta) * rng.choice([-1.0, 1.0], size=delta.size)
        out[name] = (original.reshape(-1) + shuffled).reshape(original.shape).astype(original.dtype)
    return out


def write_report(path: str | os.PathLike, report: dict) -> None:
    lines = [f"# wsplab run: {report['verdict']}", ""]
    for key, label in (("baseline", "Baseline"), ("final", "Edited")):
        if report.get(key) is not None:
            lines.append(f"- {label}: {report[key]:.4f}")
    for reason in report.get("reasons", []):
        lines.append(f"- {reason}")
    if report.get("patch"):
        lines += ["", f"Patch: `{report['patch']}`"]
    if report.get("answer"):
        lines += ["", "## Editor's final answer", "", report["answer"]]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
