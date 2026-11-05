"""The whole job in one call: load the network, score it, let the editor work, build the result, score it again, judge."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import patch as P
from .editors import Editor, get_editor
from .eval.gate import UNMEASURED, Gate, decide, random_control, write_report
from .eval.stats import Paired, paired
from .eval.suites import Scores, check_disjoint
from .io.safetensors import Checkpoint
from .kernel import KernelConfig, Limits, SandboxKernel
from .passes import TorchForward
from .passes.edits import apply_edits, last_block, render_weights
from .prompts import PromptBuilder
from .runners.torch import TorchRunner, state_to_checkpoint, zero_checkpoint
from .session import V5, Messages, Policy, Session
from .store import WeightStore
from .upstream import home, sha256_file

LIBRARIES = "numpy, scipy, scikit-learn, PyTorch, pandas, safetensors, tokenizers"
INLINE_LIMIT = 1_000_000
HARNESS = "reconstructed"


def available_memory_gb() -> float:
    """What a kernel may use by default: most of the machine, leaving room for everything else."""
    return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30 * 0.6, 1)


def prepare_weights(model, folder: Path) -> Path:
    """Any source of weights as a safetensors checkpoint. `model`: a path, a state dict, or an nn.Module."""
    if isinstance(model, (str, os.PathLike)):
        path = Path(model)
        if path.is_dir() or path.suffix == ".safetensors":
            Checkpoint(path)
            return path
        out = folder / "base" / "model.safetensors"
        if path.suffix == ".npz":
            import torch

            with np.load(path, allow_pickle=False) as z:
                return state_to_checkpoint({k: torch.from_numpy(z[k]) for k in z.files}, out)
        if path.suffix in (".pt", ".pth", ".bin", ".ckpt"):
            import torch

            state = torch.load(path, map_location="cpu", weights_only=True)
            return state_to_checkpoint(state.get("state_dict", state), out)
        raise ValueError(f"unsupported weights file: {path}")
    state = model.state_dict() if hasattr(model, "state_dict") else model
    return state_to_checkpoint(state, folder / "base" / "model.safetensors")


class TorchEvaluation:
    """Scores any network with a Scorer."""

    def __init__(self, runner: TorchRunner, scorer, name: str = "the evaluation data"):
        self.runner = runner
        self.scorer = scorer
        self.name = name

    def evaluate(self, weights: Path) -> Scores:
        return self.scorer.score(self.runner.call(weights), self.scorer.ids())

    def describe(self) -> str:
        return f"the network is run on held-out items and scored with {getattr(self.scorer, 'metric_name', 'the scorer')}"


@dataclass
class RunResult:
    verdict: str
    folder: Path
    patch: Path | None
    baseline: float | None
    final: float | None
    comparison: Paired | None
    reasons: list[str]
    answer: str | None
    state: object
    weights: Path | None = None
    runner: object = None
    extra: dict = field(default_factory=dict)

    @property
    def report(self) -> dict:
        return json.loads((self.folder / "report.json").read_text())

    def infer(self, inputs):
        """Run the edited network."""
        return self.runner.call(self.weights)(inputs)


@dataclass
class Job:
    """Everything one run needs. Language models and generic networks differ only in `tools`, `evaluation` and slots."""

    weights: Path
    kind: str
    mode: str
    folder: Path
    editor: Editor
    prompt: PromptBuilder
    practice_dir: Path | None
    tools_factory: object
    evaluation: object
    regressions: dict
    limits: Limits
    frozen: tuple
    names: list | None
    budget: int
    max_runs: int
    access: str
    sandbox: str
    policy: Policy
    messages: Messages
    control: bool
    z: float
    runner: object
    prelude: str = ""
    meta: dict = field(default_factory=dict)


def build_edit(job: Job, base: Path, edited_dir: Path, out: Path, *, meta: dict | None = None) -> tuple[P.Patch, Path]:
    """Turn the submitted tensors into a patch and a model. Forward passes and the evaluation both run this, so what gets
    measured is exactly what the patch contains."""
    checkpoint = Checkpoint(base)
    patch = P.from_submission(checkpoint, P.read_submission(edited_dir), out / "patch", meta=meta)
    target = out / ("model.safetensors" if len(checkpoint.files) == 1 and Path(base).is_file() else "model")
    return patch, P.apply(patch, checkpoint, target)


def edited_builder(job: Job, base: Path, store: WeightStore):
    def build() -> Path:
        out = job.folder / "checkpoints" / (store.digest() or "none")
        model = out / "model.safetensors"
        folder = out / "model"
        if model.exists() or folder.exists():
            return model if model.exists() else folder
        if not store.edited():
            return base
        return build_edit(job, base, store.dir, out)[1]

    return build


def run_session(job: Job, base: Path) -> tuple[str | None, object, Path]:
    work = job.folder / "work"
    work.mkdir(parents=True, exist_ok=True)
    config = KernelConfig(model=base, work=work, practice=job.practice_dir, mode=job.mode, frozen=job.frozen,
                          refuse_rescale=job.mode == "edit", names=job.names, prelude=job.prelude, limits=job.limits,
                          sandbox=job.sandbox)
    with SandboxKernel(config) as kernel:
        tools = job.tools_factory(job, base, kernel.store) if job.budget else {}
        session = Session(job.prompt.build(), job.editor, kernel, tools, budget=job.budget, max_runs=job.max_runs,
                          folder=job.folder / "session", messages=job.messages, policy=job.policy)
        result = session.run()
    return result.final_answer, result.state, work / "edited"


def run_inline(job: Job, base: Path) -> tuple[str | None, object, Path]:
    """One reply, no tools: the weights are in the prompt and the edits come back in an `edits` block."""
    from .prompts import SYSTEM
    from .session import State

    folder = job.folder / "session" / "turns" / "t001"
    folder.mkdir(parents=True, exist_ok=True)
    message = job.prompt.build()
    (folder / "prompt.txt").write_text(message)
    reply = job.editor.reply(SYSTEM, [{"role": "user", "content": message}]).text
    (folder / "answer.md").write_text(reply)
    state = State(turn=1, done=True, stop_reason="finished")
    edited = job.folder / "work" / "edited"
    if job.mode == "read":
        return reply, state, edited
    store = WeightStore(Checkpoint(base), edited, mode=job.mode, frozen=job.frozen, refuse_rescale=False, names=job.names)
    block = last_block(reply)
    applied = apply_edits(block or "", store.load)
    for name, tensor in applied.tensors.items():
        store.save_edit(name, tensor)
    (job.folder / "session" / "apply.json").write_text(json.dumps(applied.report(), indent=1) + "\n")
    return reply, state, edited


def _paired(a: Scores, b: Scores) -> Paired:
    ids = sorted(set(a.per_item) & set(b.per_item))
    return paired([a.per_item[i] for i in ids], [b.per_item[i] for i in ids])


def _scores_json(scores: Scores) -> dict:
    return {"mean": scores.mean, "per_item": scores.per_item, **scores.extra}


def execute(job: Job, original: Path) -> RunResult:
    """`original` is what the result is compared with; a from-scratch session starts from zeros but is compared with
    those zeros too, so the score says how far the written network got."""
    folder = job.folder
    measured = job.evaluation is not None
    baseline = job.evaluation.evaluate(original) if measured else None
    regress_base = {k: e.evaluate(original) for k, e in job.regressions.items()} if measured else {}
    if baseline is not None:
        (folder / "baseline.json").write_text(json.dumps(_scores_json(baseline)) + "\n")

    start = run_inline if job.access == "inline" else run_session
    answer, state, edited_dir = start(job, original)
    meta = {**job.meta, "editor": job.editor.name, "prompt_fidelity": job.prompt.fidelity,
            "prompt_sha256": job.prompt.sha256(), "harness": HARNESS, "tool_free": True, "mode": job.mode}
    calls = folder / "session" / "tool_calls.jsonl"
    if calls.exists():
        meta["transcript_sha256"] = sha256_file(calls)

    if job.mode == "read":
        report = {"verdict": UNMEASURED, "reasons": ["this run only reads the weights"], "answer": answer,
                  "baseline": None, "final": None, "patch": None}
        return finish(job, report, None, None, None, None, state, answer)

    final_out = folder / "result"
    patch, model = build_edit(job, original, edited_dir, final_out, meta=meta)
    final = control = None
    regress: dict[str, Paired] = {}
    comparison = None
    changed = patch.manifest["changed_entries_total"]
    if measured and changed:
        final = job.evaluation.evaluate(model)
        comparison = _paired(baseline, final)
        for k, e in job.regressions.items():
            regress[k] = _paired(regress_base[k], e.evaluate(model))
        if job.control:
            control = measure_control(job, original, model, patch, final)
    elif measured:
        comparison = None
    gate: Gate = decide(comparison, changed_entries=changed, control=control, regressions=regress, z=job.z)
    report = {"verdict": gate.verdict, "reasons": gate.reasons, "answer": answer,
              "baseline": baseline.mean if baseline else None, "final": final.mean if final else None,
              "comparison": comparison.as_dict() if comparison else None,
              "control": control.as_dict() if control else None,
              "regressions": {k: p.as_dict() for k, p in regress.items()},
              "patch": str(final_out / "patch"), "changed_entries": changed}
    evidence = {k: report[k] for k in ("verdict", "reasons", "baseline", "final", "comparison", "control", "regressions")}
    P.write_meta(final_out / "patch", {**meta, "evidence": evidence})
    return finish(job, report, final_out / "patch", model, comparison, final, state, answer)


def measure_control(job: Job, original: Path, model: Path, patch: P.Patch, final: Scores) -> Paired | None:
    """How the real edit compares with a random edit of the same size: (random, real)."""
    with tempfile.TemporaryDirectory(dir=job.folder) as tmp:
        tmp = Path(tmp)
        base = Checkpoint(original)
        noise = random_control(base, patch, Checkpoint(model))
        random_patch = P.from_submission(base, noise, tmp / "patch")
        target = tmp / ("model.safetensors" if Path(original).is_file() else "model")
        scores = job.evaluation.evaluate(P.apply(random_patch, base, target))
    return _paired(scores, final)


def finish(job, report, patch, model, comparison, final, state, answer) -> RunResult:
    (job.folder / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    write_report(job.folder / "report.md", report)
    return RunResult(report["verdict"], job.folder, patch, report.get("baseline"), report.get("final"), comparison,
                     report.get("reasons", []), answer, state, model, job.runner)


def _prompt_slots(job_slots: dict, limits: Limits, budget: int, max_runs: int) -> dict:
    return {"libraries": LIBRARIES, "run_seconds": limits.seconds, "run_memory": f"{limits.memory_gb:g} GB",
            "output_cut": f"{limits.output_chars:,}", "max_runs": max_runs, "passes": budget, **job_slots}


def _make_builder(prompt, base: str, slots: dict, note, passes: bool) -> PromptBuilder:
    if isinstance(prompt, PromptBuilder):
        prompt.slots = {**slots, **prompt.slots}
        prompt.note = prompt.note or note
        return prompt
    if isinstance(prompt, str):
        return PromptBuilder(template=prompt, slots=slots, note=note)
    return PromptBuilder(base, slots=slots, note=note, passes=passes)


def run(model, *, build=None, description: str, note: str | None = None, mode: str = "edit", practice=None, evaluate=None,
        editor=None, budget: int = 25, max_runs: int = 300, access: str = "kernel", folder=None, frozen=(), prompt=None,
        task_sentence: str | None = None, regressions: dict | None = None, control: bool = True, limits: Limits | None = None,
        sandbox: str = "bwrap", policy: Policy | None = None, device: str = "cpu", z: float = 2.0, prelude: str = "",
        messages: Messages = V5, no_eval: bool = False) -> RunResult:
    """Run the whole job on any PyTorch network.

    `build` says how to run the weights: a function returning an nn.Module, "path/model.py:function", or
    {"mlp": [sizes], "activation": "tanh"}. `practice` and `evaluate` are Scorers: the editor sees the first and is judged
    on the second. If only `evaluate` is an ArrayScorer, it is split in two. `note` is your own words to the editor.
    """
    if mode not in ("read", "edit", "write"):
        raise ValueError(f"unknown mode {mode!r}")
    if access not in ("kernel", "inline"):
        raise ValueError(f"unknown access {access!r}")
    if mode != "read" and evaluate is None and not no_eval:
        raise ValueError("give `evaluate` to measure the result, or no_eval=True to accept an unmeasured verdict")
    folder = Path(folder or tempfile.mkdtemp(prefix="run-", dir=_runs_dir()))
    folder.mkdir(parents=True, exist_ok=True)
    limits = limits or Limits()
    if limits.memory_gb is None:
        limits.memory_gb = available_memory_gb()
    weights = prepare_weights(model, folder)
    if mode == "write":
        weights = zero_checkpoint(Checkpoint(weights), folder / "zero" / "model.safetensors")
    runner = TorchRunner(build, device) if build is not None else None
    if practice is None and evaluate is not None and hasattr(evaluate, "split"):
        practice, evaluate = evaluate.split(0.5)
    if practice is not None and evaluate is not None:
        check_disjoint(practice, evaluate)
    practice_dir = None
    if practice is not None:
        practice_dir = folder / "practice"
        practice.export(practice_dir)
    if budget and (runner is None or practice is None):
        raise ValueError("forward passes need `build` and `practice`; pass budget=0 for a weights-only session")

    slots = {"network": description, "benchmark": getattr(evaluate, "name", "held-out data"),
             "protocol": "the network is run on held-out items and scored",
             "practice": practice.describe() if practice is not None else "There is no practice data.",
             "modules": ", ".join(runner.module_names(weights)) if runner else "",
             "out_of_scope": ", ".join(frozen) or None}
    if task_sentence:
        slots["task_sentence"] = task_sentence
    if access == "inline":
        checkpoint = Checkpoint(weights)
        text = render_weights(checkpoint)
        if len(text) > INLINE_LIMIT:
            raise ValueError(f"the weights are {len(text):,} characters: too large for inline access, use kernel access")
        slots.update(weights=text, tensor_names=", ".join(checkpoint.names()),
                     start_note="Every weight starts at 0." if mode == "write" else "")
        template = "inline_read" if mode == "read" else "inline_edit"
    else:
        template = f"generic_{mode}"

    def tools(job: Job, base: Path, store: WeightStore) -> dict:
        return {"forward": TorchForward(job.runner, practice, work=job.folder / "work", base=base,
                                        edited=edited_builder(job, base, store), edit_digest=store.digest)}

    evaluation = TorchEvaluation(runner, evaluate, slots["benchmark"]) if evaluate is not None and not no_eval else None
    job = Job(weights=weights, kind="generic", mode=mode, folder=folder, editor=get_editor(editor),
              prompt=_make_builder(prompt, template, _prompt_slots(slots, limits, budget, max_runs), note, budget > 0),
              practice_dir=practice_dir, tools_factory=tools, evaluation=evaluation,
              regressions={k: TorchEvaluation(runner, v, k) for k, v in (regressions or {}).items()}, limits=limits,
              frozen=tuple(frozen), names=None, budget=budget, max_runs=max_runs, access=access, sandbox=sandbox,
              policy=policy or Policy(), messages=messages, control=control and mode == "edit", z=z, runner=runner,
              prelude=prelude, meta={"kind": "generic"})
    return execute(job, weights)


def _runs_dir() -> Path:
    path = home() / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["RunResult", "run"]
