"""The whole job in one call: load the network, score it, let the editor work, build the result, score it again, judge."""

from __future__ import annotations

import json
import os
import shutil
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
from .prompts import SYSTEM, PromptBuilder
from .runners.torch import TorchRunner, state_to_checkpoint, zero_checkpoint
from .semantic import SemanticMap, kernel_prelude, prompt_addendum
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

    def generate(self, message: str, **sampling) -> str:
        """Ask the edited language model something."""
        from .runners.base import Prompt

        options = {"temperature": 0.7, "top_p": 0.8, "max_tokens": 512, **sampling}
        prompt = Prompt("user", self.runner.render(message), 0)
        return self.runner.generate(self.weights, [prompt], 1, options)[0][0].text


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
    binds: dict = field(default_factory=dict)
    force_tools: bool = False
    resume: bool = False


def build_edit(job: Job, base: Path, edited_dir: Path, out: Path, *, meta: dict | None = None) -> tuple[P.Patch, Path]:
    """Turn the submitted tensors into a patch and a model. Forward passes and the evaluation both run this, so what gets
    measured is exactly what the patch contains."""
    checkpoint = Checkpoint(base)
    patch = P.from_submission(checkpoint, P.read_submission(edited_dir), out / "patch", meta=meta)
    target = out / ("model.safetensors" if len(checkpoint.files) == 1 and Path(base).is_file() else "model")
    return patch, P.apply(patch, checkpoint, target)


def edited_builder(job: Job, base: Path, store: WeightStore, keep: int = 2):
    """The edited model for a forward pass, built once per distinct edit. Only the latest few stay on disk: a language
    model's edited shard is a full copy."""

    def build() -> Path:
        if not store.edited():
            return base
        root = job.folder / "checkpoints"
        out = root / store.digest()
        model = out / "model.safetensors"
        folder = out / "model"
        if model.exists() or folder.exists():
            return model if model.exists() else folder
        older = sorted(root.glob("*"), key=lambda p: p.stat().st_mtime) if root.is_dir() else []
        for stale in older[:max(0, len(older) - keep + 1)]:
            shutil.rmtree(stale)
        return build_edit(job, base, store.dir, out)[1]

    return build


def run_session(job: Job, base: Path) -> tuple[str | None, object, Path]:
    work = job.folder / "work"
    work.mkdir(parents=True, exist_ok=True)
    config = KernelConfig(model=base, work=work, practice=job.practice_dir, mode=job.mode, frozen=job.frozen,
                          refuse_rescale=job.mode == "edit", names=job.names, prelude=job.prelude, limits=job.limits,
                          sandbox=job.sandbox, binds=job.binds)
    if job.mode != "read" and not job.policy.warn_unsaved:
        job.policy.warn_unsaved = 3
    with SandboxKernel(config) as kernel:
        tools = job.tools_factory(job, base, kernel.store) if job.budget or job.force_tools else {}
        session = Session(job.prompt.build(), job.editor, kernel, tools, budget=job.budget, max_runs=job.max_runs,
                          folder=job.folder / "session", messages=job.messages, policy=job.policy, resume=job.resume)
        result = session.run()
    return result.final_answer, result.state, work / "edited"


def run_inline(job: Job, base: Path) -> tuple[str | None, object, Path]:
    """One reply, no tools: the weights are in the prompt and the edits come back in an `edits` block."""
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
    saved = folder / "baseline.json"
    if job.resume and measured and saved.exists():
        data = json.loads(saved.read_text())
        baseline = Scores(data["baseline"]["per_item"])
        regress_base = {k: Scores(v["per_item"]) for k, v in data["regressions"].items()}
    else:
        baseline = job.evaluation.evaluate(original) if measured else None
        regress_base = {k: e.evaluate(original) for k, e in job.regressions.items()} if measured else {}
        if baseline is not None:
            saved.write_text(json.dumps({"baseline": _scores_json(baseline),
                                         "regressions": {k: _scores_json(v) for k, v in regress_base.items()}}) + "\n")

    start = run_inline if job.access == "inline" else run_session
    answer, state, edited_dir = start(job, original)
    meta = {**job.meta, "editor": job.editor.name, "prompt_fidelity": job.prompt.fidelity,
            "prompt_sha256": job.prompt.sha256(), "harness": HARNESS, "mode": job.mode,
            "tool_free": getattr(job.editor, "tool_free", True)}
    if hasattr(job.editor, "evidence"):
        meta["council"] = job.editor.evidence()
    calls = folder / "session" / "tool_calls.jsonl"
    if calls.exists():
        meta["transcript_sha256"] = sha256_file(calls)

    if job.mode == "read":
        report = {"verdict": UNMEASURED, "reasons": ["this run only reads the weights"], "answer": answer,
                  "baseline": None, "final": None, "patch": None}
        return finish(job, report, None, None, None, None, state, answer)

    final_out = folder / "result"
    patch, model = build_edit(job, original, edited_dir, final_out, meta=meta)
    return judge(job, original, baseline, regress_base, patch, model, final_out, meta, answer, state)


def judge(job: Job, original: Path, baseline, regress_base: dict, patch: P.Patch, model: Path, final_out: Path, meta: dict,
          answer, state) -> RunResult:
    """Score an edit on the evaluation, and reach the verdict."""
    measured = job.evaluation is not None
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


def run(model, *, build=None, description: str | None = None, note: str | None = None, mode: str = "edit", practice=None,
        evaluate=None, eval=None,
        editor=None, budget: int = 25, max_runs: int = 300, access: str = "kernel", folder=None, frozen=(), prompt=None,
        task_sentence: str | None = None, regressions: dict | None = None, control: bool = True, limits: Limits | None = None,
        sandbox: str = "bwrap", policy: Policy | None = None, device: str = "cpu", z: float = 2.0, prelude: str = "",
        messages: Messages = V5, no_eval: bool = False, semantic: bool = False, tools=None, binds: dict | None = None,
        weights_names: list | None = None, resume: bool = False, **lm_options) -> RunResult:
    """Run the whole job on any PyTorch network.

    `build` says how to run the weights: a function returning an nn.Module, "path/model.py:function", or
    {"mlp": [sizes], "activation": "tanh"}. `practice` and `evaluate` are Scorers: the editor sees the first and is judged
    on the second. If only `evaluate` is an ArrayScorer, it is split in two. `note` is your own words to the editor.
    `semantic` gives the editor M, the semantic layer, in its kernel, and says so in the prompt.

    For a language model, give `eval` instead: "math", or an LMSuite. Options such as `thinking`, `revision` or `generator`
    then pass through to run_lm, and `description` is generated from the model's config.

    `tools` replaces the forward tool: a function (work folder, base weights, store, edited model builder) -> {fence: PassTool},
    for networks that need passes of their own, such as playing games. `binds` adds read-only {host path: path in the sandbox}.
    """
    if eval is not None:
        return run_lm(model, eval=eval, editor=editor, note=note, mode=mode, budget=budget, max_runs=max_runs,
                      description=description, folder=folder, frozen=frozen or None, prompt=prompt, task_sentence=task_sentence,
                      regressions=regressions, control=control, limits=limits, sandbox=sandbox, policy=policy, z=z, prelude=prelude,
                      messages=messages, semantic=semantic, resume=resume, **lm_options)
    if lm_options:
        raise TypeError(f"unknown options for this network: {sorted(lm_options)}; language model options need `eval`")
    if description is None:
        raise ValueError("give `description`: what the network is, for the editor")
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
    if resume and (folder / "base" / "model.safetensors").exists() and not isinstance(model, (str, os.PathLike)):
        weights = folder / "base" / "model.safetensors"
    else:
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
    if budget and tools is None and (runner is None or practice is None):
        raise ValueError("forward passes need `build` and `practice`; pass budget=0 for a weights-only session")

    slots = {"network": description.strip().rstrip("."), "benchmark": getattr(evaluate, "name", "held-out data"),
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

    def default_tools(job: Job, base: Path, store: WeightStore) -> dict:
        if tools is not None:
            return tools(job.folder / "work", base, store, edited_builder(job, base, store))
        return {"forward": TorchForward(job.runner, practice, work=job.folder / "work", base=base,
                                        edited=edited_builder(job, base, store), edit_digest=store.digest)}

    evaluation = TorchEvaluation(runner, evaluate, slots["benchmark"]) if evaluate is not None and not no_eval else None
    builder = _make_builder(prompt, template, _prompt_slots(slots, limits, budget, max_runs), note, budget > 0)
    if semantic:
        builder.addendum = prompt_addendum(SemanticMap(Checkpoint(weights).names(), {}, load=None, save=None,
                                                       shape=Checkpoint(weights).shape).family)
        prelude = kernel_prelude() + prelude
    job = Job(weights=weights, kind="generic", mode=mode, folder=folder, editor=get_editor(editor), prompt=builder,
              practice_dir=practice_dir, tools_factory=default_tools, evaluation=evaluation,
              regressions={k: TorchEvaluation(runner, v, k) for k, v in (regressions or {}).items()}, limits=limits,
              frozen=tuple(frozen), names=weights_names, binds=binds or {}, force_tools=tools is not None, resume=resume,
              budget=budget, max_runs=max_runs, access=access, sandbox=sandbox,
              policy=policy or Policy(), messages=messages, control=control and mode == "edit", z=z, runner=runner,
              prelude=prelude, meta={"kind": "generic"})
    return execute(job, weights)


@dataclass
class SearchResult:
    """`success_rate` is the share of runs whose edit beat the base model on the selection set, with z >= `z`."""

    result: RunResult
    success_rate: float
    candidates: list[dict]
    selection: dict


def search(model, *, editors: list, runs: int | None = None, build, description: str, practice=None, evaluate,
           selection: str = "metric", spending=None, method: str = "borda", folder=None, z: float = 2.0, seed: int = 0,
           **options) -> SearchResult:
    """Several independent sessions, then one winner. The practice data is split: editors work on one part and candidates
    are compared on the other, so only the winner ever meets the evaluation.

    `selection` is "metric" (the best score on the selection set) or "vote" (the editors vote on the finished candidates,
    the `weights` scope of a council). `spending` is required: this runs a whole session per candidate.
    """
    from .council import vote
    from .editors.cost import Metered
    from .session.state import State

    if spending is None:
        raise ValueError("search runs many sessions: give spending=Spending(max_tokens=... or max_usd=...)")
    if selection not in ("metric", "vote"):
        raise ValueError('selection must be "metric" or "vote"')
    folder = Path(folder or tempfile.mkdtemp(prefix="search-", dir=_runs_dir()))
    folder.mkdir(parents=True, exist_ok=True)
    if practice is None:
        practice, evaluate = evaluate.split(0.5)
    work_set, select_set = practice.split(0.5, seed)
    check_disjoint(select_set, evaluate)
    check_disjoint(work_set, evaluate)
    runner = TorchRunner(build, options.get("device", "cpu"))
    base = prepare_weights(model, folder)
    selector = TorchEvaluation(runner, select_set)
    baseline = selector.evaluate(base)
    count = runs or len(editors)
    candidates = []
    for k in range(count):
        editor = get_editor(editors[k % len(editors)])
        metered = Metered(editor, spending)
        one = run(base, build=build, description=description, practice=work_set, evaluate=None, no_eval=True,
                  editor=metered, folder=folder / f"run_{k}", **options)
        row = {"run": k, "editor": editor.name, "folder": str(one.folder), "patch": str(one.patch), "answer": one.answer,
               "stop_reason": getattr(one.state, "stop_reason", None)}
        changed = P.Patch.load(one.patch).manifest["changed_entries_total"]
        row["changed_entries"] = changed
        if changed:
            scores = selector.evaluate(one.weights)
            comparison = _paired(baseline, scores)
            row.update(score=scores.mean, gain=comparison.difference, standard_error=comparison.standard_error,
                       z=comparison.z())
        else:
            row.update(score=baseline.mean, gain=0.0, standard_error=0.0, z=0.0)
        candidates.append(row)
        if spending_exhausted(spending):
            break
    success = sum(c["z"] >= z for c in candidates) / len(candidates)
    usable = [c for c in candidates if c["changed_entries"]]
    decision: dict = {"method": selection}
    if not usable:
        winner = candidates[0]
    elif selection == "metric":
        winner = max(usable, key=lambda c: c["score"])
    else:
        text = [f"Changed entries: {c['changed_entries']:,}\nGain on the selection set: {c['gain']:+.4f} +/- "
                f"{c['standard_error']:.4f}\n\nThe editor's account:\n{c['answer']}" for c in usable]
        members = [get_editor(e) for e in editors]
        decision.update(vote(text, members, system=SYSTEM, spending=spending, method=method, seed=seed,
                             measured=[c["score"] for c in usable],
                             messages=[{"role": "user", "content": f"The network: {description}"}]))
        winner = usable[decision["winner"]]
    decision["run"] = winner["run"]

    final_out = folder / "winner"
    patch = P.Patch.load(winner["patch"])
    model_path = final_out / ("model.safetensors" if Path(base).is_file() else "model")
    final_out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(winner["patch"], final_out / "patch", dirs_exist_ok=True)
    P.apply(patch, Checkpoint(base), model_path)
    meta = {**patch.meta, "search": {"runs": len(candidates), "success_rate": success, "selection": selection}}
    job = Job(weights=base, kind="generic", mode="edit", folder=folder, editor=get_editor(editors[0]),
              prompt=PromptBuilder(template="unused"), practice_dir=None, tools_factory=None,
              evaluation=TorchEvaluation(runner, evaluate), regressions={}, limits=Limits(), frozen=(), names=None, budget=0,
              max_runs=0, access="kernel", sandbox="none", policy=Policy(), messages=V5,
              control=options.get("control", True), z=z, runner=runner)
    result = judge(job, base, job.evaluation.evaluate(base), {}, P.Patch.load(final_out / "patch"), model_path, final_out, meta,
                   winner["answer"], State())
    summary = {"success_rate": success, "selection": decision, "candidates": candidates}
    (folder / "search.json").write_text(json.dumps(summary, indent=1) + "\n")
    return SearchResult(result, success, candidates, decision)


def spending_exhausted(spending) -> bool:
    from .editors.cost import BudgetExceeded

    try:
        spending.check()
    except BudgetExceeded:
        return True
    return False


def resolve_model(model: str, revision: str | None = None) -> Path:
    """A local folder as it is, otherwise a Hugging Face repo id."""
    path = Path(model)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, revision=revision, allow_patterns=[
        "*.safetensors", "*.safetensors.index.json", "*.json", "*.txt", "*.jinja", "*.model"]))


def run_lm(model: str, *, eval, editor, note: str | None = None, mode: str = "edit", budget: int = 25, max_runs: int = 300,
           description: str | None = None, name: str | None = None, thinking: bool = False, revision: str | None = None,
           generator: str = "vllm", device: str = "cuda", folder=None, frozen=None, prompt=None, task_sentence: str | None = None,
           regressions: dict | None = None, control: bool = True, limits: Limits | None = None, sandbox: str = "bwrap",
           policy: Policy | None = None, z: float = 2.0, prelude: str = "", messages: Messages = V5, semantic: bool = False,
           runner=None, forward_limits=None, pass_time: str = "several minutes", full_attention=None,
           resume: bool = False, cache_answers: bool = True) -> RunResult:
    """Run the whole job on a language model. `eval` is a suite: "math", or an LMSuite of your own data."""
    from .arch_text import default_frozen, describe, text_tensors
    from .eval.lm_suite import LMEvaluation, LMSuite, math_suite
    from .passes import ForwardLimits, GenerationCache, LMForward
    from .runners import LM

    suite: LMSuite = math_suite() if eval == "math" else eval
    folder = Path(folder or tempfile.mkdtemp(prefix="run-", dir=_runs_dir()))
    folder.mkdir(parents=True, exist_ok=True)
    limits = limits or Limits()
    if limits.memory_gb is None:
        limits.memory_gb = available_memory_gb()
    weights = resolve_model(model, revision)
    checkpoint = Checkpoint(weights)
    config = checkpoint.config
    text = config.get("text_config") or config
    names = text_tensors(checkpoint.names())
    frozen = default_frozen(names) if frozen is None else tuple(frozen)
    runner = runner or LM(weights, thinking=thinking, generator=generator, device=device)
    layers = text["num_hidden_layers"]
    types = text.get("layer_types") or ["full_attention"] * layers
    full = tuple(full_attention if full_attention is not None else
                 [i for i, t in enumerate(types) if t in ("full_attention", "attention")])
    lim = forward_limits or ForwardLimits(n_layers=layers, full_attention=full, max_tokens=suite.sampling["max_tokens"],
                                          tf_max_tokens=suite.sampling["max_tokens"])
    practice_dir = folder / "practice"
    suite.export(practice_dir)
    rows = {r["id"]: r for r in suite.practice}

    slots = {"network": description or describe(name or str(model).rsplit("/", 1)[-1], config), "benchmark": suite.benchmark,
             "protocol": suite.describe_protocol(thinking), "practice": suite.describe_practice(),
             "out_of_scope": "the final norm and the embedding / output matrix" if frozen else None,
             "max_tokens": f"{lim.max_tokens:,}", "n_layers": layers, "gen_max": f"{lim.gen_max:,}",
             "both_max": f"{lim.gen_max // 2:,}", "act_tokens": f"{lim.act_max_tokens:,}", "tf_texts": f"{lim.tf_max_texts:,}",
             "tf_tokens": f"{lim.tf_max_tokens:,}", "tf_total": f"{lim.tf_total_tokens:,}", "pass_time": pass_time}
    if task_sentence:
        slots["task_sentence"] = task_sentence

    def tools(job: Job, base: Path, store: WeightStore) -> dict:
        work = job.folder / "work"
        (work / "eval_prompt_format.txt").write_text(runner.render("<the problem>"))
        return {"forward": LMForward(runner, practice=rows, grader=suite.grader, limits=lim, sampling=suite.sampling, work=work,
                                     base=base, edited=edited_builder(job, base, store), edit_digest=store.digest,
                                     user_message=suite.user_message,
                                     cache=GenerationCache(job.folder / "cache" / "generations.jsonl") if cache_answers else None)}

    builder = _make_builder(prompt, f"lm_{mode}", _prompt_slots(slots, limits, budget, max_runs), note, budget > 0)
    if semantic:
        builder.addendum = prompt_addendum("llama")
        prelude = kernel_prelude() + prelude
    job = Job(weights=weights, kind="lm", mode=mode, folder=folder, editor=get_editor(editor), prompt=builder,
              practice_dir=practice_dir, tools_factory=tools, evaluation=LMEvaluation(runner, suite),
              regressions={k: LMEvaluation(runner, v) for k, v in (regressions or {}).items()}, limits=limits, frozen=frozen,
              names=names, budget=budget, max_runs=max_runs, access="kernel", sandbox=sandbox, policy=policy or Policy(),
              messages=messages, control=control and mode == "edit", z=z, runner=runner, prelude=prelude, resume=resume,
              meta={"kind": "lm", "base": {"repo": str(model), "revision": revision}})
    return execute(job, weights)


def _runs_dir() -> Path:
    path = home() / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["RunResult", "SearchResult", "run", "run_lm", "search"]
