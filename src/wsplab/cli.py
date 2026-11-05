"""The wsplab command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from . import patch as P
from .io import Checkpoint


def _print(obj) -> None:
    print(json.dumps(obj, indent=1))


def cmd_patch_create(a) -> int:
    base = Checkpoint(a.base)
    meta = {"name": a.name, "description": a.description, "base": {"repo": a.base_repo, "revision": a.base_revision}}
    meta = {k: v for k, v in meta.items() if v}
    if a.edited:
        patch = P.from_checkpoints(base, Checkpoint(a.edited), a.out, meta=meta)
    else:
        patch = P.from_submission(base, P.read_submission(a.submission), a.out, meta=meta)
    print(f"{len(patch.tensors)} tensors, {patch.manifest['changed_entries_total']:,} entries changed -> {a.out}")
    for name in patch.manifest.get("no_effect", []):
        print(f"warning: {name}: the edit rounds back to the base model and has no effect", file=sys.stderr)
    return 0


def cmd_patch_apply(a) -> int:
    out = P.apply(P.Patch.load(a.patch), Checkpoint(a.base), a.out, link=not a.copy)
    print(f"edited model -> {out}")
    return 0


def cmd_patch_verify(a) -> int:
    patch = P.Patch.load(a.patch)
    if a.base is None:
        print(f"{a.patch}: well formed, digests match. Pass --base to check it against a model.")
        return 0
    for row in P.verify(patch, Checkpoint(a.base)):
        print(f"{row['name']}: {row['form']}, {row['changed_entries']:,} entries changed, hash OK")
    return 0


def cmd_patch_inspect(a) -> int:
    patch = P.Patch.load(a.patch)
    rows = [{k: t.get(k) for k in ("name", "form", "dtype", "shape", "changed_entries", "relative_change")}
            for t in patch.tensors]
    _print({"meta": patch.meta, "format_version": patch.version, "tensors": rows})
    return 0


def cmd_patch_stack(a) -> int:
    out = P.stack([P.Patch.load(p) for p in a.patches], Checkpoint(a.base), a.out)
    print(f"{len(out.tensors)} tensors -> {a.out}")
    return 0


def cmd_patch_import(a) -> int:
    meta = P.import_release(a.source, a.out, name=a.name, base_repo=a.base_repo, base_revision=a.base_revision,
                            license=a.license, attribution=a.attribution)
    print(f"imported {a.source} -> {a.out}: {meta.get('name')}")
    return 0


def parse_build(spec: str | None):
    """"mlp:2,4,2:relu" for a built-in MLP, otherwise path/to/model.py:function."""
    if spec and spec.startswith("mlp:"):
        _, sizes, *activation = spec.split(":")
        return {"mlp": [int(x) for x in sizes.split(",")], "activation": activation[0] if activation else "relu"}
    return spec


def load_scorer(path: str | None, metric: str, prefix: str = "item"):
    if path is None:
        return None
    import numpy as np

    from .eval.suites import ArrayScorer

    with np.load(path, allow_pickle=False) as z:
        return ArrayScorer(z["inputs"], z["targets"], metric, name=Path(path).stem, prefix=prefix)


def patched_weights(model: str, patch: str | None, folder: Path) -> Path:
    """The model to run: the base, or the base with a patch applied."""
    if patch is None:
        return Path(model)
    base = Checkpoint(model)
    target = folder / ("model.safetensors" if Path(model).is_file() else "model")
    return P.apply(P.Patch.load(patch), base, target)


def cmd_setup(a) -> int:
    from . import doctor

    print(f"sandbox environment: {doctor.setup(force=a.force)}")
    return 0


def cmd_doctor(a) -> int:
    from . import doctor

    results = doctor.checks()
    for c in results:
        print(f"{'ok ' if c.ok else 'NO '} {c.name}: {c.detail} (for {c.needed_for})")
    return 0 if all(c.ok for c in results if c.needed_for in ("everything", "editing sessions")) else 1


def load_suite(a):
    """A language model suite from the command line, or None for a generic network."""
    if a.eval == "math":
        from .eval.lm_suite import math_suite

        return math_suite(samples=a.samples, limit=a.limit)
    if a.eval and a.eval.endswith(".jsonl"):
        from .eval.lm_suite import custom_suite

        return custom_suite(a.practice, a.eval, grader=a.grader, samples=a.samples)
    return None


def _print_result(result) -> int:
    print(f"verdict: {result.verdict}")
    for reason in result.reasons:
        print(f"  {reason}")
    if result.patch:
        print(f"patch: {result.patch}")
    print(f"report: {result.folder / 'report.md'}")
    return 0


def cmd_run(a) -> int:
    import wsplab

    from .kernel import Limits

    limits = Limits(seconds=a.run_seconds, memory_gb=a.memory_gb)
    suite = load_suite(a)
    if suite is not None:
        if len(a.editor) > 1:
            raise ValueError("a council from the command line works on generic networks; use the Python API for language models")
        return _print_result(wsplab.run(a.model, eval=suite, editor=a.editor[0], note=a.note, description=a.description,
                                        mode=a.mode, budget=a.budget, max_runs=a.max_runs, folder=a.out, sandbox=a.sandbox,
                                        control=not a.no_control, limits=limits, thinking=a.thinking, revision=a.revision))
    if a.description is None:
        raise ValueError("give --description: what the network is, for the editor")
    evaluate = load_scorer(a.eval, a.metric, "eval")
    practice = load_scorer(a.practice, a.metric)
    if len(a.editor) > 1:
        if a.max_tokens is None and a.max_usd is None:
            raise ValueError("a council multiplies LLM calls: set --max-tokens or --max-usd")
        from .council import Council
        from .editors import Spending, get_editor

        editor = Council([get_editor(e) for e in a.editor], Spending(max_tokens=a.max_tokens, max_usd=a.max_usd),
                         method=a.vote)
    else:
        editor = a.editor[0]
    result = wsplab.run(a.model, build=parse_build(a.build), description=a.description, note=a.note, mode=a.mode,
                        practice=practice, evaluate=evaluate, editor=editor, budget=a.budget, max_runs=a.max_runs,
                        access=a.access, folder=a.out, sandbox=a.sandbox, control=not a.no_control, no_eval=a.no_eval,
                        limits=limits, frozen=tuple(a.frozen or ()))
    return _print_result(result)


def cmd_infer(a) -> int:
    import tempfile

    import numpy as np

    a.device = a.device or ("cuda" if a.prompt is not None else "cpu")

    from .runners.torch import TorchRunner

    with tempfile.TemporaryDirectory() as tmp:
        weights = patched_weights(a.model, a.patch, Path(tmp))
        if a.prompt is not None:
            from .runners import LM
            from .runners.base import Prompt

            lm = LM(a.model, thinking=a.thinking, generator=a.generator, device=a.device)
            sampling = {"temperature": 0.7, "top_p": 0.8, "max_tokens": a.max_tokens}
            print(lm.generate(weights, [Prompt("user", lm.render(a.prompt), 0)], 1, sampling)[0][0].text)
            return 0
        if a.build is None or a.input is None or a.out is None:
            raise ValueError("give --prompt for a language model, or --build, --input and --out for any other network")
        out = TorchRunner(parse_build(a.build), a.device).call(weights)(np.load(a.input, allow_pickle=False))
    np.save(a.out, out)
    print(f"outputs {list(out.shape)} -> {a.out}")
    return 0


def cmd_eval(a) -> int:
    import tempfile

    a.device = a.device or ("cuda" if a.suite else "cpu")

    from .runners.torch import TorchRunner

    with tempfile.TemporaryDirectory() as tmp:
        weights = patched_weights(a.model, a.patch, Path(tmp))
        if a.suite is not None:
            from .eval.lm_suite import LMEvaluation, custom_suite, math_suite
            from .runners import LM

            suite = (math_suite(samples=a.samples, limit=a.limit) if a.suite == "math"
                     else custom_suite(a.suite, a.suite, grader=a.grader, samples=a.samples, check_overlap=False))
            scores = LMEvaluation(LM(a.model, thinking=a.thinking, device=a.device), suite).evaluate(weights)
            print(f"{suite.benchmark}: {scores.mean:.6f} over {len(scores.per_item)} items")
            return 0
        if a.build is None or a.data is None:
            raise ValueError("give --suite for a language model, or --build and --data for any other network")
        scorer = load_scorer(a.data, a.metric)
        scores = scorer.score(TorchRunner(parse_build(a.build), a.device).call(weights), scorer.ids())
    print(f"{a.metric}: {scores.mean:.6f} over {len(scores.per_item)} items")
    return 0


def _say(answer: dict) -> int:
    if answer.get("done"):
        report = answer["report"]
        print(f"verdict: {report['verdict']}")
        for reason in report.get("reasons", []):
            print(f"  {reason}")
        for key in ("baseline", "final", "patch", "report"):
            if report.get(key) is not None:
                print(f"{key}: {report[key]}")
        return 0 if report["verdict"] != "error" else 1
    print(answer["message"])
    return 0


def _wait_for_socket(folder: Path, process, seconds: float = 180) -> None:
    import time

    from .editors.agent import SOCKET

    deadline = time.time() + seconds
    while not (folder / SOCKET).exists():
        if process.poll() is not None:
            raise ValueError(f"the session failed to start; see {folder / 'daemon.log'}")
        if time.time() > deadline:
            raise ValueError("the session did not start in time")
        time.sleep(0.2)


def cmd_session_start(a) -> int:
    import subprocess

    from .editors.agent import talk

    folder = Path(a.folder)
    folder.mkdir(parents=True, exist_ok=True)
    config = {k: getattr(a, k) for k in ("model", "build", "description", "note", "mode", "practice", "eval", "metric", "budget",
                                         "run_seconds", "memory_gb", "no_eval", "sandbox", "semantic")}
    (folder / "agent.json").write_text(json.dumps(config, indent=1) + "\n")
    log = open(folder / "daemon.log", "w")
    process = subprocess.Popen([sys.executable, "-m", "wsplab.session.daemon", str(folder)], stdout=log, stderr=log,
                               start_new_session=True)
    _wait_for_socket(folder, process)
    return _say(talk(folder, {"get": True}))


def _block(kind: str, why: str, body: str) -> str:
    return f"{why}\n```{kind}\n{body}\n```"


def cmd_session_brief(a) -> int:
    from .editors.agent import talk

    return _say(talk(a.folder, {"get": True}))


def cmd_session_exec(a) -> int:
    from .editors.agent import talk

    code = Path(a.file).read_text() if a.file else a.code
    return _say(talk(a.folder, {"reply": _block("python", a.why, code)}))


def cmd_session_pass(a) -> int:
    from .editors.agent import talk

    return _say(talk(a.folder, {"reply": _block(a.fence, a.why, a.request)}))


def cmd_session_finish(a) -> int:
    from .editors.agent import talk

    return _say(talk(a.folder, {"reply": a.answer}))


def cmd_patch_push(a) -> int:
    from .patch.hub import push

    print(push(a.patch, a.repo, private=a.private))
    return 0


def cmd_patch_pull(a) -> int:
    from .patch.hub import pull

    patch = pull(a.repo, a.out, revision=a.revision)
    print(f"{a.repo} -> {patch.path}: {len(patch.tensors)} tensors, digests match")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wsplab", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    top = parser.add_subparsers(dest="command", required=True)

    patch = top.add_parser("patch", help="create, apply and check weight patches").add_subparsers(dest="sub", required=True)
    c = patch.add_parser("create", help="record an edit as a patch")
    c.add_argument("--base", required=True, help="the base model: a .safetensors file or a folder")
    src = c.add_mutually_exclusive_group(required=True)
    src.add_argument("--edited", help="the edited model")
    src.add_argument("--submission", help="a folder of <tensor name>.npy files")
    c.add_argument("--out", required=True)
    c.add_argument("--name")
    c.add_argument("--description")
    c.add_argument("--base-repo")
    c.add_argument("--base-revision")
    c.set_defaults(fn=cmd_patch_create)

    c = patch.add_parser("apply", help="write the edited model")
    c.add_argument("patch")
    c.add_argument("--base", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--copy", action="store_true", help="copy untouched files instead of hard-linking them")
    c.set_defaults(fn=cmd_patch_apply)

    c = patch.add_parser("verify", help="check a patch, and rebuild it against a model with --base")
    c.add_argument("patch")
    c.add_argument("--base")
    c.set_defaults(fn=cmd_patch_verify)

    c = patch.add_parser("inspect", help="print what a patch contains")
    c.add_argument("patch")
    c.set_defaults(fn=cmd_patch_inspect)

    c = patch.add_parser("stack", help="combine patches into one")
    c.add_argument("patches", nargs="+")
    c.add_argument("--base", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_patch_stack)

    c = patch.add_parser("import", help="turn a released edit/ folder (edit.npz + manifest.json) into a patch")
    c.add_argument("source", type=Path)
    c.add_argument("--out", required=True)
    c.add_argument("--name")
    c.add_argument("--base-repo")
    c.add_argument("--base-revision")
    c.add_argument("--license")
    c.add_argument("--attribution")
    c.set_defaults(fn=cmd_patch_import)

    c = patch.add_parser("push", help="publish a patch to the Hugging Face Hub")
    c.add_argument("patch")
    c.add_argument("--repo", required=True, help="user/name")
    c.add_argument("--private", action="store_true")
    c.set_defaults(fn=cmd_patch_push)

    c = patch.add_parser("pull", help="download a patch from the Hugging Face Hub and check it")
    c.add_argument("repo")
    c.add_argument("--out", required=True)
    c.add_argument("--revision")
    c.set_defaults(fn=cmd_patch_pull)

    session = top.add_parser("session", help="be the editor yourself: for coding agents").add_subparsers(dest="sub", required=True)
    c = session.add_parser("start", help="start a session and print the task")
    c.add_argument("--folder", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--build")
    c.add_argument("--description", required=True)
    c.add_argument("--note")
    c.add_argument("--mode", choices=["read", "edit", "write"], default="edit")
    c.add_argument("--practice")
    c.add_argument("--eval")
    c.add_argument("--metric", default="accuracy", choices=["accuracy", "neg_mse"])
    c.add_argument("--budget", type=int, default=25)
    c.add_argument("--run-seconds", type=int, default=900)
    c.add_argument("--memory-gb", type=float)
    c.add_argument("--no-eval", action="store_true")
    c.add_argument("--sandbox", choices=["bwrap", "none"], default="bwrap")
    c.add_argument("--semantic", action="store_true", help="define M, the semantic layer, in the kernel")
    c.set_defaults(fn=cmd_session_start)
    c = session.add_parser("brief", help="print the latest message")
    c.add_argument("folder")
    c.set_defaults(fn=cmd_session_brief)
    c = session.add_parser("exec", help="run Python in the session's kernel")
    c.add_argument("folder")
    c.add_argument("--why", required=True, help="what the last result showed and what you do now")
    source = c.add_mutually_exclusive_group(required=True)
    source.add_argument("--file")
    source.add_argument("--code")
    c.set_defaults(fn=cmd_session_exec)
    c = session.add_parser("forward", help="spend a forward pass")
    c.add_argument("folder")
    c.add_argument("request", help="the JSON request")
    c.add_argument("--why", required=True)
    c.add_argument("--fence", default="forward")
    c.set_defaults(fn=cmd_session_pass)
    c = session.add_parser("finish", help="give the final answer, and get the verdict")
    c.add_argument("folder")
    c.add_argument("answer")
    c.set_defaults(fn=cmd_session_finish)

    c = top.add_parser("setup", help="build the sandbox environment editing sessions run in")
    c.add_argument("--force", action="store_true", help="rebuild it")
    c.set_defaults(fn=cmd_setup)

    c = top.add_parser("doctor", help="check what this machine can run")
    c.set_defaults(fn=cmd_doctor)

    c = top.add_parser("run", help="the whole job: baseline, editing session, final evaluation, patch")
    c.add_argument("--model", required=True, help="weights: .safetensors, a folder, .pt or .npz")
    c.add_argument("--build", help="how to run them: path/model.py:function or mlp:2,4,2:relu")
    c.add_argument("--description", help="what the network is, for the editor; generated for language models")
    c.add_argument("--note", help="your own words to the editor, such as: this model really sucks at math")
    c.add_argument("--mode", choices=["read", "edit", "write"], default="edit")
    c.add_argument("--eval", help=".npz with inputs and targets, or for a language model `math` or a .jsonl: the editor "
                                  "never sees it")
    c.add_argument("--practice", help=".npz or .jsonl: the editor works on it")
    c.add_argument("--grader", default="exact", help="for a language model: math, exact, choice or regex:<pattern>")
    c.add_argument("--samples", type=int, default=1, help="for a language model: answers per evaluation item")
    c.add_argument("--limit", type=int, help="evaluate only this many items of the math suite")
    c.add_argument("--thinking", action="store_true", help="for a language model: use its thinking mode")
    c.add_argument("--revision", help="for a language model: the Hugging Face revision")
    c.add_argument("--metric", default="accuracy", choices=["accuracy", "neg_mse"])
    c.add_argument("--editor", required=True, action="append",
                   help="anthropic:<model>, openai:<model>, codex:<model> or script:<file.json>; repeat it for a council")
    c.add_argument("--vote", choices=["borda", "plurality", "approval"], default="borda", help="how a council votes")
    c.add_argument("--max-tokens", type=int, help="stop a council at this many tokens")
    c.add_argument("--max-usd", type=float, help="stop a council at this cost")
    c.add_argument("--budget", type=int, default=25, help="forward passes the editor may spend")
    c.add_argument("--max-runs", type=int, default=300)
    c.add_argument("--access", choices=["kernel", "inline"], default="kernel")
    c.add_argument("--out", help="the run's folder")
    c.add_argument("--sandbox", choices=["bwrap", "none"], default="bwrap")
    c.add_argument("--run-seconds", type=int, default=900)
    c.add_argument("--memory-gb", type=float)
    c.add_argument("--frozen", action="append", help="a tensor the editor may not change (repeatable)")
    c.add_argument("--no-control", action="store_true", help="skip the random-edit control")
    c.add_argument("--no-eval", action="store_true", help="accept an unmeasured verdict")
    c.set_defaults(fn=cmd_run)

    c = top.add_parser("infer", help="run a network, patched or not, on inputs")
    c.add_argument("--model", required=True)
    c.add_argument("--build")
    c.add_argument("--patch")
    c.add_argument("--input", help=".npy with a batch of inputs")
    c.add_argument("--out")
    c.add_argument("--prompt", help="ask a language model something")
    c.add_argument("--thinking", action="store_true")
    c.add_argument("--generator", choices=["vllm", "hf"], default="hf")
    c.add_argument("--max-tokens", type=int, default=512)
    c.add_argument("--device", default=None)
    c.set_defaults(fn=cmd_infer)

    c = top.add_parser("eval", help="score a network, patched or not")
    c.add_argument("--model", required=True)
    c.add_argument("--build")
    c.add_argument("--patch")
    c.add_argument("--data", help=".npz with inputs and targets")
    c.add_argument("--metric", default="accuracy", choices=["accuracy", "neg_mse"])
    c.add_argument("--suite", help="for a language model: math, or a .jsonl of id, problem and answer")
    c.add_argument("--grader", default="exact")
    c.add_argument("--samples", type=int, default=1)
    c.add_argument("--limit", type=int)
    c.add_argument("--thinking", action="store_true")
    c.add_argument("--device", default=None)
    c.set_defaults(fn=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (P.PatchError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
