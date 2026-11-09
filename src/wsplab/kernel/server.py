"""The persistent Python kernel, running inside the sandbox.

Requests arrive as JSON lines on stdin and replies leave on a private copy of stdout; the code's own stdout and stderr,
including those of subprocesses, go to a file, so nothing it prints can break the protocol. Variables persist between
requests. This is new code written to the paper's prompt: the authors did not release their kernel.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import traceback

MAX_REPLY = 1 << 22


def split_last_expression(code: str):
    """(statements, last expression or None): the value of a final expression is shown, as in a notebook."""
    tree = ast.parse(code, filename="<run>")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
        return tree, last
    return tree, None


def run_code(code: str, namespace: dict) -> None:
    try:
        tree, last = split_last_expression(code)
        exec(compile(tree, "<run>", "exec"), namespace)
        if last is not None:
            value = eval(compile(last, "<run>", "eval"), namespace)
            if value is not None:
                print(repr(value))
    except BaseException as e:  # noqa: BLE001 - everything the code raises is the editor's to read
        if isinstance(e, KeyboardInterrupt):
            raise
        frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename != __file__]
        text = "Traceback (most recent call last):\n" if frames else ""
        text += "".join(traceback.format_list(frames)) + "".join(traceback.format_exception_only(type(e), e))
        print(text, end="")


def build_namespace(config: dict) -> dict:
    from wsplab.io.safetensors import Checkpoint
    from wsplab.store import WeightStore

    store = WeightStore(Checkpoint(config["model"]), os.path.join(config["work"], "edited"), mode=config["mode"],
                        frozen=config["frozen"], refuse_rescale=config["refuse_rescale"], names=config.get("names"),
                        max_changed=config.get("max_changed"))
    ns: dict = {"__name__": "__main__", "NAMES": list(store.names)}
    try:
        import torch

        torch.set_grad_enabled(False)
        ns["torch"] = torch

        def wrap(fn):
            return lambda name: torch.from_numpy(fn(name))
    except ImportError:
        def wrap(fn):
            return fn
    import numpy as np

    ns.update(np=np, load=wrap(store.load), load_original=wrap(store.load_original), edited=store.edited)
    if config["mode"] != "read":
        ns.update(save_edit=store.save_edit, drop_edit=store.drop_edit)
    tokenizer = os.path.join(config["model"], "tokenizer.json")
    if os.path.exists(tokenizer):
        try:
            from tokenizers import Tokenizer

            ns["tok"] = Tokenizer.from_file(tokenizer)
        except ImportError:
            pass
    ns["_store"] = store
    if config.get("prelude"):
        exec(compile(config["prelude"], "<prelude>", "exec"), ns)
    return ns


def main(config: dict) -> None:
    replies = os.fdopen(os.dup(1), "w", buffering=1)
    requests = os.fdopen(os.dup(0), "r")
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    capture = os.path.join(config.get("scratch", "/tmp"), "kernel_output.txt")
    namespace = build_namespace(config)
    store = namespace.pop("_store")
    print(json.dumps({"ready": True}), file=replies)
    for line in requests:
        request = json.loads(line)
        store.locked = not request.get("allow_edits", True)
        fd = os.open(capture, os.O_CREAT | os.O_TRUNC | os.O_WRONLY)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)
        sys.stdout = os.fdopen(os.dup(1), "w", buffering=1)
        sys.stderr = sys.stdout
        run_code(request["code"], namespace)
        sys.stdout.flush()
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        with open(capture, errors="replace") as f:
            print(json.dumps({"output": f.read(MAX_REPLY)}), file=replies)
