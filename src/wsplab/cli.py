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
