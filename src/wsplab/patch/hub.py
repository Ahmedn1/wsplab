"""Share patches on the Hugging Face Hub. A patch is a few small files, so sharing one needs no copy of the model.

A pulled patch is untrusted: it is checked on load, and its hashes prove what it is, not that it is safe or that its claims
are true. Measure a patch yourself before you rely on it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .format import Patch

FILES = ("edit.npz", "manifest.json", "patch.json")
TAG = "wsplab-patch"


def model_card(patch: Patch) -> str:
    meta = patch.meta
    base = meta.get("base") or {}
    evidence = meta.get("evidence") or {}
    front = ["---", "tags:", f"- {TAG}", "- weight-space-programming"]
    if base.get("repo"):
        front.append(f"base_model: {base['repo']}")
    if meta.get("license"):
        front.append(f"license: {str(meta['license']).lower()}")
    front.append("---")
    title = meta.get("name") or "A wsplab patch"
    lines = [f"# {title}", "", meta.get("description") or "A weight patch, made with wsplab.", "",
             f"- Base model: `{base.get('repo', 'unknown')}` at `{base.get('revision') or 'unknown revision'}`",
             f"- Tensors changed: {len(patch.tensors)}, entries changed: {patch.manifest.get('changed_entries_total', 'unknown'):,}",
             f"- Verdict when made: **{evidence.get('verdict', 'unmeasured')}**", f"- Made by: {meta.get('editor', 'unknown')}", ""]
    if meta.get("attribution"):
        lines += [f"Attribution: {meta['attribution']}", ""]
    lines += ["## Use", "", "```sh", "pip install wsplab", "wsplab patch pull <this repo> --out patch",
              "wsplab patch apply patch --base <the base model> --out edited", "```", "",
              "Hashes prove that this is the edit that was published and that it fits the base model. They do not prove it is "
              "safe or that its claims hold: measure it yourself.", ""]
    return "\n".join(front + [""] + lines)


def push(patch_dir: str | os.PathLike, repo: str, *, private: bool = False, api=None) -> str:
    """Publish a patch and return its URL. This makes it public unless `private`."""
    patch = Patch.load(patch_dir)
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    api.create_repo(repo, private=private, exist_ok=True)
    for name in FILES:
        path = Path(patch_dir) / name
        if path.exists():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=repo)
    api.upload_file(path_or_fileobj=model_card(patch).encode(), path_in_repo="README.md", repo_id=repo)
    return f"https://huggingface.co/{repo}"


def pull(repo: str, out: str | os.PathLike, *, revision: str | None = None, download=None) -> Patch:
    """Download a patch into `out` and validate it. Only the patch's own files are fetched: nothing else from the repo."""
    if download is None:
        from huggingface_hub import hf_hub_download as download
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        got = download(repo_id=repo, filename=name, revision=revision, local_dir=out)
        if Path(got).resolve().parent != out.resolve():
            raise ValueError(f"{name} was saved outside {out}")
    json.loads((out / "patch.json").read_text())
    return Patch.load(out)
