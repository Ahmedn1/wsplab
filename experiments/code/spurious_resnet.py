"""Unmodified torchvision ResNet-50 on Spurious ImageNet: a frozen fixture, no editor calls and no training.

Spurious ImageNet (Neuhaus et al., ICCV 2023) holds real OpenImages photos that show a class's spurious cue (the water
behind a snorkel, the ice behind a puck) without the object itself. The public ResNet-50 calls many of them the absent
class. The editors must cut those false predictions while keeping ordinary accuracy, especially on the same classes.

Data on disk (never shipped): ~/.wsplab/data/spurious_imagenet/images (Dropbox mirror named in the official README,
archive sha256 ce3ef1b6...), ImageNetV2 matched-frequency, threshold0.7 and top-images (deduplicated by content) for
clean photos, and the torchvision IMAGENET1K_V1 checkpoint (sha256 0676ba61...).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torchvision

from wsplab.eval.suites import ArrayScorer, check_disjoint
from wsplab.runners.torch import TorchRunner, state_to_checkpoint

SEED = 20261108
DATA = Path(os.path.expanduser("~/.wsplab/data"))
CHECKPOINT_SHA = "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
V2_VARIANTS = ("matched-frequency", "threshold0.7", "top-images")
TARGET_CLASSES = 50          # classes whose spurious photos the editors see; the other 50 are an unseen transfer test
SPURIOUS_PRACTICE = 25       # of 75 photos per target class; the other 50 are evaluation
CLEAN_PRACTICE_TARGET = 15   # clean photos per target class given as practice
CLEAN_PRACTICE_OTHER = 3     # clean photos per other class given as practice
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, algorithm).hexdigest()


class Net(torchvision.models.ResNet):
    """Standard ResNet-50 that takes uint8 NCHW 224x224 RGB and normalizes inside, so tensor names stay torchvision's."""

    def __init__(self):
        super().__init__(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255, persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255, persistent=False)

    def forward(self, x):
        return super().forward((x.float() - self.mean) / self.std)


def build():
    return Net()


def encode_spurious(k: int) -> int:
    """Targets: a clean photo of class y has target y; a spurious photo of class k has target -(k+1)."""
    return -(k + 1)


def make_metric(target_classes):
    """Per-item scores whose mean is the average of three group means: spurious photos of target classes not called
    that class, clean target-class photos correct, clean other-class photos correct. Each group counts one third.
    Spurious photos of the unseen classes get weight 0 here; they are the separately reported transfer test."""
    targets_set = np.array(sorted(target_classes))

    def balanced(outputs, targets):
        pred = np.asarray(outputs).argmax(-1)
        spurious = targets < 0
        cls = np.where(spurious, -targets - 1, targets)
        ok = np.where(spurious, pred != cls, pred == cls).astype(np.float64)
        tgt = np.isin(cls, targets_set)
        group = np.where(spurious, np.where(tgt, 0, 3), np.where(tgt, 1, 2))
        present = np.unique(group[group < 3])
        weights = np.zeros(len(group))
        for g in present:
            weights[group == g] = len(group) / (len(present) * (group == g).sum())
        return ok * weights

    return balanced


def metrics(outputs, targets, target_classes):
    """The pre-registered components, computed outside the editors."""
    pred = np.asarray(outputs).argmax(-1)
    spurious = targets < 0
    cls = np.where(spurious, -targets - 1, targets)
    tgt = np.isin(cls, list(target_classes))
    out = {}
    if (spurious & tgt).any():
        out["target_spurious_hit"] = float((pred[spurious & tgt] == cls[spurious & tgt]).mean())
    if (spurious & ~tgt).any():
        out["unseen_spurious_hit"] = float((pred[spurious & ~tgt] == cls[spurious & ~tgt]).mean())
    if (~spurious & tgt).any():
        out["target_clean_top1"] = float((pred[~spurious & tgt] == cls[~spurious & tgt]).mean())
    if (~spurious).any():
        out["clean_top1"] = float((pred[~spurious] == cls[~spurious]).mean())
    return out


def _load_spurious():
    root = DATA / "spurious_imagenet/images"
    rows = []
    for d in sorted(os.listdir(root)):
        m = re.match(r"\d+_class_(\d+)_(.*)_component_\d+$", d)
        if m:
            for f in sorted((root / d).glob("*.png"), key=lambda p: int(p.stem)):
                rows.append((int(m.group(1)), m.group(2), f))
    return rows


def _load_clean():
    """ImageNetV2 photos from the three variants, plus the ImageNet validation photos of the 100 spurious classes, one copy
    per distinct file content."""
    seen, rows = {}, []
    sources = [sorted((DATA / f"imagenetv2/imagenetv2-{v}-format-val").glob("*/*.jpeg")) for v in V2_VARIANTS]
    sources.append(sorted((DATA / "imagenet_val/images").glob("*/*.JPEG")))
    for files in sources:
        for f in files:
            h = digest(f)
            if h not in seen:
                seen[h] = f
                rows.append((int(f.parent.name), f, h))
    return rows


def _pixels(path, spurious):
    """Official preprocessing: spurious photos are resized to 224x224 without a crop; clean photos are resized to 256 and
    centre-cropped to 224. Bicubic, antialiased. Returns uint8 CHW."""
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import v2 as T

    img = T.PILToTensor()(Image.open(path).convert("RGB"))
    if spurious:
        img = T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True)(img)
    else:
        img = T.CenterCrop(224)(T.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True)(img))
    return img.numpy()


def prepare(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "manifest.json").exists():
        raise ValueError("fixture already frozen; use its existing manifest")
    ckpt = DATA / "torch/resnet50-0676ba61.pth"
    if digest(ckpt) != CHECKPOINT_SHA:
        raise ValueError("checkpoint hash mismatch")
    rng = np.random.default_rng(SEED)
    spurious = _load_spurious()
    classes = sorted({k for k, _, _ in spurious})
    if len(classes) != 100:
        raise ValueError(f"expected 100 spurious classes, found {len(classes)}")
    target = sorted(rng.choice(classes, TARGET_CLASSES, replace=False).tolist())
    clean = _load_clean()

    practice, evaluation = [], []          # (path, target, spurious?, sha256)
    for k in classes:
        photos = [f for c, _, f in spurious if c == k]
        order = rng.permutation(len(photos))
        for rank, i in enumerate(order):
            row = (photos[i], encode_spurious(k), True, digest(photos[i]))
            (practice if k in target and rank < SPURIOUS_PRACTICE else evaluation).append(row)
    for y in range(1000):
        photos = [(f, h) for c, f, h in clean if c == y]
        order = rng.permutation(len(photos))
        keep = CLEAN_PRACTICE_TARGET if y in target else CLEAN_PRACTICE_OTHER
        for rank, i in enumerate(order):
            row = (photos[i][0], y, False, photos[i][1])
            (practice if rank < keep else evaluation).append(row)
    # 22 Spurious ImageNet photos sit in two classes' folders; never evaluate on a photo an editor can see.
    seen = {r[3] for r in practice}
    dropped = sum(r[3] in seen for r in evaluation)
    evaluation = [r for r in evaluation if r[3] not in seen]

    for name, rows in (("practice", practice), ("evaluation", evaluation)):
        order = rng.permutation(len(rows))
        rows = [rows[i] for i in order]
        x = np.lib.format.open_memmap(folder / f"{name}_inputs.npy", "w+", np.uint8, (len(rows), 3, 224, 224))
        for i, r in enumerate(rows):
            x[i] = _pixels(r[0], r[2])
        x.flush()
        del x
        np.save(folder / f"{name}_targets.npy", np.array([r[1] for r in rows], dtype=np.int64))
        (folder / f"{name}_sources.json").write_text(json.dumps(
            [{"file": str(r[0].relative_to(DATA)), "sha256": r[3]} for r in rows]) + "\n")

    net = build()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=True)
    state_to_checkpoint(state, folder / "model.safetensors")
    names = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
    (folder / "classes.json").write_text(json.dumps(names) + "\n")
    (folder / "split.json").write_text(json.dumps({"seed": SEED, "target_classes": target,
                                                   "unseen_classes": [k for k in classes if k not in target]}) + "\n")
    files = ["model.safetensors", "classes.json", "split.json"] + [
        f"{n}_{p}" for n in ("practice", "evaluation") for p in ("inputs.npy", "targets.npy", "sources.json")]
    manifest = {"model": "torchvision resnet50 IMAGENET1K_V1", "checkpoint_sha256": CHECKPOINT_SHA, "seed": SEED,
                "spurious_archive_sha256": "ce3ef1b6bc2099d179931780be84cf0a6c1c63a855e8c54f4a02868b0ebe15f6",
                "clean_sources": [f"ImageNetV2 {v}" for v in V2_VARIANTS] + ["ImageNet-1k validation, 100 spurious classes"],
                "counts": {"practice": len(practice), "evaluation": len(evaluation),
                           "evaluation_copies_dropped_as_seen_in_practice": dropped},
                "files": {n: digest(folder / n) for n in files}}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return verify(folder)


def load_fixture(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"fixture changed: {name}")
    split = json.loads((folder / "split.json").read_text())
    metric = make_metric(split["target_classes"])

    def make(name, prefix):
        x = np.load(folder / f"{name}_inputs.npy", mmap_mode="r")
        y = np.load(folder / f"{name}_targets.npy")
        return ArrayScorer(x, y, metric=metric, name="Spurious ImageNet + ImageNetV2", batch=200, prefix=prefix)

    return folder / "model.safetensors", make("practice", "item"), make("evaluation", "eval"), split


def verify(folder, device="cuda"):
    """Baseline on both partitions; checks the loading reproduces the published behaviour, not a favourable split."""
    folder = Path(folder)
    model, practice, evaluation, split = load_fixture(folder)
    check_disjoint(practice, evaluation)
    runner = TorchRunner(build, device)
    call = runner.call(model)
    result = {}
    for name, suite in (("practice", practice), ("evaluation", evaluation)):
        out = suite.outputs(call, suite.ids())
        result[name] = {"items": len(suite.ids()), "balanced": float(suite.score(call, suite.ids()).mean),
                        **metrics(out, suite.targets, split["target_classes"])}
        np.save(folder / f"baseline_{name}_logits.npy", out.astype(np.float16))
    ev = result["evaluation"]
    # Published: ~48% spurious hit over the 100 classes; ~63% top-1 on ImageNetV2 matched-frequency (the other variants are easier).
    if not (0.35 < ev["target_spurious_hit"] < 0.65 and 0.55 < ev["clean_top1"] < 0.80):
        raise ValueError(f"baseline does not look like the public model: {ev}")
    (folder / "baseline.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.folder) if args.verify else prepare(args.folder), indent=2))
