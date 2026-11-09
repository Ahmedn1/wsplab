"""Unmodified public CIFAR-100 ResNet-20 fixture; no editor calls or training."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import torch

from wsplab.eval.suites import ArrayScorer, check_disjoint
from wsplab.runners.torch import TorchRunner, state_to_checkpoint

if __package__:
    from .vendor.cifar_resnet import BasicBlock, CifarResNet
else:
    from vendor.cifar_resnet import BasicBlock, CifarResNet


SEED = 20261106
MEAN = np.array([.5070, .4865, .4409], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([.2673, .2564, .2761], dtype=np.float32).reshape(1, 3, 1, 1)
REVISION = "786c16252c0fc58ee9adac063f8337cc4a7a497a"


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, algorithm).hexdigest()


def build():
    return CifarResNet(BasicBlock, [3, 3, 3], num_classes=100)


def convert_hf(folder):
    """Decode the version-pinned alternate distribution; requires pyarrow and Pillow."""
    import io

    import pyarrow.parquet as pq
    from PIL import Image

    folder = Path(folder)
    provenance = json.loads((folder / "source/hf-provenance.json").read_text())
    if digest(folder / "test.parquet") != provenance["sha256"]:
        raise ValueError("alternate dataset distribution checksum mismatch")
    table = pq.read_table(folder / "test.parquet")
    features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    names = features["fine_label"]["names"]
    images = np.stack([np.asarray(Image.open(io.BytesIO(x["bytes"])).convert("RGB")).transpose(2, 0, 1)
                       for x in table["img"].to_pylist()])
    rows = np.column_stack([np.array(table["coarse_label"]), np.array(table["fine_label"]),
                            images.reshape(len(images), -1)]).astype(np.uint8)
    (folder / "test.bin").write_bytes(rows.tobytes())
    (folder / "fine_label_names.txt").write_text("\n".join(names) + "\n")
    provenance["test_bin_sha256"] = digest(folder / "test.bin")
    provenance["names_sha256"] = digest(folder / "fine_label_names.txt")
    (folder / "source/hf-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def split_indices(labels, seed=SEED):
    rng = np.random.default_rng(seed)
    practice, evaluation = [], []
    for label in range(100):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) != 100:
            raise ValueError("expected 100 official test examples per class")
        practice.extend(indices[:20])
        evaluation.extend(indices[20:])
    return rng.permutation(practice), rng.permutation(evaluation)


def load_fixture(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"fixture changed: {name}")
    inputs = np.load(folder / "inputs.npy", mmap_mode="r")
    targets = np.load(folder / "targets.npy")
    splits = json.loads((folder / "indices.json").read_text())
    def make(ids, prefix):
        return ArrayScorer(inputs[ids], targets[ids], name="CIFAR-100 fine labels", batch=64, prefix=prefix)
    return folder / "model.safetensors", make(splits["practice"], "item"), make(splits["evaluation"], "eval")


def prepare(folder):
    folder = Path(folder)
    if (folder / "manifest.json").exists():
        raise ValueError("fixture already frozen; use its existing manifest")
    archive = folder / "cifar-100-binary.tar.gz"
    checkpoint = folder / "cifar100_resnet20-23dac2f1.pt"
    if not digest(checkpoint).startswith("23dac2f1"):
        raise ValueError("public checkpoint hash mismatch")
    if archive.exists():
        if digest(archive, "md5") != "03b5dce01913d631647c71ecec9e9cb8":
            raise ValueError("CIFAR archive checksum mismatch")
        # Only read named archive members; no extraction or pickle execution.
        with tarfile.open(archive) as tar:
            raw = tar.extractfile("cifar-100-binary/test.bin").read()
            names = tar.extractfile("cifar-100-binary/fine_label_names.txt").read().decode().splitlines()
        dataset_provenance = {"archive_sha256": digest(archive), "source": "original binary archive"}
    else:
        dataset_provenance = json.loads((folder / "source/hf-provenance.json").read_text())
        if (digest(folder / "test.bin") != dataset_provenance["test_bin_sha256"]
                or digest(folder / "fine_label_names.txt") != dataset_provenance["names_sha256"]):
            raise ValueError("converted dataset changed")
        raw = (folder / "test.bin").read_bytes()
        names = (folder / "fine_label_names.txt").read_text().splitlines()
    rows = np.frombuffer(raw, np.uint8).reshape(10000, 3074)
    inputs = (rows[:, 2:].reshape(-1, 3, 32, 32).astype(np.float32) / 255 - MEAN) / STD
    targets = rows[:, 1].astype(np.int64)
    practice, evaluation = split_indices(targets)
    np.save(folder / "inputs.npy", inputs)
    np.save(folder / "targets.npy", targets)
    (folder / "indices.json").write_text(json.dumps({
        "seed": SEED, "practice": practice.tolist(), "evaluation": evaluation.tolist()}, indent=2) + "\n")
    (folder / "classes.json").write_text(json.dumps(names, indent=2) + "\n")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net = build()
    net.load_state_dict(state, strict=True)
    state_to_checkpoint(state, folder / "model.safetensors")
    files = ["model.safetensors", "inputs.npy", "targets.npy", "indices.json", "classes.json"]
    manifest = {"model": "cifar100_resnet20", "source_revision": REVISION,
                "checkpoint_sha256": digest(checkpoint), "dataset_provenance": dataset_provenance,
                "seed": SEED, "mean": MEAN.flatten().tolist(), "std": STD.flatten().tolist(),
                "parameters": sum(p.numel() for p in net.parameters()),
                "files": {name: digest(folder / name) for name in files}}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    model, practice_suite, evaluation_suite = load_fixture(folder)
    check_disjoint(practice_suite, evaluation_suite)
    work, selection = practice_suite.split(.5, SEED)
    check_disjoint(work, selection)
    runner = TorchRunner(build)
    suite = ArrayScorer(inputs, targets, batch=64)
    correct = suite.outputs(runner.call(model), suite.ids()).argmax(1) == targets
    full_accuracy = float(correct.mean())
    # Detect wrong preprocessing/loading, not select the most favorable task.
    if abs(full_accuracy - .6883) > .015:
        raise ValueError(f"baseline {full_accuracy:.4f} does not reproduce published 0.6883")
    result = {"full_standard_test_accuracy": full_accuracy, "published_accuracy": .6883,
              "practice_pool_accuracy": float(correct[practice].mean()),
              "external_evaluation_accuracy": float(correct[evaluation].mean()),
              "external_evaluation_items": len(evaluation), "practice_items": len(work.ids()),
              "selection_items": len(selection.ids()), "parameters": manifest["parameters"],
              "unaltered_public_checkpoint": True, "content_disjoint": True}
    np.save(folder / "baseline_correct.npy", correct)
    (folder / "baseline.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--convert-hf", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(convert_hf(args.folder) if args.convert_hf else prepare(args.folder), indent=2))
