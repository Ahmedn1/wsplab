"""Fixed scripted comparator for the Spurious ResNet task, written from the published description of the "A3" recipe
(no code was taken from that project): for each target class k, damp the fc weights of the penultimate channels that fire
on k's spurious photos but not on its clean photos, then boost the channels that carry the clean object.

It only sees the editors' work half of the practice data and is scored on the selection half with the official scorer.
It is a comparator and a positive control for the edit path, not an editor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wsplab import patch as P
from wsplab.io.safetensors import Checkpoint
from wsplab.runners.torch import TorchRunner

if __package__:
    from . import spurious_resnet as task
else:
    import spurious_resnet as task

SETTINGS = {"s1": {"cue": 32, "cue_scale": 0.25, "boost": 32, "boost_scale": None},
            "s2": {"cue": 50, "cue_scale": 0.0, "boost": 50, "boost_scale": 1.5}}


def features(runner, model, suite):
    """Penultimate (avgpool) activations through the official runner's hooks."""
    parts = [runner.capture(model, suite.inputs[i:i + 200], ["avgpool"])["avgpool"]
             for i in range(0, len(suite.inputs), 200)]
    return np.concatenate(parts).reshape(len(suite.inputs), -1)


def recipe(w, feats, targets, target_classes, cue, cue_scale, boost, boost_scale):
    w = w.copy()
    for k in target_classes:
        s = feats[targets == task.encode_spurious(k)]
        c = feats[targets == k]
        if not len(s) or not len(c):
            continue
        row = w[k]
        cue_score = np.where(row > 0, (s.mean(0) - c.mean(0)) * row, -np.inf)
        cues = np.argsort(-cue_score)[:cue]
        cues = cues[np.isfinite(cue_score[cues]) & (cue_score[cues] > 0)]
        before = c.mean(0) @ row
        row[cues] *= cue_scale
        obj_score = np.where(row > 0, (c.mean(0) - s.mean(0)) * row, -np.inf)
        obj_score[cues] = -np.inf
        objs = np.argsort(-obj_score)[:boost]
        objs = objs[np.isfinite(obj_score[objs]) & (obj_score[objs] > 0)]
        if boost_scale is None:  # restore the clean-photo logit lost by damping
            lost = before - c.mean(0) @ row
            gain = c.mean(0)[objs] @ row[objs]
            boost_scale = 1 + max(lost, 0) / gain if gain > 0 else 1.0
        row[objs] *= boost_scale
        w[k] = row
    return w


def run(fixture, out, seed=task.SEED, device="cuda"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model, practice, _, split = task.load_fixture(fixture)
    work, selection = practice.split(.5, seed)
    runner = TorchRunner(task.build, device)
    feats = features(runner, model, work)
    base = Checkpoint(model)
    w = base.float("fc.weight").astype(np.float32)
    baseline = selection.score(runner.call(model), selection.ids())
    sel_out = selection.outputs(runner.call(model), selection.ids())
    result = {"baseline": {"balanced": baseline.mean, **task.metrics(sel_out, selection.targets, split["target_classes"])}}
    for name, s in SETTINGS.items():
        edited = recipe(w, feats, work.targets, split["target_classes"], **s)
        patch = P.from_submission(base, {"fc.weight": edited}, out / name / "patch")
        weights = P.apply(patch, base, out / name / "model.safetensors")
        scores = selection.score(runner.call(weights), selection.ids())
        o = selection.outputs(runner.call(weights), selection.ids())
        result[name] = {"balanced": scores.mean, "changed_entries": patch.manifest["changed_entries_total"],
                        **task.metrics(o, selection.targets, split["target_classes"])}
    (out / "selection-results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.fixture, args.out), indent=2))
