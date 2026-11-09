"""Regression checks for the benchmark and read-only monitoring utilities."""
import json

import numpy as np
import torch
import modular_net as task
import monitor_run

from wsplab.io.safetensors import Checkpoint
from wsplab.runners.torch import state_to_checkpoint
from wsplab.store import WeightStore


def test_benchmark_images_are_unique_and_splits_are_disjoint():
    seen = set()
    for images, labels in task.image_splits().values():
        raw = {row.tobytes() for row in images}
        assert len(raw) == len(images)
        assert not raw & seen
        seen |= raw
        inputs, targets = task.queries(images, labels)
        assert len({row[:64].tobytes() for row in inputs}) == len(inputs)
        routes = inputs[:, 64:]
        assert np.all(routes.sum(1) == 1)
        counts = []
        for digit in range(10):
            mask = routes[:, digit] == 1
            counts.append(mask.sum())
            assert targets[mask].mean() == 0.5
        assert len(set(counts)) == 1


def test_restoring_one_branch_cannot_change_another_routes_outputs(tmp_path):
    torch.manual_seed(1)
    clean = task.build()
    broken = task.damage(clean)
    inputs, _ = task.queries(*task.image_splits()["calibration"])
    inputs = torch.from_numpy(inputs)
    with torch.no_grad():
        original = broken(inputs)
        for digit in task.FAULTS:
            restored = task.restore(broken, clean, [digit])(inputs)
            others = inputs[:, 64 + digit] == 0
            assert torch.equal(original[others], restored[others])
        all_fixed = task.restore(broken, clean, task.FAULTS)
        assert torch.equal(all_fixed(inputs), clean(inputs))
    state_to_checkpoint(broken.state_dict(), tmp_path / "broken.safetensors")
    store = WeightStore(Checkpoint(tmp_path / "broken.safetensors"), tmp_path / "edits")
    for name, value in clean.state_dict().items():
        if not torch.equal(value, broken.state_dict()[name]):
            store.save_edit(name, value.detach().numpy())


def test_monitor_requires_a_report_not_just_a_finished_editor(tmp_path, monkeypatch):
    (tmp_path / "session").mkdir()
    (tmp_path / "session/state.json").write_text(json.dumps({"done": True, "passes": 50}))
    monkeypatch.setattr(monitor_run, "process_identity", lambda pid: "start")
    assert monitor_run.snapshot(tmp_path, 123, "start")["status"] == "final_evaluation"
    monkeypatch.setattr(monitor_run, "process_identity", lambda pid: None)
    assert monitor_run.snapshot(tmp_path, 123, "start")["status"] == "process_exited_without_report"
    (tmp_path / "report.json").write_text(json.dumps({"verdict": "regressed"}))
    result = monitor_run.snapshot(tmp_path, 123, "start")
    assert result["status"] == "finished"
    assert result["report"]["verdict"] == "regressed"


def test_monitor_handles_partial_writes_and_pid_reuse(tmp_path, monkeypatch):
    (tmp_path / "session").mkdir()
    (tmp_path / "session/state.json").write_text('{"done":')
    monkeypatch.setattr(monitor_run, "process_identity", lambda pid: "new-start")
    result = monitor_run.snapshot(tmp_path, 123, "old-start")
    assert result["state"] is None
    assert result["status"] == "process_exited_without_report"
