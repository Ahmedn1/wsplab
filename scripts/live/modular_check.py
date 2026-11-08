"""Check oracle repair visibility using calibration data only; no LLM calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
from pathlib import Path

import numpy as np
import torch

from wsplab.io.safetensors import Checkpoint
from wsplab.runners.torch import state_to_checkpoint
from wsplab.store import WeightStore

if __package__:
    from . import modular_net as task
else:
    import modular_net as task


def calibrate(seed):
    clean = task.trained(seed)
    broken = task.damage(clean)
    inputs, targets = task.queries(*task.image_splits(seed)["calibration"], seed=seed)
    clean_score, clean_branches = task.scores(clean, inputs, targets)
    baseline, broken_branches = task.scores(broken, inputs, targets)
    # Verify the oracle repairs are actually legal under the editor's edit policy.
    # Scalar output tensors in an earlier prototype failed this check.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)
        state_to_checkpoint(broken.state_dict(), path / "broken.safetensors")
        store = WeightStore(Checkpoint(path / "broken.safetensors"), path / "edits")
        for name, value in clean.state_dict().items():
            if not torch.equal(value, broken.state_dict()[name]):
                store.save_edit(name, value.detach().numpy())
    rows = []
    with torch.no_grad():
        original_outputs = broken(torch.from_numpy(inputs)).numpy()
    routes = inputs[:, 64:].argmax(1)
    for digit, fault in task.FAULTS.items():
        repaired = task.restore(broken, clean, [digit])
        score, _ = task.scores(repaired, inputs, targets)
        with torch.no_grad():
            outputs = repaired(torch.from_numpy(inputs)).numpy()
        unchanged = np.array_equal(outputs[routes != digit], original_outputs[routes != digit])
        rows.append({"digit": digit, "fault": fault, "gain": score - baseline,
                     "other_routes_unchanged": unchanged})
    # Check all 256 subsets by composing logits; routing never mixes branches.
    # The single-branch model checks above verify this composition's premise.
    with torch.no_grad():
        fixed_outputs = clean(torch.from_numpy(inputs)).numpy()
    max_error = 0.
    for mask in range(1 << len(rows)):
        selected = [d for d in task.FAULTS if mask & (1 << d)]
        outputs = np.where(np.isin(routes, selected)[:, None], fixed_outputs, original_outputs)
        observed = float((outputs.argmax(1) == targets).mean())
        predicted = baseline + sum(rows[d]["gain"] for d in selected)
        max_error = max(max_error, abs(observed - predicted))
    digest = hashlib.sha256()
    for name, tensor in clean.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().numpy().tobytes())
    return {"seed": seed, "split": "calibration", "queries": len(targets),
            "clean_sha256": digest.hexdigest(), "clean": clean_score, "baseline": baseline,
            "clean_per_branch": clean_branches, "broken_per_branch": broken_branches,
            "repairs": rows, "oracle_edits_legal": True, "max_subset_additivity_error": max_error,
            "accepted": clean_score >= 0.9 and all(r["gain"] >= 0.005 and r["other_routes_unchanged"] for r in rows)
                        and max_error < 1e-12}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[21, 22, 23])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = []
    for seed in args.seeds:
        result = calibrate(seed)
        results.append(result)
        print(json.dumps(result), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "benchmark": "routed-digits-v1", "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "task_source_sha256": hashlib.sha256(Path(task.__file__).read_bytes()).hexdigest(),
        "results": results, "accepted": all(r["accepted"] for r in results),
    }, indent=2) + "\n")
    return 0 if all(r["accepted"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
