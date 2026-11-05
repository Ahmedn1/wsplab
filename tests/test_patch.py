"""Patch format: round trips across dtypes and forms, the upstream bugs, and hostile patches."""

import json
import shutil

import numpy as np
import pytest

from wsplab import patch as P
from wsplab.io import Checkpoint, bf16_bits, bf16_to_f32, cast_to, save, to_bits

rng = np.random.default_rng(0)


def bf16(x):
    return bf16_bits(np.asarray(x, dtype=np.float32))


def write_model(path, tensors):
    path.parent.mkdir(parents=True, exist_ok=True)
    save(path, tensors)
    return Checkpoint(path)


def base_tensors():
    return {
        "w": ("BF16", bf16(rng.standard_normal((64, 48)))),
        "small": ("BF16", bf16(rng.standard_normal((4, 6)))),
        "bias": ("BF16", bf16(rng.standard_normal(32))),
        "conv": ("BF16", bf16(rng.standard_normal((8, 1, 4)))),
        "f32": ("F32", rng.standard_normal((16, 16)).astype(np.float32)),
        "f16": ("F16", rng.standard_normal((16, 16)).astype(np.float16)),
        "f64": ("F64", rng.standard_normal(10)),
    }


def roundtrip(tmp_path, sub):
    base = write_model(tmp_path / "base.safetensors", base_tensors())
    patch = P.from_submission(base, sub(base), tmp_path / "patch")
    P.verify(patch, base)
    out = P.apply(patch, base, tmp_path / "out.safetensors")
    return base, patch, Checkpoint(out)


def test_sparse_roundtrip_and_other_tensors_untouched(tmp_path):
    def sub(base):
        w = base.float("w")
        w[:, 3] = 0
        return {"w": w}

    base, patch, out = roundtrip(tmp_path, sub)
    assert patch.tensors[0]["form"] == "sparse"
    assert np.array_equal(out.float("w")[:, 3], np.zeros(64))
    for name in ("small", "bias", "conv", "f32", "f16", "f64"):
        assert np.array_equal(to_bits(out.raw(name)[1]), to_bits(base.raw(name)[1]))


def test_rank_one_edit_is_stored_low_rank(tmp_path):
    def sub(base):
        w = base.float("w").astype(np.float64)
        d = rng.standard_normal(64)
        d /= np.linalg.norm(d)
        return {"w": (w - 1.5 * np.outer(d, d @ w)).astype(np.float32)}

    _, patch, _ = roundtrip(tmp_path, sub)
    assert patch.tensors[0]["form"] == "lowrank" and patch.tensors[0]["rank"] == 1


def test_true_rank_above_limit_becomes_dense(tmp_path):
    _, patch, out = roundtrip(tmp_path, lambda b: {"w": b.float("w") + rng.standard_normal((64, 48)).astype(np.float32)})
    assert patch.tensors[0]["form"] == "dense"
    assert patch.tensors[0]["edited_sha256"] == P.format.sha(to_bits(out.raw("w")[1]))


def test_full_rank_tensor_smaller_than_rank_limit(tmp_path):
    """Upstream: s[r] raises IndexError when min(shape) <= 8 and the edit has full rank."""
    _, patch, _ = roundtrip(tmp_path, lambda b: {"small": b.float("small") + rng.standard_normal((4, 6)).astype(np.float32)})
    assert patch.tensors[0]["form"] in ("lowrank", "dense")


def test_one_dimensional_tensor(tmp_path):
    """Upstream: mask.any(1) raises AxisError on a 1-D tensor."""
    _, patch, _ = roundtrip(tmp_path, lambda b: {"bias": b.float("bias") * 2 + 1})
    assert "changed_rows" not in patch.tensors[0]


def test_three_dimensional_tensor(tmp_path):
    """Upstream: a dense 3-D edit goes to a batched SVD."""
    _, patch, _ = roundtrip(tmp_path, lambda b: {"conv": b.float("conv") + 1})
    assert patch.tensors[0]["form"] == "dense"


@pytest.mark.parametrize("name", ["f32", "f16", "f64"])
def test_other_dtypes(tmp_path, name):
    """Upstream asserts BF16; Qwen3.5 itself has F32 tensors."""
    _, patch, out = roundtrip(tmp_path, lambda b: {name: b.float(name) * 1.25})
    assert patch.tensors[0]["edited_sha256"] == P.format.sha(to_bits(out.raw(name)[1]))


def test_edit_that_rounds_back_is_recorded_not_a_crash(tmp_path):
    """Upstream: changed == subs fails when a submission rounds back to the base."""
    base = write_model(tmp_path / "base.safetensors", base_tensors())
    tiny = base.float("w") * (1 + 2.0**-20)
    patch = P.from_submission(base, {"w": tiny}, tmp_path / "patch")
    assert patch.tensors == [] and patch.manifest["no_effect"] == ["w"]


def test_non_finite_values_are_refused(tmp_path):
    base = write_model(tmp_path / "base.safetensors", base_tensors())
    w = base.float("w")
    w[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        P.from_submission(base, {"w": w}, tmp_path / "patch")


def test_float32_max_overflowing_bfloat16_is_refused():
    with pytest.raises(ValueError, match="overflow"):
        cast_to(np.array([np.finfo(np.float32).max], np.float32), "BF16")


def test_bf16_rounding_matches_torch():
    torch = pytest.importorskip("torch")
    x = rng.standard_normal(4096).astype(np.float32)
    want = torch.from_numpy(x).to(torch.bfloat16).view(torch.uint16).numpy()
    assert np.array_equal(bf16_bits(x), want)
    assert np.array_equal(bf16_to_f32(want), torch.from_numpy(x).to(torch.bfloat16).float().numpy())


def test_index_width_chosen_by_size():
    assert P.diff._flat_index(np.array([1]), 2**31 - 1).dtype == np.int32
    assert P.diff._flat_index(np.array([1]), 2**31).dtype == np.int64


def test_from_checkpoints_matches_edited_model(tmp_path):
    tensors = base_tensors()
    base = write_model(tmp_path / "a.safetensors", tensors)
    edited_w = base.float("w")
    edited_w[5] = 1
    tensors = dict(tensors, w=("BF16", bf16(edited_w)))
    edited = write_model(tmp_path / "b.safetensors", tensors)
    patch = P.from_checkpoints(base, edited, tmp_path / "patch")
    assert [t["name"] for t in patch.tensors] == ["w"]
    out = Checkpoint(P.apply(patch, base, tmp_path / "c.safetensors"))
    assert (tmp_path / "c.safetensors").read_bytes() != (tmp_path / "a.safetensors").read_bytes()
    assert np.array_equal(to_bits(out.raw("w")[1]), to_bits(edited.raw("w")[1]))


def sharded(tmp_path):
    t = base_tensors()
    folder = tmp_path / "model"
    folder.mkdir()
    save(folder / "model-00001-of-00002.safetensors", {"w": t["w"], "bias": t["bias"]})
    save(folder / "model-00002-of-00002.safetensors", {"f32": t["f32"]})
    index = {"weight_map": {"w": "model-00001-of-00002.safetensors", "bias": "model-00001-of-00002.safetensors",
                            "f32": "model-00002-of-00002.safetensors"}}
    (folder / "model.safetensors.index.json").write_text(json.dumps(index))
    (folder / "config.json").write_text("{}")
    return Checkpoint(folder)


def test_sharded_apply_links_untouched_files(tmp_path):
    base = sharded(tmp_path)
    w = base.float("w")
    w[:, 0] = 2
    patch = P.from_submission(base, {"w": w}, tmp_path / "patch")
    out = P.apply(patch, base, tmp_path / "out")
    src = tmp_path / "model" / "model-00002-of-00002.safetensors"
    assert (out / "model-00002-of-00002.safetensors").stat().st_ino == src.stat().st_ino
    edited_shard = out / "model-00001-of-00002.safetensors"
    assert edited_shard.stat().st_ino != (tmp_path / "model" / edited_shard.name).stat().st_ino
    assert (out / "config.json").exists()
    assert np.array_equal(Checkpoint(out).float("w")[:, 0], np.full(64, 2.0))
    assert Checkpoint(tmp_path / "model").float("w")[0, 0] != 2.0


def test_apply_never_copies_what_merely_sits_beside_a_model(tmp_path):
    """Found by an agent: applying a patch to a lone weights file copied the evaluation data into the output."""
    base = write_model(tmp_path / "net.safetensors", base_tensors())
    for name in ("heldout.npz", "practice.npz", "notes.json"):
        (tmp_path / name).write_text("private")
    patch = P.from_submission(base, {"bias": base.float("bias") + 1}, tmp_path / "patch")
    out = P.apply(patch, base, tmp_path / "edited")
    assert sorted(f.name for f in out.iterdir()) == ["net.safetensors"]


def test_a_model_folder_keeps_only_its_model_files(tmp_path):
    base = sharded(tmp_path)
    (tmp_path / "model" / "tokenizer.json").write_text("{}")
    (tmp_path / "model" / "heldout.json").write_text("private")
    patch = P.from_submission(base, {"w": base.float("w") + 1}, tmp_path / "patch")
    names = {f.name for f in P.apply(patch, base, tmp_path / "out").iterdir()}
    assert "tokenizer.json" in names and "config.json" in names and "heldout.json" not in names


def test_apply_can_copy_instead_of_linking(tmp_path):
    base = sharded(tmp_path)
    patch = P.from_submission(base, {"w": base.float("w") + 1}, tmp_path / "patch")
    out = P.apply(patch, base, tmp_path / "out", link=False)
    src = tmp_path / "model" / "model-00002-of-00002.safetensors"
    assert (out / src.name).stat().st_ino != src.stat().st_ino


def test_wrong_base_is_refused(tmp_path):
    base = write_model(tmp_path / "a.safetensors", base_tensors())
    patch = P.from_submission(base, {"w": base.float("w") + 1}, tmp_path / "patch")
    other = write_model(tmp_path / "b.safetensors", base_tensors())
    with pytest.raises(P.PatchError, match="differs from the one"):
        P.verify(patch, other)


def make_patch(tmp_path):
    base = write_model(tmp_path / "a.safetensors", base_tensors())
    w = base.float("w")
    w[:, 1] = 0
    return base, P.from_submission(base, {"w": w}, tmp_path / "patch"), tmp_path / "patch"


def test_tampered_archive_is_refused(tmp_path):
    _, _, folder = make_patch(tmp_path)
    with open(folder / "edit.npz", "ab") as f:
        f.write(b"x")
    with pytest.raises(P.PatchError, match="was changed"):
        P.Patch.load(folder)


def test_tampered_manifest_is_refused(tmp_path):
    _, _, folder = make_patch(tmp_path)
    m = json.loads((folder / "manifest.json").read_text())
    m["tensors"][0]["changed_entries"] = 1
    (folder / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(P.PatchError, match="was changed"):
        P.Patch.load(folder)


@pytest.mark.parametrize("name", ["../evil", "a/../../b", "/etc/passwd", "..\\x", ".hidden", "a//b", ""])
def test_path_traversal_names_are_refused(tmp_path, name):
    _, _, folder = make_patch(tmp_path)
    m = json.loads((folder / "manifest.json").read_text())
    m["tensors"][0]["name"] = name
    (folder / "manifest.json").write_text(json.dumps(m))
    (folder / "patch.json").unlink()
    with pytest.raises(P.PatchError, match="tensor name"):
        P.Patch.load(folder)


def test_jax_style_names_with_slashes_work(tmp_path):
    base = write_model(tmp_path / "a.safetensors", {"actor/dense1/kernel": ("F32", np.ones((4, 3), np.float32))})
    w = base.float("actor/dense1/kernel")
    w[0, 0] = 2
    patch = P.from_submission(base, {"actor/dense1/kernel": w}, tmp_path / "p")
    assert Checkpoint(P.apply(patch, base, tmp_path / "o.safetensors")).float("actor/dense1/kernel")[0, 0] == 2


def test_index_outside_tensor_is_refused(tmp_path):
    base, _, folder = make_patch(tmp_path)
    z = dict(np.load(folder / "edit.npz"))
    z["t0_idx"] = z["t0_idx"] + 10_000_000
    np.savez(folder / "edit.npz", **z)
    (folder / "patch.json").unlink()
    with pytest.raises(P.PatchError, match="outside the tensor"):
        P.verify(P.Patch.load(folder), base)


def test_pickled_arrays_are_refused(tmp_path):
    base, _, folder = make_patch(tmp_path)
    z = dict(np.load(folder / "edit.npz"))
    z["t0_idx"] = np.array([object()], dtype=object)
    np.savez(folder / "edit.npz", **z)
    (folder / "patch.json").unlink()
    with pytest.raises(ValueError, match="pickle"):
        P.verify(P.Patch.load(folder), base)


def test_decompression_bomb_is_refused(tmp_path):
    _, _, folder = make_patch(tmp_path)
    np.savez_compressed(folder / "edit.npz", t0_idx=np.zeros(1 << 26, np.int64))
    (folder / "patch.json").unlink()
    with pytest.raises(P.PatchError, match="uncompressed"):
        P.Patch.load(folder)


def test_stack_disjoint_and_chained(tmp_path):
    base = write_model(tmp_path / "a.safetensors", base_tensors())
    w1 = base.float("w")
    w1[:, 0] = 1
    p1 = P.from_submission(base, {"w": w1}, tmp_path / "p1")
    p2 = P.from_submission(base, {"bias": base.float("bias") + 1}, tmp_path / "p2")
    both = P.stack([p1, p2], base, tmp_path / "both")
    assert sorted(t["name"] for t in both.tensors) == ["bias", "w"]

    mid = Checkpoint(P.apply(p1, base, tmp_path / "mid.safetensors"))
    w2 = mid.float("w")
    w2[:, 1] = 2
    p3 = P.from_submission(mid, {"w": w2}, tmp_path / "p3")
    chained = P.stack([p1, p3], base, tmp_path / "chained")
    out = Checkpoint(P.apply(chained, base, tmp_path / "out.safetensors"))
    assert np.array_equal(out.float("w")[:, :2], np.stack([np.ones(64), np.full(64, 2.0)], 1))


def test_stack_refuses_conflicts(tmp_path):
    base = write_model(tmp_path / "a.safetensors", base_tensors())
    p1 = P.from_submission(base, {"w": base.float("w") + 1}, tmp_path / "p1")
    p2 = P.from_submission(base, {"w": base.float("w") + 2}, tmp_path / "p2")
    with pytest.raises(P.PatchError, match="conflict"):
        P.stack([p1, p2], base, tmp_path / "both")


def test_cli_roundtrip(tmp_path, capsys):
    from wsplab.cli import main

    base = write_model(tmp_path / "a.safetensors", base_tensors())
    sub = tmp_path / "sub"
    sub.mkdir()
    w = base.float("w")
    w[:, 2] = 3
    np.save(sub / "w.npy", w)
    args = ["patch", "create", "--base", str(tmp_path / "a.safetensors"), "--submission", str(sub), "--out",
            str(tmp_path / "p"), "--name", "demo"]
    assert main(args) == 0
    assert main(["patch", "verify", str(tmp_path / "p"), "--base", str(tmp_path / "a.safetensors")]) == 0
    assert main(["patch", "apply", str(tmp_path / "p"), "--base", str(tmp_path / "a.safetensors"), "--out",
                 str(tmp_path / "o.safetensors")]) == 0
    assert main(["patch", "inspect", str(tmp_path / "p")]) == 0
    assert '"name": "demo"' in capsys.readouterr().out
    shutil.rmtree(tmp_path / "p")
    assert main(["patch", "verify", str(tmp_path / "p")]) == 1
