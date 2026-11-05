import numpy as np
import pytest

from wsplab.io import Checkpoint, bf16_bits, save
from wsplab.store import EditRefused, EditsLocked, WeightStore

rng = np.random.default_rng(1)


@pytest.fixture
def ckpt(tmp_path):
    t = {"w": ("BF16", bf16_bits(rng.standard_normal((8, 4)).astype(np.float32))),
         "frozen.weight": ("BF16", bf16_bits(rng.standard_normal(4).astype(np.float32))),
         "a/b": ("F32", rng.standard_normal(3).astype(np.float32))}
    save(tmp_path / "m.safetensors", t)
    return Checkpoint(tmp_path / "m.safetensors")


def store(ckpt, tmp_path, **kw):
    return WeightStore(ckpt, tmp_path / "edited", **kw)


def test_load_prefers_the_edit(ckpt, tmp_path):
    s = store(ckpt, tmp_path)
    w = s.load("w")
    w[0, 0] += 1
    s.save_edit("w", w)
    assert s.load("w")[0, 0] == w[0, 0]
    assert s.load_original("w")[0, 0] != w[0, 0]
    assert s.edited() == ["w"]
    s.drop_edit("w")
    assert s.edited() == [] and s.digest() is None


def test_digest_tracks_the_edit(ckpt, tmp_path):
    s = store(ckpt, tmp_path)
    w = s.load("w")
    w[0, 0] += 1
    s.save_edit("w", w)
    first = s.digest()
    w[0, 1] += 1
    s.save_edit("w", w)
    assert len(first) == 10 and s.digest() != first


def test_names_with_slashes_become_safe_file_names(ckpt, tmp_path):
    s = store(ckpt, tmp_path)
    s.save_edit("a/b", s.load("a/b") + 1)
    assert s.edited() == ["a/b"]
    assert len(list((tmp_path / "edited").iterdir())) == 1


def test_frozen_is_refused(ckpt, tmp_path):
    with pytest.raises(EditRefused, match="out of scope"):
        store(ckpt, tmp_path, frozen=["frozen.weight"]).save_edit("frozen.weight", np.zeros(4))


@pytest.mark.parametrize("c", [0.5, 2.0, -1.0])
def test_whole_tensor_rescale_is_refused(ckpt, tmp_path, c):
    s = store(ckpt, tmp_path)
    with pytest.raises(EditRefused, match="constant"):
        s.save_edit("w", s.load("w") * c)


def test_scaling_part_of_a_tensor_is_allowed(ckpt, tmp_path):
    s = store(ckpt, tmp_path)
    w = s.load("w")
    w[:, 0] *= 2
    assert "entries differ" in s.save_edit("w", w)


def test_zeroing_a_whole_tensor_is_allowed(ckpt, tmp_path):
    assert "entries differ" in store(ckpt, tmp_path).save_edit("w", np.zeros((8, 4)))


def test_save_reports_edits_lost_to_rounding(ckpt, tmp_path):
    s = store(ckpt, tmp_path)
    assert "0 of 32 entries differ" in s.save_edit("w", s.load("w") * (1 + 2.0**-20))


def test_read_mode_refuses_edits(ckpt, tmp_path):
    s = store(ckpt, tmp_path, mode="read")
    with pytest.raises(EditRefused, match="only reads"):
        s.save_edit("w", s.load("w") + 1)


def test_wrong_shape_unknown_name_and_nan_are_refused(ckpt, tmp_path):
    s = store(ckpt, tmp_path)
    with pytest.raises(EditRefused, match="shape"):
        s.save_edit("w", np.zeros((2, 2)))
    with pytest.raises(KeyError):
        s.load("nope")
    with pytest.raises(ValueError, match="non-finite"):
        s.save_edit("w", np.full((8, 4), np.nan))


def test_locked_store_refuses_writes(ckpt, tmp_path):
    s = store(ckpt, tmp_path)
    s.locked = True
    with pytest.raises(EditsLocked):
        s.save_edit("w", s.load("w") + 1)
    with pytest.raises(EditsLocked):
        s.drop_edit("w")


def test_names_limit_what_is_visible(ckpt, tmp_path):
    s = store(ckpt, tmp_path, names=["w"])
    with pytest.raises(KeyError):
        s.load("frozen.weight")
