"""The authors' released edits must rebuild bit for bit against the public Qwen3.5-0.8B weights."""

import pytest

from wsplab import patch as P
from wsplab import upstream
from wsplab.io import Checkpoint

REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
RUNS = ["p10c", "p25c", "p50c"]


@pytest.fixture(scope="module")
def qwen():
    hub = pytest.importorskip("huggingface_hub")
    try:
        path = hub.snapshot_download("Qwen/Qwen3.5-0.8B", revision=REVISION, local_files_only=True,
                                     allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"])
    except hub.errors.LocalEntryNotFoundError as e:
        pytest.skip(f"Qwen3.5-0.8B is not in the local cache, run scripts/get_qwen.py: {e}")
    return Checkpoint(path)


def release(run):
    upstream.fetch(f"exp8_math/runs/{run}/edit/manifest.json")
    return upstream.fetch(f"exp8_math/runs/{run}/edit/edit.npz").parent


@pytest.mark.network
@pytest.mark.model
@pytest.mark.parametrize("run", RUNS)
def test_every_edited_hash_matches(qwen, run):
    patch = P.Patch.load(release(run))
    assert patch.version == 1
    rows = P.verify(patch, qwen)
    assert len(rows) == len(patch.tensors) > 0


@pytest.mark.network
@pytest.mark.model
def test_import_and_apply_p50c(qwen, tmp_path):
    P.import_release(release("p50c"), tmp_path / "patch", name="qwen35-0.8b-p50c", base_repo="Qwen/Qwen3.5-0.8B",
                     base_revision=REVISION, license="CC-BY-4.0",
                     attribution="Takehi, Teoh, Zhang, Khattab: Sparks of Weight-Space Programming (2026)")
    patch = P.Patch.load(tmp_path / "patch")
    assert patch.meta["license"] == "CC-BY-4.0"
    out = Checkpoint(P.apply(patch, qwen, tmp_path / "out"))
    assert set(out.names()) == set(qwen.names())
    for record in patch.tensors:
        from wsplab.io import to_bits
        assert P.format.sha(to_bits(out.raw(record["name"])[1])) == record["edited_sha256"]
    untouched = next(n for n in qwen.names() if n not in {t["name"] for t in patch.tensors})
    assert (out.raw(untouched)[1] == qwen.raw(untouched)[1]).all()


@pytest.mark.network
@pytest.mark.model
def test_semantic_layer_reproduces_the_published_p25c_edit(qwen, tmp_path):
    """The paper's 25-pass edit switches off attention head 7 of layer 15. Written with the semantic layer, on the real
    weights, it must give the authors' published hash."""
    from wsplab.semantic import SemanticMap
    from wsplab.store import WeightStore

    release_patch = P.Patch.load(release("p25c"))
    assert len(release_patch.tensors) == 1
    record = release_patch.tensors[0]

    store = WeightStore(qwen, tmp_path / "edited")
    M = SemanticMap.from_store(store)
    assert M.family == "llama" and M.layers[0].attn is None and M.layers[15].attn is not None
    M.layers[15].attn.head[7].out.zero()

    ours = P.from_submission(qwen, P.read_submission(tmp_path / "edited"), tmp_path / "patch")
    assert ours.tensors[0]["name"] == record["name"]
    assert ours.tensors[0]["edited_sha256"] == record["edited_sha256"]
