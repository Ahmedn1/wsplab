import json

import numpy as np
import pytest

import wsplab
from wsplab import patch as P
from wsplab.editors import ScriptedEditor
from wsplab.eval.suites import ArrayScorer
from wsplab.io import Checkpoint, save
from wsplab.kernel import Limits
from wsplab.kernel.client import KernelError, sandbox_python
from wsplab.semantic import Ref, SemanticMap, kernel_prelude, prompt_addendum
from wsplab.store import WeightStore

rng = np.random.default_rng(4)
H, FF, HEADS, KV, D = 8, 12, 4, 2, 2


def tensors_llama(prefix="model.layers"):
    t = {}
    for i in range(2):
        p = f"{prefix}.{i}"
        t[f"{p}.mlp.gate_proj.weight"] = rng.standard_normal((FF, H))
        t[f"{p}.mlp.up_proj.weight"] = rng.standard_normal((FF, H))
        t[f"{p}.mlp.down_proj.weight"] = rng.standard_normal((H, FF))
        t[f"{p}.self_attn.q_proj.weight"] = rng.standard_normal((HEADS * D, H))
        t[f"{p}.self_attn.k_proj.weight"] = rng.standard_normal((KV * D, H))
        t[f"{p}.self_attn.v_proj.weight"] = rng.standard_normal((KV * D, H))
        t[f"{p}.self_attn.o_proj.weight"] = rng.standard_normal((H, HEADS * D))
    return {k: ("F32", v.astype(np.float32)) for k, v in t.items()}


def llama(tmp_path, **config):
    save(tmp_path / "m.safetensors", tensors_llama())
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 2, "hidden_size": H, "num_attention_heads": HEADS,
                                                      "num_key_value_heads": KV, "head_dim": D, **config}))
    store = WeightStore(Checkpoint(tmp_path / "m.safetensors"), tmp_path / "edited", refuse_rescale=False)
    return store, SemanticMap.from_store(store)


def test_family_detected(tmp_path):
    assert llama(tmp_path)[1].family == "llama"


def test_neuron_addresses(tmp_path):
    store, M = llama(tmp_path)
    down = store.load("model.layers.1.mlp.down_proj.weight")
    assert np.array_equal(M.layers[1].mlp.neuron[5].out.read(), down[:, 5:6])
    gate = store.load("model.layers.1.mlp.gate_proj.weight")
    assert np.array_equal(M.layers[1].mlp.neuron[5].gate.read(), gate[5:6])


def test_head_addresses_with_grouped_query_attention(tmp_path):
    store, M = llama(tmp_path)
    o = store.load("model.layers.0.self_attn.o_proj.weight")
    k = store.load("model.layers.0.self_attn.k_proj.weight")
    head = M.layers[0].attn.head[3]
    assert np.array_equal(head.out.read(), o[:, 6:8])
    assert np.array_equal(head.k.read(), k[2:4])  # heads 2 and 3 share key/value head 1


def test_verbs_lower_to_save_edit(tmp_path):
    store, M = llama(tmp_path)
    M.layers[0].mlp.neuron[2].out.zero()
    assert store.edited() == ["model.layers.0.mlp.down_proj.weight"]
    w = store.load("model.layers.0.mlp.down_proj.weight")
    assert (w[:, 2] == 0).all() and (w[:, 3] != 0).all()
    M.layers[0].mlp.neuron[3].out.scale(2)
    assert np.allclose(store.load("model.layers.0.mlp.down_proj.weight")[:, 3],
                       2 * store.load_original("model.layers.0.mlp.down_proj.weight")[:, 3])


def test_project_out_matches_the_papers_formula(tmp_path):
    store, M = llama(tmp_path)
    name = "model.layers.0.mlp.down_proj.weight"
    w = store.load_original(name).astype(np.float64)
    d = rng.standard_normal(H)
    M.layers[0].mlp.down.project_out(d, 1.5)
    u = d / np.linalg.norm(d)
    assert np.allclose(store.load(name), w - 1.5 * np.outer(u, u @ w), atol=1e-5)
    M2 = M.layers[0].mlp.neuron[1].out
    M2.project_out(d, 1.0)
    assert abs(float(u @ store.load(name)[:, 1])) < 1e-5


def test_out_of_range_is_a_clear_error(tmp_path):
    _, M = llama(tmp_path)
    with pytest.raises(IndexError, match="layer 9 is out of range: there are 2"):
        M.layers[9]
    with pytest.raises(IndexError, match="neuron"):
        M.layers[0].mlp.neuron[FF]


def test_hybrid_layers_have_no_attention(tmp_path):
    t = tensors_llama()
    for k in [k for k in t if k.startswith("model.layers.0.self_attn")]:
        del t[k]
    t["model.layers.0.linear_attn.out_proj.weight"] = ("F32", np.ones((H, H), np.float32))
    save(tmp_path / "m.safetensors", t)
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {
        "num_hidden_layers": 2, "hidden_size": H, "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": D,
        "layer_types": ["linear_attention", "full_attention"]}}))
    store = WeightStore(Checkpoint(tmp_path / "m.safetensors"), tmp_path / "e")
    M = SemanticMap.from_store(store)
    assert M.layers[0].attn is None and M.layers[0].kind == "linear_attention"
    assert M.layers[1].attn is not None and M.layers[0].mlp is not None


def gpt2(tmp_path):
    h, ff, heads = 6, 8, 3
    t = {"transformer.h.0.attn.c_attn.weight": rng.standard_normal((h, 3 * h)),
         "transformer.h.0.attn.c_proj.weight": rng.standard_normal((h, h)),
         "transformer.h.0.mlp.c_fc.weight": rng.standard_normal((h, ff)),
         "transformer.h.0.mlp.c_proj.weight": rng.standard_normal((ff, h))}
    save(tmp_path / "m.safetensors", {k: ("F32", v.astype(np.float32)) for k, v in t.items()})
    (tmp_path / "config.json").write_text(json.dumps({"n_layer": 1, "n_embd": h, "n_head": heads}))
    store = WeightStore(Checkpoint(tmp_path / "m.safetensors"), tmp_path / "e", refuse_rescale=False)
    return store, SemanticMap.from_store(store)


def test_gpt2_is_transposed_and_fused(tmp_path):
    store, M = gpt2(tmp_path)
    assert M.family == "gpt2"
    proj = store.load("transformer.h.0.mlp.c_proj.weight")
    assert np.array_equal(M.layers[0].mlp.neuron[2].out.read(), proj[2:3])
    fused = store.load("transformer.h.0.attn.c_attn.weight")
    head = M.layers[0].attn.head[1]
    assert np.array_equal(head.q.read(), fused[:, 2:4])
    assert np.array_equal(head.k.read(), fused[:, 8:10])
    assert np.array_equal(head.v.read(), fused[:, 14:16])
    d = rng.standard_normal(6)
    M.layers[0].mlp.neuron[2].out.project_out(d)
    u = d / np.linalg.norm(d)
    assert abs(float(store.load("transformer.h.0.mlp.c_proj.weight")[2] @ u)) < 1e-5


def test_generic_walker_handles_linear_conv_and_bias(tmp_path):
    t = {"net.fc.weight": rng.standard_normal((4, 3)), "net.fc.bias": rng.standard_normal(4),
         "conv.weight": rng.standard_normal((2, 1, 3, 3))}
    save(tmp_path / "m.safetensors", {k: ("F32", v.astype(np.float32)) for k, v in t.items()})
    store = WeightStore(Checkpoint(tmp_path / "m.safetensors"), tmp_path / "e", refuse_rescale=False)
    M = SemanticMap.from_store(store)
    assert M.family == "generic" and M.layers is None and sorted(M.module) == ["conv", "net.fc"]
    M.module["net.fc"].unit[1].weights.zero()
    M.module["net.fc"].unit[1].bias.write(5)
    assert (store.load("net.fc.weight")[1] == 0).all() and store.load("net.fc.bias")[1] == 5
    assert M.module["conv"].unit[1].weights.read().shape == (1, 1, 3, 3)
    assert M.module["net.fc"].column[2].read().shape == (4, 1)
    with pytest.raises(KeyError, match="no module"):
        M.module["nope"]


def test_raw_tensor_access_always_works(tmp_path):
    store, M = llama(tmp_path)
    M.tensor("model.layers.0.mlp.up_proj.weight").add(1.0)
    assert store.edited() == ["model.layers.0.mlp.up_proj.weight"]


def test_the_semantic_layer_computes_no_statistics():
    """It must stay an address book: the paper's editor measures things with its own code."""
    public = {n for n in dir(Ref) if not n.startswith("_")} - set(Ref.__dataclass_fields__)
    assert public == {"read", "write", "scale", "zero", "add", "project_out"}
    public = {n for n in dir(SemanticMap) if not n.startswith("_")}
    assert public == {"from_store", "tensor"}


def test_frozen_tensors_stay_frozen_through_the_semantic_layer(tmp_path):
    save(tmp_path / "m.safetensors", tensors_llama())
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 2, "hidden_size": H, "num_attention_heads": HEADS,
                                                      "num_key_value_heads": KV, "head_dim": D}))
    frozen = "model.layers.0.mlp.down_proj.weight"
    store = WeightStore(Checkpoint(tmp_path / "m.safetensors"), tmp_path / "e", frozen=[frozen])
    with pytest.raises(ValueError, match="out of scope"):
        SemanticMap.from_store(store).layers[0].mlp.neuron[0].out.zero()


def _sandbox():
    try:
        sandbox_python()
    except KernelError:
        pytest.skip("no sandbox environment")


def test_every_level_gives_the_identical_patch(tmp_path):
    """One edit, zeroing a neuron's output, made at each level, must give the same bytes."""
    _sandbox()
    import torch

    from wsplab.runners.torch import mlp, state_to_checkpoint

    net = mlp([3, 5, 2])
    torch.manual_seed(0)
    state_to_checkpoint(net.state_dict(), tmp_path / "base" / "m.safetensors")
    base = Checkpoint(tmp_path / "base" / "m.safetensors")
    name = "layers.2.weight"
    target = base.float(name)
    target[:, 1] = 0

    level0 = P.from_submission(base, {name: target}, tmp_path / "l0")

    store = WeightStore(base, tmp_path / "e1")
    w = store.load(name)
    w[:, 1] = 0
    store.save_edit(name, w)
    level1 = P.from_submission(base, P.read_submission(tmp_path / "e1"), tmp_path / "l1")

    store = WeightStore(base, tmp_path / "e2")
    SemanticMap.from_store(store).module["layers.2"].column[1].zero()
    level2 = P.from_submission(base, P.read_submission(tmp_path / "e2"), tmp_path / "l2")

    x = rng.standard_normal((40, 3)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    code = "w = load('layers.2.weight'); w[:, 1] = 0; save_edit('layers.2.weight', w)"
    reply = [f"Removing a unit.\n```python\n{code}\n```", "Done."]
    level5 = wsplab.run(tmp_path / "base" / "m.safetensors", build={"mlp": [3, 5, 2]}, description="a net", budget=0,
                        evaluate=ArrayScorer(x, y), editor=ScriptedEditor(reply), folder=tmp_path / "l5", control=False,
                        limits=Limits(seconds=60, memory_gb=4))
    records = [p.tensors for p in (level0, level1, level2, P.Patch.load(level5.patch))]
    assert all(r[0]["edited_sha256"] == records[0][0]["edited_sha256"] for r in records)
    assert all(r[0]["changed_entries"] == records[0][0]["changed_entries"] for r in records)


def test_semantic_layer_inside_the_kernel(tmp_path):
    _sandbox()
    import torch

    from wsplab.runners.torch import mlp

    x = rng.standard_normal((40, 3)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    code = 'print(M.family, list(M.module)); M.module["layers.0"].unit[0].weights.zero()'
    r = wsplab.run(mlp([3, 4, 2]), build={"mlp": [3, 4, 2]}, description="a net", budget=0, semantic=True, no_eval=True,
                   evaluate=ArrayScorer(x, y), editor=ScriptedEditor([f"Using M.\n```python\n{code}\n```", "Done."]),
                   folder=tmp_path / "r", limits=Limits(seconds=60, memory_gb=4))
    calls = (r.folder / "session" / "tool_calls.jsonl").read_text()
    assert "generic ['layers.0', 'layers.2']" in calls
    prompt = (r.folder / "session/turns/t001/prompt.txt").read_text()
    assert "Also defined: M, named addresses" in prompt
    assert P.Patch.load(r.patch).meta["prompt_fidelity"] == "paper+semantic"
    del torch


def test_prelude_and_addendum_exist():
    assert "SemanticMap" in kernel_prelude()
    assert "neuron" in prompt_addendum("llama") and "neuron" not in prompt_addendum("generic")
