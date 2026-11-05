"""The language model path with a fake runner: no GPU, no vLLM. The real runner is tested in tests/gpu/."""

import json

import numpy as np
import pytest

import wsplab
from wsplab import patch as P
from wsplab.arch_text import default_frozen, describe, text_tensors
from wsplab.editors import ScriptedEditor
from wsplab.eval import MathGrader
from wsplab.eval.lm_suite import LMEvaluation, LMSuite, custom_suite
from wsplab.io import Checkpoint, bf16_bits, save
from wsplab.kernel import Limits
from wsplab.kernel.client import KernelError, sandbox_python
from wsplab.passes import ForwardLimits
from wsplab.runners.fake import FakeLM

QWEN = {"text_config": {"num_hidden_layers": 4, "hidden_size": 1024, "num_attention_heads": 8, "num_key_value_heads": 2,
                        "head_dim": 256, "intermediate_size": 3584, "hidden_act": "silu", "vocab_size": 248320,
                        "tie_word_embeddings": True,
                        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]}}


def test_architecture_text_is_derived_from_the_config():
    text = describe("Qwen3.5-0.8B", QWEN)
    for needle in ("4 layers of width 1,024", "layers 3 are full-attention", "8 query heads and 2 key/value heads of size 256",
                   "linear attention layers", "SwiGLU MLP of width 3,584", "vocabulary 248,320", "tied"):
        assert needle in text


def test_architecture_text_for_a_uniform_model():
    c = {"num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 2, "intermediate_size": 16, "hidden_act": "silu"}
    assert "every layer has attention with 2 query heads and 2 key/value heads of size 4" in describe("x", c)


def test_frozen_defaults_follow_the_papers_scope():
    names = ["model.language_model.embed_tokens.weight", "lm_head.weight", "model.language_model.norm.weight",
             "model.language_model.layers.0.input_layernorm.weight", "model.language_model.layers.0.mlp.down_proj.weight",
             "model.norm.weight"]
    assert set(default_frozen(names)) == {names[0], names[1], names[2], names[5]}


def test_text_part_of_a_multimodal_checkpoint():
    assert text_tensors(["model.visual.x", "mtp.fc", "model.language_model.a"]) == ["model.language_model.a"]


def lm_folder(tmp_path):
    folder = tmp_path / "lm"
    folder.mkdir()
    rng = np.random.default_rng(0)
    t = {"model.embed_tokens.weight": ("BF16", bf16_bits(rng.standard_normal((8, 4)).astype(np.float32))),
         "model.norm.weight": ("BF16", bf16_bits(np.ones(4, np.float32))),
         "model.layers.0.mlp.down_proj.weight": ("BF16", bf16_bits(rng.standard_normal((4, 6)).astype(np.float32)))}
    save(folder / "model.safetensors", t)
    (folder / "config.json").write_text(json.dumps({"num_hidden_layers": 1, "hidden_size": 4, "num_attention_heads": 2,
                                                    "intermediate_size": 6, "vocab_size": 8, "tie_word_embeddings": True}))
    return folder


def suite():
    practice = [{"id": f"p{i}", "problem": f"p {i}", "answer": "1", "source": "math"} for i in range(6)]
    evaluation = [{"id": f"e{i}", "problem": f"e {i}", "answer": "1", "source": "bench"} for i in range(20)]
    return LMSuite(practice, evaluation, MathGrader(), {"temperature": 0.6, "max_tokens": 16}, benchmark="the bench", samples=2)


def test_overlap_between_practice_and_evaluation_is_refused():
    row = {"id": "a", "problem": "same", "answer": "1"}
    with pytest.raises(ValueError, match="both the practice"):
        LMSuite([row], [{**row, "id": "b"}], MathGrader(), {"max_tokens": 4})


def test_custom_suite_reads_jsonl(tmp_path):
    (tmp_path / "p.jsonl").write_text('{"id": "1", "problem": "a", "answer": "x"}\n')
    (tmp_path / "e.jsonl").write_text('{"id": "2", "problem": "b", "answer": "y"}\n')
    s = custom_suite(tmp_path / "p.jsonl", tmp_path / "e.jsonl", grader="exact")
    assert len(s.practice) == 1 and s.grader.grade("x", {"answer": "x"}).correct


def test_lm_evaluation_gives_per_item_samples(tmp_path):
    runner = FakeLM(correct=lambda weights, p: p.id == "e0")
    scores = LMEvaluation(runner, suite()).evaluate(tmp_path)
    assert scores.per_item["e0"] == [1.0, 1.0] and scores.per_item["e1"] == [0.0, 0.0]
    assert scores.mean == 0.05 and scores.extra["cut_share"] == 0.95


def fixed_once_edited(weights, prompt):
    return "checkpoints" in str(weights) or "result" in str(weights)


def block(kind, body, why="Testing a hypothesis about a looping direction."):
    return f"{why}\n```{kind}\n{body}\n```"


def test_full_language_model_lifecycle_with_a_fake_runner(tmp_path):
    try:
        sandbox_python()
    except KernelError:
        pytest.skip("no sandbox environment")
    fix = ("w = load('model.layers.0.mlp.down_proj.weight'); w[:, 0] = 0; "
           "print(save_edit('model.layers.0.mlp.down_proj.weight', w))")
    request = json.dumps({"type": "generate", "model": "both", "problems": ["p0", "p1"], "n": 1})
    replies = [block("python", "print(len(NAMES))"), block("python", fix), block("forward", request), "I removed a neuron."]
    runner = FakeLM(correct=fixed_once_edited)
    r = wsplab.run_lm(str(lm_folder(tmp_path)), eval=suite(), editor=ScriptedEditor(replies), runner=runner, budget=1,
                      folder=tmp_path / "run", note="This model really sucks at math", limits=Limits(seconds=60, memory_gb=4),
                      forward_limits=ForwardLimits(n_layers=1, full_attention=(0,), gen_max=8, max_tokens=16), control=False)
    assert r.baseline == 0.0 and r.final == 1.0 and r.verdict == "confirmed"
    prompt = (r.folder / "session/turns/t001/prompt.txt").read_text()
    assert prompt.startswith("TASK. You are given the weights of lm (1 layers of width 4")
    assert "This model really sucks at math" in prompt and "the bench" in prompt
    assert "EVALUATION." in prompt and "/practice/practice.jsonl" in prompt
    patch = P.Patch.load(r.patch)
    assert patch.meta["kind"] == "lm" and patch.meta["evidence"]["verdict"] == "confirmed"
    assert [t["name"] for t in patch.tensors] == ["model.layers.0.mlp.down_proj.weight"]
    assert (r.folder / "work" / "eval_prompt_format.txt").read_text() == "<user><the problem></user>"
    summary = json.loads((r.folder / "session" / "tool_calls.jsonl").read_text().splitlines()[-1])
    assert json.loads(summary["result"])["paired"]["accuracy_difference"] == 1.0


def test_one_run_function_covers_language_models(tmp_path):
    try:
        sandbox_python()
    except KernelError:
        pytest.skip("no sandbox environment")
    r = wsplab.run(str(lm_folder(tmp_path)), eval=suite(), editor=ScriptedEditor(["Nothing to do."]), runner=FakeLM(), budget=0,
                   folder=tmp_path / "run", limits=Limits(seconds=60, memory_gb=4))
    assert r.verdict == "no_fix_found"


def test_default_runner_is_built_when_none_is_given(tmp_path, monkeypatch):
    """The everyday call passes no runner, and must reach the real one."""
    try:
        sandbox_python()
    except KernelError:
        pytest.skip("no sandbox environment")
    import wsplab.runners as runners

    made = {}

    class Recording(FakeLM):
        def __init__(self, tokenizer, **options):
            super().__init__()
            made.update(options, tokenizer=str(tokenizer))

    monkeypatch.setattr(runners, "LM", Recording)
    r = wsplab.run(str(lm_folder(tmp_path)), eval=suite(), editor=ScriptedEditor(["Nothing to do."]), budget=0, thinking=True,
                   generator="hf", device="cpu", folder=tmp_path / "run", limits=Limits(seconds=60, memory_gb=4))
    assert r.verdict == "no_fix_found"
    assert made["thinking"] is True and made["generator"] == "hf" and made["tokenizer"].endswith("lm")


def test_language_model_options_need_an_eval(tmp_path):
    with pytest.raises(TypeError, match="language model options"):
        wsplab.run("x", description="d", editor=ScriptedEditor([]), thinking=True)


def test_the_cache_sits_outside_the_editors_reach_and_the_evaluation_ignores_it(tmp_path):
    try:
        sandbox_python()
    except KernelError:
        pytest.skip("no sandbox environment")

    class Spy(FakeLM):
        def __init__(self):
            super().__init__(correct=fixed_once_edited)
            self.items = []

        def generate(self, weights, prompts, n, sampling):
            self.items += [p.id for p in prompts]
            return super().generate(weights, prompts, n, sampling)

    fix = "w = load('model.layers.0.mlp.down_proj.weight'); w[:, 0] = 0; save_edit('model.layers.0.mlp.down_proj.weight', w)"
    request = {"type": "generate", "model": "both", "problems": ["p0"], "n": 1}
    check = "import os\nprint([f for f in os.listdir('/work')], os.path.exists('/cache'))"
    replies = [block("python", fix), block("forward", json.dumps(request)),
               block("forward", json.dumps({**request, "problems": ["p0", "p1"]})), block("python", check), "Done."]
    spy = Spy()
    r = wsplab.run_lm(str(lm_folder(tmp_path)), eval=suite(), editor=ScriptedEditor(replies), runner=spy, budget=2,
                      folder=tmp_path / "run", limits=Limits(seconds=60, memory_gb=4), control=False,
                      forward_limits=ForwardLimits(n_layers=1, full_attention=(0,), gen_max=8, max_tokens=16))
    assert (r.folder / "cache" / "generations.jsonl").exists()
    calls = (r.folder / "session" / "tool_calls.jsonl").read_text()
    assert "generations.jsonl" not in calls and "cache" not in json.loads(calls.splitlines()[-1])["out"]
    assert spy.items.count("e0") == 2  # the evaluation always generates: once for the baseline, once for the result
    assert spy.items.count("p0") == 2  # both models once, then reused
    assert spy.items.count("p1") == 2


def test_the_cache_can_be_switched_off(tmp_path):
    try:
        sandbox_python()
    except KernelError:
        pytest.skip("no sandbox environment")
    request = json.dumps({"type": "generate", "model": "original", "problems": ["p0"]})
    r = wsplab.run_lm(str(lm_folder(tmp_path)), eval=suite(), runner=FakeLM(), budget=1, cache_answers=False, folder=tmp_path / "run",
                      editor=ScriptedEditor([block("forward", request), "Done."]), limits=Limits(seconds=60, memory_gb=4),
                      forward_limits=ForwardLimits(n_layers=1, full_attention=(0,), gen_max=8, max_tokens=16))
    assert not (r.folder / "cache").exists()


def test_final_norm_and_embedding_are_frozen_by_default(tmp_path):
    try:
        sandbox_python()
    except KernelError:
        pytest.skip("no sandbox environment")
    bad = "print(save_edit('model.norm.weight', load('model.norm.weight') + 1))"
    r = wsplab.run_lm(str(lm_folder(tmp_path)), eval=suite(), editor=ScriptedEditor([block("python", bad), "Done."]),
                      runner=FakeLM(), budget=0, folder=tmp_path / "run", limits=Limits(seconds=60, memory_gb=4))
    assert "out of scope" in (r.folder / "session" / "tool_calls.jsonl").read_text()
    assert r.verdict == "no_fix_found"


def test_checkpoints_on_disk_stay_bounded(tmp_path):
    from wsplab.pipeline import Job, edited_builder
    from wsplab.store import WeightStore

    folder = lm_folder(tmp_path)
    store = WeightStore(Checkpoint(folder), tmp_path / "edited", refuse_rescale=False)

    class J:
        pass

    job = J()
    job.folder = tmp_path / "run"
    build = edited_builder(job, folder, store, keep=2)
    name = "model.layers.0.mlp.down_proj.weight"
    for i in range(4):
        w = store.load_original(name)
        w[:, i] = 7.0
        store.save_edit(name, w)
        assert Checkpoint(build()).float(name)[0, i] == 7.0
    assert len(list((job.folder / "checkpoints").iterdir())) <= 2
    del Job
