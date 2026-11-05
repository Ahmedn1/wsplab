"""The real language model runner on the GPU, with Qwen3.5-0.8B. Run with the GPU environment: pytest tests/gpu."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("vllm")
torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")]

from wsplab.runners import LM  # noqa: E402
from wsplab.runners.base import Prompt  # noqa: E402

REV = "2fc06364715b967f1860aea9cf38778875588b17"
QWEN = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / f"hub/models--Qwen--Qwen3.5-0.8B/snapshots/{REV}"
SHORT = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 64}


@pytest.fixture(scope="module")
def lm():
    if not QWEN.exists():
        pytest.skip("Qwen3.5-0.8B is not downloaded")
    return LM(QWEN, thinking=False, gpu_memory=0.7, max_model_len=4096, batch_size=2)


def test_vllm_generates_and_finishes(lm):
    prompts = [Prompt(str(i), lm.render(f"What is {i} + {i}? Reply with only the number."), i) for i in range(1, 5)]
    out = lm.generate(QWEN, prompts, 2, SHORT)
    assert len(out) == 4 and all(len(row) == 2 for row in out)
    assert all(g.finish_reason in ("stop", "length") and g.token_ids for row in out for g in row)
    assert any(str(2 * (i + 1)) in row[0].text for i, row in enumerate(out))


def test_same_seed_gives_the_same_answer_for_an_item(lm):
    p = [Prompt("a", lm.render("Name a colour."), 123)]
    assert lm.generate(QWEN, p, 1, SHORT)[0][0].text == lm.generate(QWEN, p, 1, SHORT)[0][0].text


def test_hf_and_vllm_agree_on_probabilities(lm, tmp_path):
    """A forward pass and the evaluation must describe the same model."""
    import subprocess
    import sys

    text = lm.render("What is the capital of France?") + "The capital of France is Paris."
    hf = lm.read(QWEN, [text], {"token_logprobs": True})["token_logprobs"][0]
    script = tmp_path / "prompt_logprobs.py"
    script.write_text(f"""
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained({str(QWEN)!r})
    ids = tok({text!r}, add_special_tokens=False).input_ids
    llm = LLM(model={str(QWEN)!r}, dtype="bfloat16", gpu_memory_utilization=0.7, max_model_len=4096, seed=0,
              limit_mm_per_prompt={{"image": 0, "video": 0}})
    out = llm.generate([{{"prompt_token_ids": ids}}], SamplingParams(max_tokens=1, prompt_logprobs=0))[0]
    json.dump([float(d[i].logprob) for d, i in zip(out.prompt_logprobs[1:], ids[1:])], open({str(tmp_path / 'lp.json')!r}, "w"))
""")
    subprocess.run([sys.executable, str(script)], check=True, capture_output=True,
                   env={**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    vllm_lp = np.array(json.load(open(tmp_path / "lp.json")))
    assert vllm_lp.shape == hf.shape
    assert np.corrcoef(vllm_lp, hf)[0, 1] > 0.98
    assert np.abs(vllm_lp - hf).mean() < 0.25


def test_activation_capture_is_consistent_with_a_read(lm):
    """Mean activations from token ids must match a teacher-forced read of the same tokens."""
    prompt = Prompt("a", lm.render("Say hello."), 1)
    from wsplab.runners.base import Generation

    answer_ids = lm.tok("Hello there, friend.", add_special_tokens=False).input_ids
    g = Generation("Hello there, friend.", "stop", answer_ids)
    acts = lm.activations(QWEN, [prompt], [g], [0, 15, 24], True, 8192)
    assert acts["hidden_last_prompt"].shape == (1, 3, 1024) and acts["mlp_mean_answer"].shape == (1, 2, 3584)
    text = lm.tok.decode(lm.tok(prompt.text, add_special_tokens=False).input_ids + answer_ids)
    full = lm.read(QWEN, [text], {"hidden_states": [15]})
    n = len(lm.tok(prompt.text, add_special_tokens=False).input_ids)
    assert np.allclose(acts["hidden_last_prompt"][0, 1], full["hidden_15"][0, n - 1], atol=0.1)
    assert np.allclose(acts["hidden_mean_answer"][0, 1], full["hidden_15"][0, n:].mean(0), atol=0.1)


def test_reading_activations_leaves_the_gpu_free_for_vllm(lm):
    """The model must really be gone afterwards, or vLLM starts short of memory: found by a real run, never by a fake one."""
    lm.read(QWEN, ["hello there"], {"token_logprobs": True, "hidden_states": [1]})
    assert torch.cuda.memory_reserved() < 64 * 2**20


def test_hf_generation_applies_the_presence_penalty(lm):
    hf = LM(QWEN, thinking=False, generator="hf")
    p = [Prompt("a", hf.render("Repeat the word cat twenty times."), 5)]
    plain = hf.generate(QWEN, p, 1, {"temperature": 0.0, "max_tokens": 40})[0][0]
    penalised = hf.generate(QWEN, p, 1, {"temperature": 0.0, "max_tokens": 40, "presence_penalty": 5.0})[0][0]
    assert len(set(penalised.token_ids)) > len(set(plain.token_ids))
