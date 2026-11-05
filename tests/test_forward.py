import json

import numpy as np
import pytest

from wsplab.eval import ChoiceGrader, ExactGrader, MathGrader, RegexGrader, extract, get_grader, is_correct, paired
from wsplab.passes import ForwardLimits, LMForward, PassFailed, PassInvalid
from wsplab.runners.fake import FakeLM

PRACTICE = {f"p{i}": {"id": f"p{i}", "source": "math", "problem": f"q{i}", "answer": "1"} for i in range(8)}
LIMITS = ForwardLimits(n_layers=4, full_attention=(3,), gen_max=32, max_tokens=64, tf_max_texts=4)


def tool(tmp_path, runner=None):
    digest = {"value": None}
    t = LMForward(runner or FakeLM(), practice=PRACTICE, grader=MathGrader(), limits=LIMITS,
                  sampling={"temperature": 0.6, "max_tokens": 64}, work=tmp_path, base=tmp_path / "base",
                  edited=lambda: tmp_path / "edited_model", edit_digest=lambda: digest["value"],
                  user_message=lambda row: row["problem"])
    t.digest = digest
    return t


def ask(t, k=1, **request):
    return t.run(json.dumps(request), k)


def test_generation_summary_has_the_logged_fields(tmp_path):
    r = ask(tool(tmp_path), type="generate", model="original", problems=["p0", "p1"], n=2)
    s = r.summary
    assert r.where == "/work/gen_1.jsonl" and s["generations"] == 4
    assert s["failures"] == {"wrong_answer": 0, "no_answer_finished": 0, "cut_at_max_tokens": 4}
    assert s["cut_share"] == 1.0 and "s" in s
    rows = [json.loads(line) for line in (tmp_path / "gen_1.jsonl").read_text().splitlines()]
    assert {"id", "source", "sample", "text", "pred", "correct", "finish", "tokens"} <= set(rows[0])


def test_both_reports_a_paired_comparison(tmp_path):
    fixed = FakeLM(correct=lambda weights, p: "edited" in str(weights))
    r = ask(tool(tmp_path, fixed), type="generate", model="both", problems=["p0", "p1", "p2"], n=2)
    assert r.where == "/work/gen_1_original.jsonl and /work/gen_1_edited.jsonl"
    assert r.summary["paired"] == {"items": 3, "accuracy_difference": 1.0, "standard_error": 0.0,
                                   "samples_wrong_to_right": 6, "samples_right_to_wrong": 0}
    assert (tmp_path / "gen_1_edited.jsonl").exists()


def test_activations_are_saved(tmp_path):
    r = ask(tool(tmp_path), type="generate", model="original", problems=["p0"], n=1,
            activations={"layers": [0, 4], "mlp": True})
    assert r.summary["activations"]["files"]["hidden_last_prompt.npy"] == [1, 2, 8]
    assert np.load(tmp_path / "gen_1_acts" / "mlp_mean_answer.npy").shape == (1, 2, 16)
    assert json.loads((tmp_path / "gen_1_acts" / "index.json").read_text())["layers"] == [0, 4]


def test_teacher_forced_writes_arrays_and_meta(tmp_path):
    r = ask(tool(tmp_path), type="teacher_forced", model="original", texts=["a b", "c"],
            **{"return": {"token_logprobs": True, "hidden_states": [1], "mlp_act": [1]}})
    assert r.where == "/work/fwd_1/" and r.summary["files"]["hidden_1"] == [2, 3, 8]
    assert json.loads((tmp_path / "fwd_1" / "meta.json").read_text()) == {"lengths": [3, 2]}
    assert np.load(tmp_path / "fwd_1" / "hidden_1.npy").shape == (2, 3, 8)


@pytest.mark.parametrize("request_", [
    {"type": "other"},
    {"type": "generate", "model": "nope", "problems": ["p0"]},
    {"type": "generate", "model": "original", "problems": ["p0"], "n": 0},
    {"type": "generate", "model": "original", "problems": ["p0"], "n": 33},
    {"type": "generate", "model": "both", "problems": ["p0"], "n": 17},
    {"type": "generate", "model": "original", "problems": ["p0"], "max_tokens": 65},
    {"type": "generate", "model": "original", "problems": ["p0"], "activations": {"layers": [5]}},
    {"type": "generate", "model": "original"},
    {"type": "teacher_forced", "model": "both", "texts": ["a"]},
    {"type": "teacher_forced", "model": "original"},
    {"type": "teacher_forced", "model": "original", "texts": ["a"] * 5},
    {"type": "teacher_forced", "model": "original", "texts": ["a"], "return": {"topk": 51}},
    {"type": "teacher_forced", "model": "original", "texts": ["a"], "return": {"attn_layers": [1]}},
    {"type": "teacher_forced", "model": "original", "texts": ["a"] * 3, "return": {"attn_layers": [3]}},
])
def test_invalid_requests_cost_nothing(tmp_path, request_):
    with pytest.raises(PassInvalid):
        tool(tmp_path).run(json.dumps(request_), 1)


def test_bad_json_is_invalid(tmp_path):
    with pytest.raises(PassInvalid, match="JSONDecodeError"):
        tool(tmp_path).run("{not json", 1)


def test_unknown_ids_fail_as_logged(tmp_path):
    with pytest.raises(PassFailed, match=r"ValueError: unknown practice ids: \['gsm8k_train3513'\]"):
        ask(tool(tmp_path), type="generate", model="original", problems=["gsm8k_train3513"])


def test_backend_failure_does_not_cost_a_pass(tmp_path):
    runner = FakeLM()
    runner.fail_with = RuntimeError("engine died")
    with pytest.raises(PassFailed, match="RuntimeError: engine died"):
        ask(tool(tmp_path, runner), type="generate", model="original", problems=["p0"])


def test_identical_request_refused_until_edits_change(tmp_path):
    t = tool(tmp_path)
    req = dict(type="generate", model="edited", problems=["p0"])
    ask(t, 1, **req)
    with pytest.raises(PassInvalid, match="identical"):
        ask(t, 2, **req)
    t.digest["value"] = "abc"
    ask(t, 3, **req)


def test_own_prompts_are_generated_but_not_graded(tmp_path):
    r = ask(tool(tmp_path), type="generate", model="both", prompts=["hello"], n=1)
    assert r.summary["paired"] is None


def test_math_extraction():
    assert extract("so \\boxed{42}") == "42"
    assert extract("<think>\\boxed{1}</think> then \\boxed{2}") == "2"
    assert extract("\\boxed{unclosed") is None
    assert extract("no box") is None
    assert extract("\\boxed 5$ more") == "5"


def test_math_equivalence():
    assert is_correct("\\frac{1}{2}", "0.5")
    assert not is_correct("3", "4")
    assert not is_correct(None, "4")


def test_lenient_math_grading():
    row = {"answer": "72"}
    assert not MathGrader().grade("Final Answer: 72", row).correct
    assert MathGrader(lenient=True).grade("Final Answer: 72", row).correct
    assert MathGrader(lenient=True).grade("total is 10 and then 72", row).correct


def test_other_graders():
    assert ExactGrader().grade(" Paris ", {"answer": "paris"}).correct
    assert RegexGrader(r"ANSWER=(\d+)").grade("ANSWER=7", {"answer": "7"}).correct
    assert ChoiceGrader().grade("The answer is (B).", {"answer": "B"}).correct
    assert not ChoiceGrader().grade("The answer is (C).", {"answer": "B"}).correct
    assert get_grader(lambda text, row: text == "x").grade("x", {}).correct
    with pytest.raises(ValueError):
        get_grader("nope")


def test_paired_standard_error_matches_a_logged_value():
    """exp8_math/runs/p50c gen_5: 32 items, 2 wrong to right: +0.0625 with standard error 0.0435."""
    a = [[0.0]] * 32
    b = [[1.0]] * 2 + [[0.0]] * 30
    p = paired(a, b)
    assert round(p.difference, 4) == 0.0625 and round(p.standard_error, 4) == 0.0435
    assert p.wrong_to_right == 2 and p.right_to_wrong == 0
