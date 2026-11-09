import json

import pytest
import torch
from mgsm_controls import CONDITIONS, grade_response, load_records, presence_processor
from mgsm_controls_report import read_rows, report


def test_presence_penalty_counts_generated_tokens_once_and_ignores_prompt():
    pytest.importorskip('transformers')
    processor = presence_processor(1.5, 3)
    scores = torch.zeros((1, 8))
    output = processor(torch.tensor([[1, 2, 3, 4, 4, 5]]), scores)
    assert output.tolist() == [[0., 0., 0., 0., -1.5, -1.5, 0., 0.]]
    assert torch.equal(scores, torch.zeros_like(scores))


def test_numbers_in_translation_are_not_scored_as_math_answers():
    result = grade_response(r"There are 15 wins. \boxed{15}", CONDITIONS["translate"], {"answer": "15"})
    assert result["strict_correct"] is None and result["lenient_correct"] is None


def test_unfinished_thinking_cannot_receive_credit_for_an_intermediate_box():
    row = {"answer": "15"}
    result = grade_response(r"Maybe \boxed{15} but let me check", CONDITIONS["thinking"], row)
    assert result["thinking_unfinished"] and not result["strict_correct"]
    result = grade_response(r"Maybe \boxed{15}</think>Actually \boxed{16}", CONDITIONS["thinking"], row)
    assert not result["thinking_unfinished"] and not result["strict_correct"]
    result = grade_response(r"Done.</think>\boxed{১৫}", CONDITIONS["thinking"], row)
    assert result["strict_correct"]


@pytest.mark.parametrize("ids", [["a", "a"], ["a", "outside-pilot"]])
def test_resume_rejects_duplicate_or_foreign_items(tmp_path, ids):
    path = tmp_path / "responses.jsonl"
    path.write_text("".join(json.dumps({"id": i}) + "\n" for i in ids))
    with pytest.raises(ValueError, match="Duplicate or unexpected"):
        load_records(path, ["a", "b"])


def test_paired_uncertainty_treats_translations_as_one_problem_cluster(tmp_path):
    run = tmp_path / "controls-gpu-32"
    run.mkdir()
    (run / "execution.json").write_text(json.dumps({"groups": 32, "max_output_tokens": 32768,
                                                   "order": ["nonthinking", "english_reason"]}))
    (run / "status.json").write_text(json.dumps({"status": "completed"}))
    for condition in ["nonthinking", "english_reason"]:
        rows = [{"group": g, "language": lang, "truncated": False,
                 "strict_correct": lang == "en" or (condition == "english_reason" and g < 16)}
                for g in range(32) for lang in ["en", "bn", "sw"]]
        (run / f"{condition}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = report(tmp_path, tmp_path / "report")
    conditions = result["arms"]["gpu_full_pilot"]["conditions"]
    delta = conditions["english_reason"]["target_change_vs_nonthinking"]
    assert delta["independent_problem_groups"] == 32
    assert delta["difference"] == .5
    assert delta["standard_error_clustered_by_problem"] == pytest.approx(1 / (2 * 31**.5))
    assert conditions["nonthinking"]["difficulty_rule"] == "passes_numeric_rule"
    assert conditions["english_reason"]["difficulty_rule"] == "fails_numeric_rule"


def test_live_report_ignores_incomplete_multibyte_append(tmp_path):
    path = tmp_path / "stream.jsonl"
    path.write_bytes(b'{"id":"complete"}\n{"text":"' + "ব".encode()[:2])
    assert read_rows(path) == [{"id": "complete"}]
