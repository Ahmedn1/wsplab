# HotpotQA screen: complete, not qualified

All 192 responses completed, with original fixture/model hashes verified after
inference. No output was truncated, no provider calls were made, and the external
partition was not evaluated. Ten targeted tests for the HotpotQA and subsequent
FinQA tools pass; lint passes.

The model has a real weakness on this random pilot, but the proposed diagnostic
failed the frozen qualification rule. Full-context thinking and BM25 thinking
both achieve 16/32 (50%) official normalized answer EM. Gold supporting paragraphs
raise thinking EM to 20/32 (62.5%), below the required 65%, with a 12.5-point gap
rather than the required 20 points. Nonthinking full/oracle EM is 11/32 versus
17/32, also short of the specified oracle level and gap. Missing answer markers
exceed 10% in every condition; plain final answers were still scored as a whole.

Oracle minus full paired EM is +18.75 points with SE 9.46 in nonthinking and
+12.5 points with SE 5.94 in thinking. These are diagnostic comparisons of inputs,
not weight-edit improvements or evidence of a council advantage. No threshold,
sample, gold answer or model setting was changed to manufacture qualification.

Answer F1 is 65.7% full-context thinking, 74.2% oracle thinking, and 60.3% BM25
thinking. Strict EM understates some valid answers: a manual initial review found
four clear equivalents rejected in the full-context nonthinking condition,
including `35` versus `35 people` and the short versus full name of Nabokov.
Wrong years, wrong comparison directions and wrong requested entity types also
occur, so the weakness is not entirely an extraction artifact. One reference
answers a city question with an entire biographical sentence. These issues are
retained in the [grading review](natural-hotpotqa-grading-review.json), without
replacing the official metrics with a favorable subjective score.

Decision: retain this as a fallback; do not spend on a council based on the
present diagnostic. Next screen: [FinQA numerical reading](../08_finqa/natural-finqa-plan.md),
which tests published financial-report questions and explicitly audits numerical
units and reference inconsistencies. Its direct-answer metric is distinct from
the official program-execution task.

[All six-condition results](natural-hotpotqa-results.md).
