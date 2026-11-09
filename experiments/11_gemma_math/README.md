# Gemma 3 270M math

The paper's small-gain case (Astra: 6.8% → 7.0–7.7% on MATH-500). Same model revision, v5 prompt, sampling and
lenient grading.

- Trial: `python code/gemma_trial.py --out ~/.wsplab/gemma/trial1 --openai-cap 0.75 --anthropic-cap 2.5 --gemini-cap 2.0`
  (both arms in one process, sharing one GPU runner)
- Audit screen, then evaluation: `python code/gemma_report.py <dir>` and `... --evaluate` (MATH-500 with 8 samples,
  GSM8K test, WikiText log-likelihood). Not run here: both arms kept the original model.

Report: section 2 of [../10_seaquest/cheap-council-and-gemma-results.md](../10_seaquest/cheap-council-and-gemma-results.md);
numbers in [results/](results).
