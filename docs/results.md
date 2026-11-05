# Live results

Measured with real LLM editors, on one laptop with a 16 GB GPU. These are small tests, so read them as first evidence, not as
benchmarks. The scripts are in `scripts/live`.

| Test | Editor | Budget | Verdict | Baseline | Result | Cost |
|---|---|---|---|---|---|---|
| Trained digit classifier, two output rows swapped | `gpt-6-sol` | 3 passes | `confirmed` | 82.1% | 98.2% (the clean score) | $0.27 |
| Trained digit classifier, one output row with 5 flipped signs | A coding agent, from the skill alone | 6 passes | `confirmed` | 86.5% | 95.0% (the clean score) | No API cost |
| Gemma 3 270M, MATH-500 subset | `gpt-6-sol` | 5 passes | `not_significant` | 8.0% | 6.5% | $0.40 |
| Qwen3.5-0.8B, MATH-500 subset | `gpt-6-astra` | 12 passes | `not_significant` | 4.0% | 10.0% | $2.93 |

## The language model runs

The Gemma result agrees with the paper, where no edit moved that model by more than about a point.

The Qwen run is not a replication of the paper's headline. It had 12 passes rather than 50, and answers were capped at 8,192 tokens
rather than 32,768, with 100 evaluation problems and one sample each. Astra removed one direction from the layer 19 MLP, the same
kind of edit as the paper's 10-pass run.

The evidence on that edit disagrees with itself. On the hidden evaluation it gained 6.0 +/- 2.8 points, but a random edit of the same
size came within 5.0 +/- 3.0 points of it, so the gate did not confirm it. On Astra's own 64-problem holdout it lost 1.0 +/- 1.5 points,
and Astra reported that plainly. The likeliest reading is no reliable effect.

Separately, the authors' released 50-pass edit, rerun on their logged final pass, gave +48.4 +/- 6.3 points against their logged
+39.1 +/- 6.2. So wsplab reproduces the measurement of that edit. It has not reproduced the discovery of one.

## The council

See [Councils and search](council.md). No council beat its best member.
