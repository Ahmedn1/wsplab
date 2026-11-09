# Fidelity to the paper

wsplab is built from the authors' release, pinned at
`RikiyaT/lm_understand-logs@4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`, and new code. It contains no code from other
weight-space programming frameworks. `provenance/PROVENANCE.md` lists what derives from what.

## What was released, and what was not

The authors released their logs, prompts, edit tools, evaluation code and networks. They did not release their agent driver,
persistent kernel or forward-pass server, so wsplab's versions are **reconstructions** from the prompts and logs. Patches say so
with `"harness": "reconstructed"`.

## What has been checked

The tests in `tests/conformance` check, against the authors' files:

| Check | Result |
|---|---|
| The released p10c, p25c and p50c edits rebuild against the public weights | Every published hash matches |
| The base template renders the authors' v5 prompt | Byte for byte |
| Replaying runs p50c and g50b through the driver | All 275 prompts match byte for byte, and final counters match |
| Logged forward requests | Same accept or reject, except the two that hit a bug the authors later fixed |
| Published scores recompute from their raw parts | Exact |
| `M.layers[15].attn.head[7].out.zero()` on the real weights | Reproduces the published p25c hash |
| Released replies for experiments 4 and 5 | Rebuild the released networks |
| The authors' last logged pass of run p50c, repeated with their released edit (64 problems, thinking on, 32,768 tokens) | +48.4 +/- 6.3 points, against +39.1 +/- 6.2 logged, and +39 to +53 over their last four passes |

## What is inferred or differs

- **Seeds.** The prompts say sampling seeds are fixed by the problem, without a formula, so wsplab uses its own.
- **The rescale check.** The prompts say multiplying a whole tensor by a constant is refused, without saying how close counts.
- **Handoff text and rules** come from logs, but rarely triggered paths did not appear there. The nudge, the block-count message
  and the truncation message are new.
- **The harness changed during the authors' runs.** The replay reproduces those changes with hooks, and wsplab's defaults use the
  final behaviour.
- **Limits are true to your machine.** The default fidelity states real limits rather than the authors' 96 GB. `exact` mode
  reproduces their text.
- **Repeated answers are reused.** A pass that asks the same weights the same question reads the earlier answer instead of
  generating it again, and its summary says how many items were reused. Seeds are fixed by the item, so the answer would nearly
  match anyway, and in the authors' setup repeats were regenerated. Pass `cache_answers=False` to regenerate. The final
  evaluation never uses this.
- **Kernel persistence.** Experiment 3 ran each script in a fresh process. wsplab keeps variables, which only makes it easier.

## The paper's experiments

`experiments/paper` expresses all eight as configuration.

| Experiment | Status |
|---|---|
| 1, 2, 3 | Everything needed is released |
| 4 | Reply reading is checked. Scoring needs Kinetix and an evaluator of your own |
| 5 | Reply reading is checked. Scoring needs MinAtar, which you install yourself because it is GPL-3.0 |
| 6 | The authors' opponents were not released, so only `random` works and scores will not match the paper |
| 7 | The fine-tuned checkpoints must be rebuilt, and will not match the published hashes |
| 8 | Works. A full evaluation needs a large GPU |

## Known gaps

- `sandbox="none"` runs with no isolation, and prompts still say `/work` and `/model`, which are then not the real paths.
- There is no Docker fallback yet, so editing sessions need Linux and bubblewrap. Patch tools work anywhere.
- Patches carry digests, not signatures, and there is no `verify --rerun`: re-measure a patch with `wsplab eval --patch`.
- Quantized checkpoints are not supported.

## What this does not show

Matching the authors' harness does not mean you will get their results. Their headline used `gpt-6-astra` at maximum effort, and
one independent rerun with the same editor scored far lower. The gain was loop repair on a model that looped on 93% of answers,
and a model that did not loop barely moved.
