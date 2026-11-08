# Peer council: a positive case on routed digit repair

On construction seed 22, the two-peer search selected a repair scoring **88.82%**
on 340 held-out queries. The same models working independently scored **83.53%**
(Claude) and **78.82%** (terra). This is a live model comparison, not the scripted
oracle smoke test. The damaged model scored 50.00%.

| Session | Selection score | Held-out score | Python runs | Forward passes | Stop |
|---|---:|---:|---:|---:|---|
| Peer terra, selected | 91.43% | **88.82%** | 12 | 6 | Finished |
| Peer Claude | 91.43% | 87.94% | 22 | 6 | Finished |
| Independent terra | 85.71% | 78.82% | 13 | 6 | Finished |
| Independent Claude, selected | 90.00% | **83.53%** | 27 | 6 | Finished |

The peer gain over the selection-chosen independent candidate is **5.29 percentage
points**: 25 queries change from wrong to right and 7 from right to wrong. The
paired query-level standard error is 1.64 points. This measures uncertainty across
these queries conditional on these runs; it does not measure variation across
model runs, construction seeds, or tasks.

## What was held equal

The [plan](peer-confirmation-plan.md) was fixed before calls. Both arms used
`gpt-6.1-terra` and `claude-opus-5-5`, high effort, identical prompts and damaged
weights, the same practice and selection split, and limits of 40 Python runs,
6 forward passes and 60 turns per editor. Both received the same clarification
forbidding counterfactual network evaluation in Python. Each arm had provider
caps of $0.75 OpenAI and $3.50 Anthropic, using the corrected accounting guard.
All four sessions finished normally; none was stopped by a cap or provider error.

Candidates were chosen on 140 selection queries before external evaluation.
The two peers tied on selection score; the existing first-candidate tie rule
selected terra. Claude won the independent arm's selection. The other candidates'
held-out scores above are post-run diagnostics, not a basis for choosing winners.
Thus the primary comparison also beats a portfolio containing one independent
attempt by each of the same two models, not just a hand-picked weaker solo.

## Cost and audit

| Arm | Estimated API cost |
|---|---:|
| Two peers together | $2.20 |
| Both independent attempts together | $1.73 |
| Claude alone | $1.50 |
| terra alone | $0.23 |

The confirmation cost $3.93 in total. Equal spending ceilings did **not** produce
equal actual costs: the peers cost about 27% more than the independent portfolio
and 47% more than Claude alone. No efficiency advantage has been established.
These are usage-based estimates, not provider invoices. There were no API errors
or missing-output-usage estimates in this phase. Prices are recorded in the
plans; the guard conservatively charges all uncached Claude input at the cache
write rate. Pricing references: [OpenAI](https://developers.openai.com/api/docs/pricing)
and [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing).

Manual review covered all four sessions, including matrix operations, and found
no prohibited modified-network replay, training, or access to external evaluation
data. Weight-template correlations and one-shot permutation matching used
practice labels and recorded activations, without computing modified-network
predictions inside Python. Peer terra's turns 11–12 explicitly use the other
peer's spatial-repair analysis, independently inspect the proposed mappings, and
then test through the official forward tool. This is evidence that findings were
used, although it does not isolate sharing as the cause through an ablation.

One qualification: the Claude peer mixed its self-designated practice HOLDOUT
into a development request. Its internal confirmation therefore is not
independent; terra explicitly noticed this. The framework's separate selection
and external evaluation data remained outside all editor kernels. The reported
final comparison uses that external evaluation, not the contaminated internal
check.

The earlier seed-21 pilot's 91.76% peer result is **excluded as clean evidence**:
a contributing Gemini peer simulated a modified network tail in Python, and
some sessions ended at guards. Its accounting also has missing Gemini output
usage. See the retained [pilot report](peer-pilot-results.md). The fresh-seed
confirmation excludes Gemini and applies the same clarified rules to both arms.

## Scope and artifacts

This establishes one positive case on a deliberately modular repair task. It
does not establish a repeatable advantage, an advantage on interacting faults,
or a win over spending the peer budget on repeated attempts by the strongest
solo model. Seed 22 was fresh for model testing, but was among the seeds used
for CPU fixture calibration. The task was designed after earlier council
failures; it is not an unbiased benchmark sample.

The [machine-readable results](peer-confirmation-results.json) include per-branch
scores, stopping states, accounting, paired comparisons and audit notes. Raw
plans, ledgers, transcripts and checkpoints remain under
`.wsplab/experiments/peer-confirmation-seed22/{peers,independent}/`.
Both arms' base checkpoint SHA-256 is
`af7ea4bfa3db107c8ee63de63c3f3f110a206e0dfff6d381757f889f18525092`.

Recompute the diagnostics without making model calls:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/.venvs/wsplab/bin/python \
  scripts/live/peer_trial_report.py .wsplab/experiments/peer-confirmation-seed22
```

This regenerates the raw comparison; the manual audit annotations in the
published JSON are a separate review step. More paid runs were not started
after this confirmation. A useful next experiment would fix several new seeds
and compare peers against both independent members and repeated Claude attempts
under the same total spending ceiling, retaining failures as well as successes.
