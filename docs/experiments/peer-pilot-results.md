# Seed-21 live pilot: promising score, failed protocol audit

These are real API editor runs, not scripted repairs. The peer-selected candidate
scored 91.76% on 340 held-out binary digit queries, compared with 75.88% from the
completed terra solo run. However, **this is not a clean confirmation**: a
contributing peer simulated counterfactual network tails inside the kernel, and
the other solo baselines were capped. The fresh-seed follow-up is specified in
[peer-confirmation-plan.md](peer-confirmation-plan.md).

| Run | Evaluation accuracy | Termination |
|---|---:|---|
| Damaged model | 53.53% | Baseline |
| Peer-selected terra | **91.76%** | Finished, 6 passes, 17 Python runs |
| Terra alone | 75.88% | Finished, 6 passes, 15 Python runs |
| Claude alone | 79.71% | Reservation cap, 2 passes |
| Gemini alone | 53.53% | Context guard, no saved edit |

The candidate selector chose Claude for the independent arm using selection data,
before evaluation. The peer advantage was +12.06 percentage points against that
selected candidate (paired standard error 2.57 points). Against completed terra it
was +15.88 points (SE 2.45), with 66 queries improving and 12 worsening. These are
conditional differences between the produced checkpoints on one task, not
uncertainty estimates across independent model runs.

## Evidence that sharing mattered

Terra's independent run repaired branches 0–3 but left branches 4–7 at chance.
The peer winner's corresponding accuracies were 100%, 94.12%, 100%, and 64.71%.
The winner used a peer's pixel-column fingerprint to identify the horizontal
mirror in branch 5 and the two-row shift in branch 6. Its turn 19 explicitly
credits that fingerprint; turn 20 applies the spatial repairs. Gemini also shared
the always-active-unit/inhibitory-column clue for branch 7, which terra used in
its later constrained permutation test. This is evidence of information transfer,
not a causal ablation proving that the peer interaction alone caused the gain.

The benchmark's planted faults leave strong structural clues, including pixel
columns that remained near initialization. A win here would still be specific to
this repair setting, not general superiority on complex work.

## Why the audit rejects a clean claim

Gemini's tool record 6 in the corrected peer run performed this sequence:

1. Load recorded layer-2 activations.
2. Change the layer-2 bias or swap/shift the output head in local arrays.
3. Apply ReLU and compute `l3_new = h2 @ w3.T + b3`.
4. Measure the hypothetical repair's AUC and positive/negative accuracy.

The code called this analysis of saved activations, but it is a counterfactual
forward pass through a modified network tail. The existing rule forbids running
or simulating the network in the kernel. An absence of gradient training or
`fit()` calls is insufficient to pass this audit. The winner itself was manually
reviewed; it used weight/activation analyses and official forward calls, but a
shared run cannot ignore a contributing peer's violation.

Both solo Claude and solo Gemini were interrupted by guards, so their saved
candidates cannot establish their fully resourced solo performance. The first
peer attempt was also discarded after its byte-count guard stopped members early.

## Cost and accounting limits

The local reports total $5.9685 including the discarded attempt and a tiny usage
probe. **That number may understate actual cost.** Twenty nonempty Gemini responses
reported zero output tokens. At their requested maximum of 32,768 output tokens,
the missing counts could add $6.5536, producing $12.5221 of potential exposure
assuming the requested limits were honored. Actual billing has not been verified,
so the initial $12.50 pilot ceiling cannot be certified from these reports.

Future guarded calls preserve the reported usage, flag zero-token nonempty output,
and charge its requested output ceiling rather than treating it as free. No more
Gemini calls were launched after this issue was identified. Pricing assumptions
are linked in [the original plan](peer-pilot-plan.md).

Machine-readable scores, stopping reasons and audit findings:
[peer-pilot-results.json](peer-pilot-results.json). Local full traces remain under
`.wsplab/experiments/peer-pilot-20261105/` and
`.wsplab/experiments/peer-pilot-corrected/`.
