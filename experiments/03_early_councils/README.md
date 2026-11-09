# Early councils

Digit-classifier tasks with planted faults, comparing single editors with the council variants wsplab first had.
They use `code/legacy_council.py` (`Council`, `Team`).

| Test | Script |
|---|---|
| Two faults, chair council voting on scripts | `code/council.py`, `code/council3.py` |
| Two faults, ablation with a vote on finished weights | `code/ablate.py`, `code/ablate_job.py` (needs tag `pre-0.1-cleanup`) |
| 62 planted faults: solo, chair council, no self-votes | `code/hard_net.py`, `code/hard_job.py <arm> <rep> <folder> [members] [--exclude-self]`, `code/hard_full.py` |
| Pre-registered repeat of the 85.7% run (0 of 5) | `code/hard_repeat.py` |
| Cooperative team with a lead | `code/hard_team.py` |
| A redesigned branch task (not used: repairs didn't add up) | `code/hard2_net.py`, `code/hard2_check.py` |

Results are under "Councils" in [FINDINGS.md](../../FINDINGS.md); the chart is `docs/figures/early-hard-digits-*.png`.
