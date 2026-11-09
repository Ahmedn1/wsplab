# FinQA

Qwen3-0.6B on FinQA numerical questions over financial tables.

| Step | Script |
|---|---|
| Screen and rescoring | `code/finqa_screen.py --root <dir> --prepare`, `code/finqa_rescore.py --root <dir>` |
| Prompt and evidence controls | `code/finqa_controls.py --root <dir>` |
| Local single-edit pilot | `code/finqa_edit.py --root <dir> --out <dir>`, `code/finqa_edit_complete.py` |
| Council vs Claude solo (no edits tested) | `code/finqa_peer_trial.py --root <dir> --out <dir> --run-editors` |
| Required interventions | `code/finqa_intervention_trial.py ... --run-editors`, guard in `code/finqa_intervention_guard.py` |

Each paid runner refuses to start without a recorded approval and refuses external evaluation without a protocol
audit. Reports: [natural-finqa-peer-comparison-results.md](natural-finqa-peer-comparison-results.md),
[natural-finqa-intervention-results.md](natural-finqa-intervention-results.md); audit files in [results/](results).
