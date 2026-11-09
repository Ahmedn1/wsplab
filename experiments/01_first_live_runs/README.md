# First live runs

The first runs with real editors.

| Run | Script | Typical command |
|---|---|---|
| Digit classifier, two output rows swapped | `code/pilot.py` | `python code/pilot.py <model> <cap_usd> <folder>` |
| One editor, any model | `code/solo.py` | `python code/solo.py <model>` |
| Gemma 3 270M, 5 passes | `code/gemma.py` | `python code/gemma.py openai:gpt-6-sol 2 runs/gemma5` |
| Qwen3.5-0.8B with Astra, 12 passes | `code/astra.py` | `python code/astra.py gpt-6-astra 10 runs/astra12 12` |

`code/common.py` holds the editors and per-provider spending ledgers these scripts share. Results are in the table under
"Single-editor runs" in [FINDINGS.md](../../FINDINGS.md). The agent-from-the-skill test builds its task with
`code/make_agent_task.py`. `code/watchdog.py` stopped runaway calls, and `code/audit*.py` and `code/hard_cal.py` are the
audits and calibration behind the early council results.
