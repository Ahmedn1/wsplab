# Astra, 50 passes

`python code/astra.py gpt-6-astra 15 runs/astra50 50` (Qwen3.5-0.8B, 200 MATH-500 problems, 8,192-token answers;
`--resume` continues a crashed run). `code/monitor_run.py` watched it.

Result: 9.5% → 14.0%, verdict `not_significant` (gain 4.5 against a gate of 4.55; exact paired p = 0.078). Report:
[astra50-result.md](astra50-result.md), numbers: [astra50-result.json](astra50-result.json).
