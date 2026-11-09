# Original Astra 50-pass run — final result

The original background run has completed all 50 editing passes and its final
evaluation. The framework verdict is **`not_significant`**.

| External evaluation | Result |
|---|---:|
| Original Qwen3.5-0.8B | 9.5% (19/200) |
| Edited model | 14.0% (28/200) |
| Paired difference | +4.50 percentage points |
| Paired standard error | 2.27 percentage points |
| Wrong to right / right to wrong | 15 / 6 |

This uses 200 MATH-500 problems, one sample each, and an 8,192-output-token cap,
with temperature 0.6, top-p 0.95, top-k 20 and presence penalty 1.5. It is not a
replication of the paper's full MATH-500, multi-sample, 32,768-token protocol.

The submission changes 134,144 entries: doubling columns 256:384 of layer 22's
linear-attention output projection and zeroing columns 22, 46 and 94 of layer
23's MLP down projection. These edit descriptions come from the final answer;
the framework reports the changed-entry count.

Astra separately reports an internal practice-holdout gain of 5.60% to 8.59%
over 128 problems with 12 samples each. That editor-reported analysis does not
replace the external verdict. The full trace has not been audited in this
continuation, so neither the internal claim nor protocol compliance is newly
certified here.

The original process was left untouched and exited normally after writing its
report. The [copied final report](astra50-result.json) includes its source path
and SHA-256. Monitoring history remains in `.wsplab/monitor/astra50/`.
