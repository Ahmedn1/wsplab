# MGSM controls continuation — 2026-11-06

Continue the [frozen task screen](natural-mgsm-screen-plan.md) using only the
original Qwen3-0.6B weights and pilot groups. No paid editor/provider calls,
training, weight edits, new data partitions, or external evaluation.

## CPU work while Astra finishes

On the same eight groups as the first feasibility run:

1. All three languages: request reasoning and an answer in English, with
   presence penalty 1.5. This is a combined cheap remedy; it does not isolate
   the contributions of prompting and the penalty.
2. Bengali and Swahili only: ask the small model to translate the problem into
   English without solving it, also with presence penalty 1.5. Review whether
   quantities and relationships match the published English counterpart.

There are 24 math responses and 16 translations. Use CPU float32, two threads,
temperature 0.7, top-p 0.8, top-k 20, the original per-problem seeds, and 2,048
output tokens. This remains a short-output feasibility check. Translation
responses have no numeric-answer accuracy assigned: copying a number from the
question must not count as solving it. A two-hour active limit is checked
between requests; partial work remains explicitly incomplete.

## Full GPU pilot after Astra releases the GPU

Use all **32 frozen groups**, with translations grouped exactly as before.
Run these conditions in this fixed order:

| Condition | Thinking | Presence penalty | Instruction |
|---|---|---:|---|
| nonthinking_pp | off | 1.5 | Original model-card math instruction |
| english_reason | off | 1.5 | Original question, explicitly reason/answer in English |
| localized | off | 1.5 | Math instruction in the question's language |
| thinking_pp | on | 1.5 | Original model-card math instruction |
| nonthinking | off | 0 | Original model-card math instruction |
| thinking | on | 0 | Original model-card math instruction |
| translate | off | 1.5 | Translate only; Bengali and Swahili |

Six math conditions produce 576 responses; the translation probe produces 64.
The localized instructions are authored by the assistant and have not been
checked by native speakers; they are a diagnostic control, not a certified
benchmark prompt. Exact strings are stored in execution metadata.

For math, allow up to 32,768 output tokens **within a 32,768-token total
context**, subtracting each prompt's token count. Translation output is limited
to 2,048 tokens. Use temperature 0.6/top-p 0.95 in thinking mode and 0.7/0.8 in
non-thinking mode, top-k 20, and the original seeds. These settings and the
presence-penalty remedy come from the pinned local model card.

The GPU engine uses the original BF16 checkpoint with vLLM, eager execution,
at most two concurrent sequences and 70% GPU memory. Re-run all GPU baselines;
CPU/FP32 and GPU/BF16 stochastic outputs are not interchangeable measurements.
The script records exact installed library versions and its source snapshots.

Before allocating GPU memory, the queue waits for the original Astra PID and
start identity to disappear, then requires two consecutive idle checks with at
least 14,000 MiB free and utilization below 10%. It never stops or restarts
Astra. Waiting is bounded at 24 hours. Active screening is bounded at 12 hours,
checked between bounded batches; if exceeded, report partial results instead
of silently reducing token allowances or claiming completion.

## Interpretation and execution evidence

The original qualification rule remains unchanged. Apply it to complete,
adequate-length math conditions, with truncation both overall and by language
reported. Do not use a large baseline gap as sufficient evidence of editability.
If cheap controls remove it, preserve those as strong baselines. If translation
fails to preserve basic meaning, regard missing comprehension as an alternative
to a removable reasoning obstacle. Do not choose any condition from external
evaluation results; that partition remains unused.

Checkpoints and fixture hashes are verified before and after inference. Raw
responses, progress, execution settings and source snapshots are durable under
`.wsplab/natural-mgsm/controls-cpu-8/` and `controls-gpu-32/`.
The root `controls-processes.json` records host PIDs and commands. A completed
response can be resumed only under identical code/settings, and duplicate or
out-of-pilot IDs are rejected.

Validation: seven targeted tests passed for presence-penalty semantics,
unfinished-thinking grading, translation-versus-math grading, and rejecting
invalid resumed records, plus uncertainty clustered by underlying problem.
The live report also tolerates incomplete UTF-8 writes while generation appends results.
Lint passes. Astra and all seven GPU control conditions have now finished; see the [final review](natural-mgsm-controls-review.md).

A read-only watcher updates [the control results](natural-mgsm-controls-results.md)
every 30 seconds and retains completed and partial condition counts. The
watcher never starts additional inference or editor calls.
