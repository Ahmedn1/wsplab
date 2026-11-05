# Provenance

wsplab contains no code from other weight-space programming frameworks (weightspace, OpenWSP, Oumi Weights Lab, or the
community skills). Every file is either new or derived from the paper's repository, pinned at
`RikiyaT/lm_understand-logs@4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. `upstream.lock.json` holds the sha256 of each
upstream file at that commit, and a test checks them.

| wsplab file | Derived from | License | What changed |
|---|---|---|---|
| `src/wsplab/io/safetensors.py` | `exp8_math/code/st_numpy.py`, `exp7_sycophancy/runs/attention_tool/sandbox/prelude.py` | MIT | Lazy reads, shards, header checks, checked casts |
| `src/wsplab/patch/diff.py` | `exp8_math/code/make_edit_diff.py` | MIT | More dtypes and shapes, dense form, 64-bit indices, bug fixes |
| `src/wsplab/patch/apply.py` | `exp8_math/code/apply_edit.py`, `exp8_math/code/build_ckpt.py` | MIT | Shards, in-place byte writes, hard links, bounds checks |
| `src/wsplab/kernel/client.py` | `harness/sandbox.py` | MIT | Persistent kernel, cgroup memory cap, read-only mounts |
| `src/wsplab/eval/graders.py` | `exp8_math/code/mathqa.py`, `exp8_math/code/evaluate.py` | MIT | Math extraction and checking kept; other graders are new |
| `src/wsplab/eval/stats.py` | `exp8_math/code/merge.py` | MIT | `merge_scores`; paired statistics are new |
| `src/wsplab/runners/vllm_worker.py` | `exp8_math/code/evaluate.py` | MIT | Engine settings |
| `src/wsplab/prompts/templates/lm_edit.txt` | `exp8_math/gemma_followup/runs/g50b/turns/t001/prompt.txt` | CC BY 4.0 | Task-specific words replaced with slots, by `scripts/make_base_template.py` |
| `src/wsplab/prompts/parts/*` | The prompts of experiments 3 to 8 | CC BY 4.0 | Adapted wording for other kinds and modes |
| `src/wsplab/session/messages.py` | `exp8_math/runs/p50c`, `exp8_math/gemma_followup/runs/g50b` logs | CC BY 4.0 | Message wording; nudge, block-count and truncation messages are new |
| `examples/paper/exp6_go.py` | `harness/sandbox.py` | MIT | The prelude's helpers |

Everything else in `src/wsplab` is new unless listed here. The agent driver, persistent kernel and forward-pass server
were not released by the authors: wsplab's versions are reconstructions from the published prompts and logs.
