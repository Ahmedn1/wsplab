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

Everything else in `src/wsplab` is new unless listed here. The agent driver, persistent kernel and forward-pass server
were not released by the authors: wsplab's versions are reconstructions from the published prompts and logs.
