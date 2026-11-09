# Paper conformance

Checks that wsplab reproduces what the authors released (RikiyaT/lm_understand-logs @ 4d09ba6d). No LLM calls.

- **Patch rebuilds and prompt replay** are part of the package's test suite: `pytest tests/conformance` (some tests
  download the pinned upstream files and a Qwen checkpoint; marked `network` / `model`).
- **Re-measuring the released 50-pass edit** on GPU: `python code/get_qwen.py` then `python code/conformance_p50c.py`.

Results: every `edited_sha256` matches; all 275 prompts replay byte for byte; `merged.json` recomputes exactly; the
p50c edit re-measures at +48.4 ± 6.3 points against the logged +39.1 ± 6.2. See [docs/fidelity.md](../../docs/fidelity.md).
