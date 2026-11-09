# Changelog

All notable changes to wsplab. The project follows [Semantic Versioning](https://semver.org/); before 1.0, minor
versions may change the API.

## Unreleased

### Added

- The project is on GitHub: https://github.com/Ahmedn1/wsplab. Project URLs, citation and a CI badge point to it.
- A logo (`docs/logo/`), shown at the top of the README.
- Command line: `run --max-changed` (the edit budget) and `run --price EDITOR=IN,OUT`, so `--max-usd` works for a
  council. `eval` and `infer` accept a Hugging Face id for a language model, and a `--revision`.
- README: a command-line section with examples for every command, and how to use an open-source editor served by
  vLLM or another OpenAI-compatible server.
- README: where each editor finds its API key.
- Progress and logs: a run reports each step, generation progress with time left, and one line per editor turn, and
  writes the same to `run.log` in its folder. `quiet=True` in Python, `wsplab -q` and `wsplab -v` on the command line.
  The vLLM worker reports progress to the run instead of having its output discarded.

### Fixed

- `wsplab.run(..., eval=..., max_changed=N)` ignored `max_changed` for language models.
- `wsplab run --max-usd` failed on the first call: there was no way to give prices from the command line.

## 0.1.0 (2026-11-08)

First release.

### Added
- End-to-end runs for language models (vLLM and Hugging Face) and any PyTorch network: baseline, a sandboxed editing
  session, building the edit, a hidden evaluation, and a verdict from a gate with paired statistics and a random
  control.
- The paper's method: the v5 base prompt, the plain-text turn protocol, a persistent kernel over lazily loaded
  tensors, the budgeted forward-pass tool and handoffs. Replay of the authors' sessions matches byte for byte.
- Shareable patches: the authors' format (plus dense and int64 forms), apply, verify, stack, import, and Hugging Face
  Hub push and pull.
- Semantic addressing of components for common families, with a fallback to raw tensors.
- Editors for OpenAI (Responses API), OpenAI-compatible servers, Anthropic, Codex and coding agents (`wsplab
  session`), with spending limits that count thinking tokens.
- An optional council of peers (`search(..., shared=True)`, or repeating `--editor` on the command line).
- An edit budget (`max_changed`) that `save_edit` enforces.
- Crash resume, answer caching for unchanged weights, `wsplab doctor` and `wsplab setup`.
- An agent skill in `skills/wsplab`.
- `experiments/`: the code, plans and result files for every experiment in FINDINGS.md.

### Removed (before release)
- The chair council voting on edit scripts, the team with a lead, voting on finished candidates (`selection="vote"`)
  and fixed council roles. Measured results did not favour them. They are kept in
  `experiments/code/legacy_council.py` for reproduction, and the git tag `pre-0.1-cleanup` marks the last commit that
  shipped them.

### Changed
- `search()` keeps the original model when no candidate beats it on the selection set.
