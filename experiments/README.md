# Experiments

Everything needed to reproduce the results in [FINDINGS.md](../FINDINGS.md). The `wsplab` package stays clean: code
that only these experiments need lives here, not in `src/`.

| Folder | Question | Paid? |
|---|---|---|
| [00_paper_conformance](00_paper_conformance) | Does wsplab reproduce the paper's released edits, prompts and scores? | No |
| [01_first_live_runs](01_first_live_runs) | Do real editors confirm fixes on small networks and LLMs? | Yes |
| [02_astra_50_passes](02_astra_50_passes) | Does a 50-pass Astra run on Qwen3.5-0.8B pass the gate? | Yes |
| [03_early_councils](03_early_councils) | Do chair councils, teams or peers beat single editors on digit tasks? | Yes |
| [04_routed_digits](04_routed_digits) | Do peers beat solo editors when separate repairs add up? | Yes |
| [05_cifar100](05_cifar100) | Can editors improve a natural public ResNet-20? | Yes |
| [06_mgsm](06_mgsm) | Is Qwen3-0.6B's multilingual math gap a fixable fault? (screen only) | No |
| [07_hotpotqa](07_hotpotqa) | Is there a usable context-reasoning gap? (screen only) | No |
| [08_finqa](08_finqa) | Can a council improve Qwen3-0.6B on financial QA? | Yes |
| [09_spurious_imagenet](09_spurious_imagenet) | Can editors remove ResNet-50's spurious-cue errors without losing accuracy? | Yes |
| [10_seaquest](10_seaquest) | Strong council, cheap council and role council on Astra's Seaquest policy | Yes |
| [11_gemma_math](11_gemma_math) | Can anyone improve Gemma 3 270M's math, the paper's small-gain case? | Yes |
| [paper](paper) | The paper's eight experiments as wsplab configuration | Optional |

## Layout

- `code/` holds every experiment script, flat, because the scripts import each other by name
  (`peer_trial.py` holds the shared editors, spending ledger and guard). Run them from that folder or with
  `PYTHONPATH=experiments/code`.
- `code/legacy_council.py` holds the council variants that wsplab no longer ships (chair council, team, vote, roles).
- `<NN_name>/` holds each experiment's README, plans, reports and a `results/` folder with the small JSON files
  behind the numbers and charts.
- `make_figures.py` redraws every chart in `docs/figures/` from those result files.
- `tests/` holds the tests for the experiment scripts: `pytest experiments/tests`.

## Running

- **Environments:** `pip install -e '.[all]'` for CPU work. For GPU work add `.[vllm]`, with
  `-c constraints-gpu.txt`.
- **MinAtar is GPL:** install it yourself (`pip install minatar==1.0.15 --no-deps`). It is never shipped with
  wsplab.
- **API keys:** paid scripts read them from `/tmp/<provider>_key.txt` (`openai`, `anthropic`, `gemini`). Every paid
  script takes per-provider dollar caps, and a spending guard stops a session before a call that could cross its
  cap.
- **Outputs:** large outputs (checkpoints, transcripts, generations) go to `~/.wsplab/...` or `.wsplab/` and are not
  in the repository.
- **The pre-0.1 tag:** a few early scripts depend on behaviour removed in 0.1.0. `code/ablate_job.py` uses voting on
  finished candidates. Run those from the git tag `pre-0.1-cleanup`.
- **Notifications:** `code/notify_watch.py` and `code/NOTIFICATIONS.md` are the job watcher used while Codex ran some
  of these experiments.
