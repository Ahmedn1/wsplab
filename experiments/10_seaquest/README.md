# MinAtar Seaquest

Improve the Seaquest policy Astra wrote in the paper (`networks/seaquest/hand.npz` in the authors' release).
MinAtar is GPL: `pip install minatar==1.0.15 --no-deps`.

- Fixture and baseline check (52.9 ± 1.4 against the paper's 53.1): `python code/minatar_seaquest.py ~/.wsplab/seaquest/fixture`
  then `--verify`
- Strong arms: `python code/seaquest_trial.py --arm peers --fixture <fixture> --out <dir>/peers --openai-cap 0.75
  --anthropic-cap 2.5 --gemini-cap 2.0`, and `--arm independent`
- Cheap arms: add `--tier cheap` (gpt-6-luna, claude-haiku-5-5, gemini-3.8-flash) with caps 0.30 / 0.60 / 1.25
- Role council: `--roles --arm peers` with `--anthropic-cap 3.5` (uses `legacy_council.Role`)
- Audit screen, then evaluation: `python code/seaquest_report.py <dir>` and `... --evaluate`; the role council is
  compared with `python code/seaquest_roles_eval.py`

Reports: [seaquest-results.md](seaquest-results.md), [cheap-council-and-gemma-results.md](cheap-council-and-gemma-results.md);
numbers in [results/](results).
