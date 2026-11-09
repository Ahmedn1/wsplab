# Routed digits

A synthetic task built so separate repairs add up: ten binary digit detectors, eight of them damaged.

- Build and calibrate: `python code/modular_check.py --out .wsplab/calibration/recheck.json`
- A scripted end-to-end check: `python code/modular_job.py --out .wsplab/smoke/run`
- Paid arms: `python code/peer_trial.py --arm peers --out <dir> --seed 22 ...` and `--arm independent`; report with
  `python code/peer_trial_report.py <dir>`.

Reports: [routed-digits.md](routed-digits.md) (design), [peer-pilot-results.md](peer-pilot-results.md) (disqualified
pilot), [peer-confirmation-results.md](peer-confirmation-results.md) (the 88.8% vs 83.5% result).
