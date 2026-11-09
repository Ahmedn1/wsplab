# Spurious ImageNet, ResNet-50

The unmodified torchvision ResNet-50 on Spurious ImageNet (Neuhaus et al., 2023), with ImageNetV2 and ImageNet
validation photos as clean data.

- Fixture: `python code/spurious_resnet.py ~/.wsplab/spurious-resnet/fixture` (needs the images under
  `~/.wsplab/data`; the Dropbox mirror in the official README works)
- Scripted recipe comparator: `python code/spurious_recipe.py <fixture> <out>`
- Offline dry run: `python code/spurious_dryrun.py <fixture> <out>`
- Arms: `python code/spurious_trial.py --arm peers --fixture <fixture> --out <dir>/peers --openai-cap 0.75
  --anthropic-cap 3 --gemini-cap 3`, and the same with `--arm independent`
- Audit screen, then evaluation: `python code/spurious_report.py <dir>` and `... --evaluate`

Plan: [spurious-resnet-plan.md](spurious-resnet-plan.md). Results: [spurious-resnet-results.md](spurious-resnet-results.md)
and [results/](results).
