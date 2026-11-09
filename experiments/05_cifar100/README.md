# CIFAR-100 ResNet-20

The unmodified public ResNet-20 (chenyaofo/pytorch-cifar-models; `code/vendor/cifar_resnet.py` with its license).

- Fixture: `python code/natural_cifar.py <folder>` (or `--convert-hf` from the Hugging Face copy)
- Arms: `python code/natural_trial.py --arm peers --fixture <folder> --out <dir>` and `--arm independent`
- Report: `python code/natural_report.py <dir>`

Reports: [natural-cifar100-results.md](natural-cifar100-results.md) and the plans next to it.
