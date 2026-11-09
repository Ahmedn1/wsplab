# Genuine-task peer experiment: pretrained ResNet-20 on CIFAR-100

Chosen before scoring any edits: the public `cifar100_resnet20-23dac2f1.pt`
checkpoint from [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models)
at source revision `786c16252c0fc58ee9adac063f8337cc4a7a497a`.
The author reports 68.83% top-1 accuracy on the ordinary CIFAR-100 test set.
The task is classification of 100 real image categories. No faults, image
corruptions, training truncation, or deliberately wrong preprocessing are added.
The model is selected for natural error headroom and CPU feasibility while
Astra occupies the only GPU. No guarantee exists that its errors are repairable
by sparse weight edits rather than requiring more capacity or training.

Data: the official [CIFAR-100 binary archive](https://cave.cs.toronto.edu/kriz/cifar.html),
MD5 `03b5dce01913d631647c71ecec9e9cb8`. Read binary records without unpickling.
Use the author's inference normalization, verified before paid calls. Load
the public checkpoint with `torch.load(weights_only=True)` and verify its hash.
The original model must reproduce approximately the published test accuracy;
otherwise investigate the harness, not declare an artificially low baseline.

Transport amendment before model calls: the original server stalled and reset
connections. Use the same standard test set from
`uoft-cs/cifar100`, revision `aadb3af77e9048adbea6b47c21a81e47dd092ae5`,
on Hugging Face instead. Decode its PNG image bytes into the original NCHW byte
layout and preserve label/index order; record the Parquet and decoded hashes.
This distribution is not claimed to pass the original archive's checksum.
The baseline reproduction and content-isolation requirements still apply.

Fix seed 20261106. From each class's 100 official test images, draw 20 for the
adaptation/selection pool and retain 80 for external evaluation: 2,000 and 8,000
images respectively. The normal search splits the first pool into 1,000 editor
practice and 1,000 hidden candidate-selection images. Preserve original indices
and hashes; check content disjointness. These are partitions of the standard
test set, so the final score is not a full-standard-test benchmark score.

Compare one three-provider peer search with the same three independent editors:
`gpt-6.1-terra`, `claude-opus-5-5`, and `gemini-4-argon`, all at high effort.
All start from the exact same unmodified checkpoint and get the same prompt,
practice examples, metric selection, 8 forward passes, 40 Python runs and 60 turns.
Each reply has at most 16,384 output tokens, reduced near its monetary cap;
missing output usage is charged at this requested ceiling. They may stop early.
Include unchanged weights as the operational fallback when
no selection-set gain is found; also retain the framework's raw winner and gate.
No additional attempts based on observed scores. Individual candidate final
scores are post-run diagnostics, not a way to change the selection decision.

Rules: no training, no optimization of a loss, no recomputing any network tail
or modified-network predictions inside Python. Recorded activations, weight
statistics and correlations are permitted. All model evaluations use the official
forward tool. Audit every contributing peer, including Gemini, and every solo.
A protocol violation disqualifies the affected arm as clean evidence; a stopped
or failed baseline is not evidence of superior reasoning by peers.

Maximum new API allocation: $11.70 total, $5.85 per arm. Per arm:
OpenAI $0.35, Anthropic $2.00, Gemini $3.50. These fit the previously authorized
provider budgets after previous experiment charges, conservative allowance for
missing historical Gemini usage, and a $17 working allocation for Astra's
original OpenAI ledger. Anthropic has about $4.26 remaining, so budget stops
are a material possibility. Do not silently raise caps or relaunch failures.
Before launch, record a fresh ledger reconciliation. The corrected guard
reserves bounded output before requests and charges nonempty responses with
zero reported output at the full requested output ceiling. Such charges are
estimates, not billing records. Missing usage must never be counted as free.

Primary evidence: selection-chosen peer minus selection-chosen independent
accuracy on the same 8,000 images, alongside each change from the unmodified
baseline, paired query-level uncertainty, actual estimated costs, stopping
conditions, and code audit. A lone positive result is exploratory; a null or
negative result is retained. No claim of cost efficiency or general superiority
without further experiments. The existing Astra run remains untouched.
