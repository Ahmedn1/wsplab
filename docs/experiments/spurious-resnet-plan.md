# Spurious ImageNet on ResNet-50: council versus solo (pre-registration)

Status: fixture frozen, offline dry run passed, **no paid call made**. A paid run needs the user's approval
of the budget below.

## Why this task

The torchvision ResNet-50 (IMAGENET1K_V1, sha256 `0676ba61…`) is unmodified. It labels many real photos as a class
whose object isn't in the photo, because the photo shows that class's usual background or context: water or
coral becomes "snorkel", ice becomes "puck". Nothing is injected, and the photos come from the official
Spurious ImageNet benchmark (Neuhaus et al., ICCV 2023). The task is multi-objective: remove the false predictions
without damaging clean accuracy on the same classes or any other class. In an independent published trial,
three solo Claude sessions each narrowly missed a preservation bar.

## Fixture (`scripts/live/spurious_resnet.py`, `~/.wsplab/spurious-resnet/fixture`)

- **Spurious photos:** Dropbox mirror named in the official README (archive sha256 `ce3ef1b6…`); the official
  Tübingen server was unreachable. 100 classes × 75 photos.
- **Clean photos:** the three ImageNetV2 variants, deduplicated by content (20,683 photos, all 1,000 classes),
  plus the 50 ImageNet-1k validation photos of each of the 100 spurious classes.
- **Classes:** 50 target classes chosen with a fixed seed (20261108). The other 50 are an unseen transfer test,
  and editors are never told about them.
- **Practice (4,850 items):** 25 of 75 spurious photos per target class, 15 clean photos per target class, and 3
  clean photos per other class. The harness halves it into the editors' work set and a selection set.
- **Evaluation (28,325 items, never mounted in the sandbox):**
  - 2,498 target spurious photos
  - 3,744 unseen spurious photos
  - 2,741 target clean photos
  - 19,342 other clean photos

  Eight photos that sit in two classes' folders appear in practice and were dropped from evaluation.
- **Preprocessing (as in the benchmark):** spurious photos are resized to 224 with bicubic interpolation; clean
  photos are resized to 256 and centre-cropped to 224. The network takes uint8 input and normalizes internally.
- **Score (identical for editors and evaluation):** the item-weighted mean of three equal groups:
  1. Target spurious photos that are *not* called their class.
  2. Clean target-class photos that are correct.
  3. Clean photos of all other classes that are correct.

  Unseen spurious photos have weight 0 and are reported separately.

### Baseline (evaluation set)

| Measure | Baseline |
|---|---:|
| Balanced score | 0.6762 |
| Target spurious hit rate | 46.0% |
| Unseen spurious hit rate | 49.7% |
| Target-class clean top-1 | 78.58% |
| Overall clean top-1 | 71.30% |

These reproduce the public model: about 48% spurious hits across the 100 classes, and 63.1% on ImageNetV2
matched-frequency alone.

## Comparator: the published recipe, re-implemented from its description

`scripts/live/spurious_recipe.py` uses only the work set. For each target class, it damps the fc weights of the
channels that fire on that class's spurious photos but not on its clean photos, then boosts the object channels.
On the selection set:

| Setting | Balanced | Target spurious hit | Target clean | Overall clean |
|---|---:|---:|---:|---:|
| Original | 0.682 | 47.8% | 82.8% | 72.4% |
| s1 (32 cue ×0.25, boost) | 0.791 | 13.9% | 82.0% | 72.0% |
| s2 (50 cue ×0, 50 boost ×1.5) | 0.800 | 9.9% | 80.9% | 71.6% |

**Risk stated in advance:** a simple recipe already removes most spurious hits. Every editor may converge on
something similar, and then council and solo will tie. Such a tie is a valid, reportable result. The recipe is a
comparator every arm is measured against, and editors are never shown it.

## Arms (`scripts/live/spurious_trial.py`)

- **Peers:** OpenAI (`gpt-6.1-terra`), Claude (`claude-opus-5-5`) and Gemini (`gemini-4-argon`). Each keeps its
  own weights; after every step each sees what the others did and what came back.
- **Independent:** the same three models alone, with the same caps, prompts and limits. This arm yields each solo
  result and the **no-communication portfolio** (its best candidate on the selection set).
- **Shared settings:** 8 forward passes, 40 Python runs and 60 turns per editor; effort high; 16k output tokens
  per call; all tensors editable; GPU forward tool with activation capture.
- **Rules (stated in the prompt):** no training, fitting or regression, and no computing outputs of modified
  weights inside Python, including replaying fc on recorded activations. The CIFAR and FinQA audits showed
  this rule needs checking, so it is audited by hand.

## Outcomes (fixed before any paid call)

**Primary:** balanced evaluation score, peers' selected candidate minus the portfolio's selected candidate, paired
by item. The selection rule is the harness's: highest selection score; the original is kept if no candidate beats
it. A council advantage is claimed only if the difference is positive with p < 0.05 (paired, item-level), the
protocol audit passes, and, if two replicates run, both have the same sign.

**Secondary:**
- Peers minus each solo.
- Each arm against the original and against recipe s1 and s2.
- The components: target spurious hit, target clean, overall clean.
- Unseen-class transfer.
- Cost per arm.

**Pass bar for "repaired"** (on evaluation, mirroring the published bar):
- Target spurious hit ≤ 23.0% (half of baseline).
- Target clean top-1 ≥ 75.58% (baseline − 3).
- Overall clean top-1 ≥ 70.80% (baseline − 0.5).
- A same-size random edit does not meet the first condition.

**Audit before any evaluation number is read:** every Python block and forward request, screened and then read
by hand, as in `natural-finqa-intervention-results.md`. The harness computes the winner's evaluation score
automatically; the trial script prints only status and spend.

## Budget

| Per arm | OpenAI | Claude | Gemini | Total |
|---|---:|---:|---:|---:|
| Cap | $0.75 | $3.00 | $3.00 | $6.75 |

One replicate (both arms) costs at most **$13.50**. A second replicate, run only if the user agrees after seeing
the first, would bring the total to $27. Approximate headroom under the cumulative limits ($22 / $47 / $30) is
about $7 OpenAI, $19 Claude and $12 Gemini, so one replicate needs no limit increase.

## Validation done (no spend)

- Fixture baseline reproduces the public model (above). The source image hashes are kept in
  `*_sources.json`; the raw images were deleted for disk space and can be downloaded again.
- Offline dry run: three scripted peers on the real fixture. Each saved an fc edit, ran a GPU forward pass with
  `avgpool` activations (2,425 × 2,048), and produced distinct patches. Selection, the final evaluation, the random
  control and the verdict all ran.
- Framework fix: the torch forward tool now captures activations in the scorer's batch size, so large networks
  don't run out of GPU memory. Covered by a new test.
