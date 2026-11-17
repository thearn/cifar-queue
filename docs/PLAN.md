# cifar-queue: plan

The decisions the code depends on (the template's `docs/PLAN.md` sections). Mode 2's own plan and log are
`docs/PLAN-mode2.md`; measurements are in `docs/RESULTS.md`.

## Problem

A test problem for runq: an image classifier with a well-known reference, small enough that a search of
dozens of trainings fits in an hour, exercising everything an ML workload needs from the queue (an environment and a
dataset shipped to remote GPUs, trainings as function calls, a search spread over several GPUs, a whole search inside
one VM job). "Good enough" is matching Page's recipe on the same split; beating it is a bonus.

## Data

- CIFAR-10 from the Hugging Face hub (`uoft-cs/cifar10`), its parquet files at revision `0b27149...` (pinned in
  `configs/data.toml`); 60000 32x32 RGB images, 184 MB decoded. Licence and terms: `docs/DATA.md`.
- A row is an image; its id is its index in the hub's file (`train:123`, `test:45`).
- No dirt: decoding refuses wrong-sized images and unknown labels, and any dropped image fails the stage.

## Metric and splits

- The primary model-comparison and new-campaign objective is the
  [relative error-cost score](metrics/README.md): maximize relative validation-error reduction with latency
  weight `0.25` and deployed-byte weight `0.10`, reporting the frozen d3 reference as 100. Measure complete
  prediction pipelines under one fixed serving protocol; NetScore is secondary.
- Legacy training/search drivers still use validation accuracy (`evaluate.OBJECTIVE`) for screening. They do
  not yet measure the cross-family end-to-end objective. Final promotion requires the primary score and an
  uncertainty assessment; this adoption does not retroactively change historical studies or their objectives.
- The new [ResNet-18 campaign](autoresearch/2026-11-16-resnet18/PLAN.md) optimizes the primary score with
  host-to-host evaluation, freshly remeasured d3, ImageNet/scratch baselines and conditional Optuna.
- After its boosting follow-up, the [FoldGrid proposal](autoresearch/2026-11-16-resnet18/ARCHITECTURE-NEXT.md)
  supplies a literature-grounded JAX structural track alongside compact-CNN distillation. It is unrun;
  training needs a separately bounded budget and must preserve the same evaluator and reference.
- Test accuracy is reported separately, never used to select candidates. Disclose prior test exposures.
- Validation: 5000 training images, the first 5000 of `numpy.random.default_rng(1234).permutation(50000)`, kept since
  the first study (not the template's hash split; changing it would make every study incomparable). Test: the hub's
  10000.

## Reference

The primary score's reference is the saved d3 recipe, not necessarily the best-scoring model. Its
[identity, historical normalization and remeasurement requirements](metrics/README.md#current-reference-and-provisional-leader)
are fixed in the metric specification. There is no default hard cap at its size or compute; hard limits require
an actual deployment or campaign resource constraint.

The original accuracy-search reference remains David Page's DAWNBench recipe (ResNet-9, 24 epochs) as a point of the space (`tune.REFERENCE`, `tune.PAGE`): 93.5 %
validation, 93.1 % test in one run on the 4060 (154 s); 94 % in his own runs on all 50000 images.

## Compute

- One training at the defaults (16 epochs): 37 s on an L4. Page's recipe: 154 s on the 4060. `tune.job_options`
  sizes each trial by width and epochs.
- Targets in `configs/compute.toml`: the 4060 (`local`), two L4s, a G4, an A100. One training at a time per card was
  best on every card measured.
- Early stopping: `tune.PRUNE` (rungs at 10/30/60 %, two thirds kept, best so far), chosen with
  `pipeline/tools/rung_check.py` from 161 complete curves. It needs the trials to reach the study, so pruned searches
  run on the 4060 (`50_tune.py --targets local --prune rungs`) or inside one VM job (`55_tune_on_vm.py`).

## Publication

`thearn/cifar10-resnet9` (the code at HEAD, the champion's network, the card) and `thearn/cifar10-resnet9-demo` (the
Gradio app; not hosted, a Gradio Space needs a paid plan). Apache-2.0. The card says how the champion was chosen, the
spread over seeds, and the dataset's hash.
