# ResNet-18: ImageNet, scratch and score-driven Optuna

Kind: model campaign. Goal: Compare ImageNet fine-tuning and random initialization, then optimize the primary end-to-end error-cost score on a local T4 within 24 GPU-hours.

The evaluation contract is `contract.yaml`. No cloud or shared-environment package modifications.

## Hypotheses

- H1: ImageNet transfer can repay ResNet-18's approximately 11.2M parameters through lower validation error.
- H2: Random-initialization training may improve the fit to CIFAR-10 despite lacking external data.
- H3: The ImageNet 7x7 stride-2 stem and maxpool discard too much small-image detail; jointly tuning resolution,
  the CIFAR 3x3/no-pool stem, and regularization may improve the score. Pretrained CIFAR stems use the center
  3x3 of the pretrained kernel. These are structural choices, not merely optimizer knobs.

## Gates and order

1. Remeasure saved d3 seeds 40-55 at deployed fp16, host-to-host batch 256. Three fresh timing processes.
2. Run a one-epoch scratch smoke. Never treat its score as a completed baseline.
3. Full-network ImageNet-1K fine-tuning: canonical stem, resolution 64, SGD .01, 30 epochs, batch 256.
4. Scratch baseline: same architecture, SGD .1, 100 epochs. Both use cosine decay and basic flip/translation.
   The schedules differ deliberately; this is a practical recipe comparison, not a causal pretraining ablation.
5. Optuna maximizes the primary score with alpha .25 and beta .10. Search initialization, resolution
   32/48/64/96, ImageNet/CIFAR stem, SGD/AdamW and conditional LR, batch 128/256/512, epochs 20-100,
   decay, momentum, warmup/cosine floor, smoothing, basic/RandAugment/TrivialAugment, mixup, cutmix,
   erasing, and pretrained backbone/head LR ratio plus optional warm-start freezing. One view throughout.
   No d3-based hard parameter or MAC cap, and no inaccurate early-epoch proxy pruning. At most 50 total trials
   including the two baselines (user reduced the original 60-trial cap on 2026-11-16),
   one hour each; reserve three hours for confirmation and stop after five consecutive invalid trials.
6. Confirm the best baseline or search recipe on fresh seeds 71-73 within the remaining allowance.
   Report a recipe-level score from mean error, timing samples, and bytes, not a mean of nonlinear scores.

## Execution and recovery

The queue executes `cifarq.resnet18_campaign.run` from a frozen source snapshot. It uses project Python 3.13
for d3 and the existing shared PyTorch interpreter read-only in isolated subprocesses.

Artifacts: `results/autoresearch/resnet18-score/`. `budget.json` charges sequential wall time conservatively
against 24 hours; after an unclean interruption even elapsed downtime is charged. `reference.json` freezes the
normalization, `optuna.db` stores search history, and each completed run has config, curves, weights and
assessment. Incomplete runs retain epoch weights but do not produce completed scores. Completed stages can
be reused after reconciling source identity, queue hours, budget and reference. Changing the evaluator
requires a new cohort and reference.

Dataset SHA256 and no-test-access gates are checked in code. RandAugment/TrivialAugment and erasing use
torchvision batch-shared transform choices; translation and flipping are per image. All timing starts at
decoded host uint8 RGB and ends at host class IDs, including normalization, resizing and transfers.

## Open questions

- Search seed 40 is screening evidence. Three fresh seeds are a first confirmation, not guaranteed significance;
  small apparent improvements may remain inconclusive. The historical forward-only leaderboard is not
  directly comparable to this end-to-end cohort.

## Follow-up requested 2026-11-16: image features and boosted trees

After E1 and its confirmation finish, E2 compares XGBoost and LightGBM on three representations:
8x8 pooled RGB; pooled RGB plus spatial color means/deviations and block-normalized gradient histograms;
and a frozen 512-dimensional backbone embedding from the best measured ResNet-18 recipe. The learned
representation is not a free input: include its saved encoder and config bytes, resizing, normalization,
GPU feature extraction, device transfers and tree prediction in deployment measurements.

Train-feature caches accelerate repeated fitting only; neither validation accuracy nor inference timing
may use them. Optional randomized PCA is fit on train rows alone, serialized and included in both bytes
and prediction time. No test arrays are read. ImageNet external data is disclosed when the chosen encoder
uses it. The user generally prefers JAX for neural work; this follow-up does not train a new neural model,
and preserves the frozen PyTorch encoder's implementation rather than mixing backend timings silently.

M5 runs six baseline combinations (three feature families by two learners). M6 maximizes the same
primary score with conditional Optuna over learner, representation, PCA, depth/leaves, number of rounds,
learning rate, sampling, regularization and histogram bins. No accuracy-only early stopping or cached-feature
inference proxy. At most 36 study entries, 15 minutes per search trial; reserve up to 30 minutes for seeds
71-73. Learned-feature confirmation uses the matching preexisting fresh-seed encoder when available;
otherwise disclose the fixed encoder and do not claim independent backbone replication.

**No new compute approval:** the user was unavailable for a budget decision. Conservatively allow at most
`min(4 hours, 24 hours - actual completed ResNet18 campaign wall time)`, including CPU and GPU stages.
The driver refuses to run while ResNet-18 is active, charges elapsed downtime after unclean interruption,
and records budget-stopped if no material allowance remains. No cloud. Queue E2 only after job 47.

Implementation: `cifarq.image_boost`, `cifarq.boost_campaign`, `pipeline/tools/image_boost.py`.
Artifacts: `results/autoresearch/image-boost/`, with complete-pipeline assessments, all fitted deployment
artifacts, PCA, feature-cache provenance, SQLite study, budget and final summary.
The `.venv-boost` environment inherits the shared PyTorch/XGBoost/sklearn installation through a read-only
site-package path; LightGBM and pytest were installed only in the isolated environment. No shared package
was installed or upgraded.

Cheap gates pass before queueing: both real tree backends save/reload/predict from images, repeated
predictions recompute features, NaN features are rejected, PCA fits only train, exported encoders preserve
the backbone, and recovery cannot exceed the existing approval. CPU-only combined suite: 53 tests passed.
