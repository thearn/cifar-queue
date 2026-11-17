# E2 result: feature boosting does not improve the complete-pipeline score

Finished 2026-11-17, queue job 48. Frozen source:
`94fb7c866b77d1285b069e5a0a84c8779e1d15a8`.
**Outcome: rejected as an improvement within the tested representations and search.**
All **36 study entries completed**: six baseline feature/learner combinations
and 30 sampled configurations. Three fresh-seed checks also completed; there
were no failed trials or failed confirmations.

## Result against the neural baselines

One view, the unchanged validation split and frozen d3 normalization, batch 256
on the same local T4 host. Latency starts with decoded host uint8 RGB and ends
with host class IDs. All encoder, PCA and tree artifacts count.

| Model / recipe | Validation accuracy | ms / 256 | MB | Primary score |
|---|---:|---:|---:|---:|
| d3, frozen 16-seed reference | 94.934% | 3.705 | 7.21 | 100.00 |
| Selected ResNet-18 recipe, seeds 71-73 | 93.720% | 7.719 | 44.81 | 55.93 |
| Selected embedding/PCA/XGBoost recipe, paired seeds 71-73 | 93.313% | 8.453 | 45.53 | 51.27 |

Replacing the selected ResNet classifier with this fitted PCA/tree pipeline
reduced mean accuracy by **0.407 percentage points**, increased latency, and
increased deployment bytes. Thus it loses to its own encoder's original
classifier as well as d3. This is a result for these features and fits, not a
claim that every tree model or feature representation is inferior.

The selected screen was **trial 23**, score **50.167696** at 93.16%, 8.423309 ms
and 45,518,930 bytes. Its recipe:

- Frozen best completed ResNet-18 backbone, pooled 512-dimensional embedding;
  ImageNet-1K external pretraining remains disclosed.
- Train-only randomized PCA to 64 components.
- XGBoost, 50 multiclass boosting rounds, maximum depth 9, learning rate
  0.195835175, row sampling 0.847554199, column sampling 0.916943762.
- L1 3.139994462, L2 6.912376123, minimum child weight 8.586490772, max bin 255.

The full configuration, including runtime settings, is retained with every
raw assessment and in the final measurement cohort.

## What each representation achieved

The following are the **highest observed scores** within each feature/learner
pair, not necessarily their highest-accuracy settings. They are single-seed
screens, not independent confirmations of all six families.

| Representation | Learner | Trial | Accuracy | ms / 256 | MB | Score |
|---|---|---:|---:|---:|---:|---:|
| Pooled RGB | XGBoost | 16 | 51.54% | 12.099 | 4.07 | 8.23 |
| Pooled RGB | LightGBM | 28 | 47.04% | 10.239 | 0.91 | 9.13 |
| Spatial gradients/color | XGBoost | 2 | 66.36% | 49.947 | 9.23 | 7.67 |
| Spatial gradients/color | LightGBM | 3 | 68.04% | 54.992 | 7.05 | 8.09 |
| Learned ResNet embedding | XGBoost | 23 | 93.16% | 8.423 | 45.52 | 50.17 |
| Learned ResNet embedding | LightGBM | 5 | 92.96% | 11.123 | 46.95 | 45.33 |

Pooled pixels and the tested engineered descriptors were not competitive.
The engineered pipeline also paid approximately 50-55 ms per request in these
best-score settings. Learned embeddings closed most of the accuracy gap but
retained the cost of the full neural encoder; cached features are not a free
inference input.

## Paired fresh-seed checks

Each fit used the corresponding **preexisting ResNet confirmation encoder**,
not the seed-40 encoder held fixed. This reuses the already-charged E1 backbone
training; it is not additional independent backbone training in E2. PCA and
trees were fitted afresh on train data for each seed.

| Seed | Original ResNet accuracy | PCA/XGBoost accuracy | Delta, pp | PCA/XGBoost ms / 256 | PCA/XGBoost score |
|---|---:|---:|---:|---:|---:|
| 71 | 93.84% | 93.28% | -0.56 | 8.453 | 51.02 |
| 72 | 93.74% | 93.48% | -0.26 | 8.457 | 52.58 |
| 73 | 93.58% | 93.18% | -0.40 | 8.444 | 50.28 |

Mean tree-pipeline accuracy is **0.9331333333**, sample seed SD
**0.15275 percentage points**. Aggregating the 300 request times gives median
**8.452740556 ms**. Mean deployment size is **45,530,022.333 bytes**, with
individual sizes 45,522,107 / 45,528,226 / 45,539,734 bytes. Their fractional
mean is a recipe-level statistic, not the size of one fractional file.
Scoring these aggregated inputs gives **51.272123157**, not the mean of the
three individual scores. Three seeds do not establish formal significance.

## Measurement and provenance checks

XGBoost fitting used CUDA; LightGBM fitting and tree prediction used CPU.
Tree prediction used four threads; learned feature extraction used the T4
with mixed-FP16 arithmetic and float32 output features/PCA. Native tree
inference precision is declared rather than presented as a fully FP16 pipeline.

Training-feature caches accelerated fitting only. Each validation/timing
prediction recomputed image features, including neural extraction when selected,
then applied PCA and the learner. All 36 study scores and the aggregate
confirmation score were independently recomputed against the frozen reference.
**171 deployment-file sizes/checksums** were verified across the 36 study models
and three confirmations. No test arrays were read.

Versions include XGBoost 3.2.0, LightGBM 4.7.0, Torch 2.11.0+cu128,
torchvision 0.26.0+cu128, scikit-learn 1.8.0 and NumPy 2.5.5. The isolated
boost environment did not modify the shared neural environment.

The final focused real-backend/evaluator/metric suite passed **53 tests**.
One warning concerns Optuna's experimental multivariate sampler, not a failed
test. No new dependencies were installed during final verification.

## Evidence, cost and next decision

[The final comparison cohort](../../metrics/resnet18-boost-final.json) contains
all 36 boosting study rows/configurations, the confirmed recipe and the five
previous ResNet/reference rows. Its artifact hashes and raw-evidence paths make
the ranking independently inspectable:

```bash
PYTHONPATH=src .venv/bin/python -m cifarq.efficiency docs/metrics/resnet18-boost-final.json
```

Multiply the CLI scores by 100. The
[comparison figure](../../metrics/resnet18-boost-final.svg) is generated from
those scored records, not hypothetical model targets.

E2 charged **749.943 seconds (12.50 minutes)**, well inside its four-hour cap.
Together with E1's 30,510.021 seconds, the conservative experimental wall
charge is **31,259.965 seconds (8.683 hours)** against the original 24 hours.
This includes CPU stages; it is not a GPU-utilization-weighted estimate or
the cost of ImageNet pretraining. No additional/cloud compute was used.

Both requested campaigns and their fresh-seed checks are finished; nothing is
queued. No new contender displaced d3 **within this complete-pipeline cohort**.
Historical compact models still need matching end-to-end measurements before
an overall cross-catalogue champion can be asserted.

The [FoldGrid JAX proposal](ARCHITECTURE-NEXT.md) remains the proposed structural
follow-up, with an equally optimized compact-CNN/KD control. It is not trained,
not a promoted result, and has no separate compute authorization.
