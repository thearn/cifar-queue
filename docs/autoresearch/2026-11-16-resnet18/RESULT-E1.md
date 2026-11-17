# E1 result: completed ResNet-18 recipes do not beat d3

Finished 2026-11-17, queue job 47. Frozen model/evaluator source:
`e5b6bb18fdf2fcc92a31371a1d97baa864f79dfc`; the explicit 50-entry cap amendment
is recorded in `results/autoresearch/resnet18-score/trial-cap-50.json`.

**Outcome: rejected as an improvement over d3 within the completed recipes.**
The bounded workflow succeeded, but the search ended before its 50-entry maximum:
18 COMPLETE entries (two baselines and 16 sampled trials), followed by five
per-trial timeouts recorded as Optuna FAIL. Those triggered the frozen
five-consecutive-failures safeguard. There were 23 study entries, not 50
completed trials. The five timed-out candidates have no valid completed score;
their quality remains unmeasured, not rejected on accuracy.

## Complete-pipeline comparison

One view, validation split, same local T4 host, decoded host uint8 RGB to host
class IDs, batch 256. Costs include preprocessing, resizing and transfers.
FP16 arithmetic does not imply FP16 storage: saved ResNet artifacts still occupy
approximately 44.81 MB. ImageNet pretraining is disclosed, not treated as CIFAR-only.

| Model / evidence | Validation accuracy | ms / 256 | MB | Primary score |
|---|---:|---:|---:|---:|
| Frozen d3 reference, 16 seeds | 94.934% | 3.705 | 7.21 | 100.00 |
| Selected ImageNet recipe, fresh seeds 71-73 | 93.720% | 7.719 | 44.81 | 55.93 |
| Same ImageNet recipe, original seed 40 | 93.300% | 7.701 | 44.81 | 52.46 |
| Best completed sampled trial, scratch 96px, seed 40 | 94.120% | 16.060 | 44.81 | 49.74 |
| Scratch baseline, seed 40 | 92.160% | 7.747 | 44.81 | 44.76 |

The original ImageNet baseline remained the highest-scoring completed study
entry. No sampled trial improved on it. The scratch trial with the highest
accuracy was also the best sampled score, but its approximately doubled latency
relative to the ImageNet baseline outweighed the accuracy improvement.

The selected recipe is canonical ImageNet stem, 64px input, full-network
ImageNet-1K initialization, SGD LR .01, momentum .9, weight decay .0005, batch
256, 30 epochs, three warmup epochs, cosine floor .01, basic augmentation, and
no label smoothing, mixup, cutmix, erasing or backbone freezing. This practical
recipe comparison is not a matched-schedule causal pretraining experiment.

## Fresh-seed checks

| Seed | Validation accuracy | Median ms / 256 | Score |
|---|---:|---:|---:|
| 71 | 93.84% | 7.721 | 57.02 |
| 72 | 93.74% | 7.732 | 56.09 |
| 73 | 93.58% | 7.702 | 54.75 |

Mean accuracy is 93.72%, sample seed SD **0.13115 percentage points**. Compute
the recipe score from that mean accuracy, the median of all 300 equally sampled
request times (**7.719451 ms**) and mean bytes (**44,805,696**), yielding
**55.934605**. It is not the arithmetic mean of the three nonlinear scores.
These are three fresh training processes/seeds, not guaranteed statistical
significance or an untouched-data generalization assessment.

## All completed study entries

Accuracy is percent; `ImageNet stem` means canonical 7x7/stride-2 plus maxpool,
whereas `CIFAR stem` is 3x3/stride-1 without the initial maxpool.

| Trial | Initialization | Stem | Input | Epochs | Accuracy | ms / 256 | Score |
|---|---|---|---:|---:|---:|---:|---:|
| 0, baseline | ImageNet | ImageNet | 64 | 30 | 93.30 | 7.701 | 52.459 |
| 1, baseline | Scratch | ImageNet | 64 | 100 | 92.16 | 7.747 | 44.765 |
| 2 | Scratch | ImageNet | 96 | 90 | 94.12 | 16.060 | 49.742 |
| 3 | Scratch | ImageNet | 48 | 90 | 67.96 | 5.050 | 12.190 |
| 4 | ImageNet | CIFAR | 48 | 60 | 92.70 | 43.031 | 31.318 |
| 5 | ImageNet | ImageNet | 96 | 100 | 91.52 | 16.080 | 34.480 |
| 6 | ImageNet | CIFAR | 32 | 20 | 89.46 | 19.422 | 26.464 |
| 7 | ImageNet | CIFAR | 32 | 90 | 93.94 | 19.357 | 46.066 |
| 8 | ImageNet | ImageNet | 48 | 80 | 76.52 | 4.982 | 16.692 |
| 9 | Scratch | ImageNet | 64 | 20 | 51.70 | 7.876 | 7.236 |
| 10 | Scratch | CIFAR | 64 | 50 | 92.88 | 72.908 | 28.144 |
| 11 | Scratch | ImageNet | 96 | 50 | 93.16 | 16.182 | 42.680 |
| 12 | Scratch | ImageNet | 96 | 70 | 93.28 | 16.116 | 43.486 |
| 13 | Scratch | ImageNet | 96 | 40 | 92.94 | 16.154 | 41.368 |
| 14 | Scratch | ImageNet | 96 | 30 | 86.94 | 16.230 | 22.336 |
| 15 | Scratch | CIFAR | 64 | 70 | 91.24 | 75.004 | 22.714 |
| 16 | Scratch | ImageNet | 32 | 100 | 46.18 | 3.281 | 8.084 |
| 17 | Scratch | ImageNet | 96 | 40 | 86.54 | 16.038 | 21.737 |

## Failed/budget-stopped attempts and limitations

Trials **18-22** all sampled ImageNet initialization, CIFAR stem, 96px input,
SGD, batch 256, RandAugment, CutMix 1 and erasing .25. Requested epochs were
70, 70, 70, 80 and 70. Each ran about **3,519 seconds** before the internal
training-time guard ended it; the outer per-run ceiling was one hour.
The retained logs show training progress, then
`TimeoutError: training reached the declared per-run budget; no completed result`.
This was not a CUDA out-of-memory failure or evidence of stalled hardware.

Exact per-trial configurations, timestamps and states remain in `optuna.db`,
`trial-XXXX.request.json`, saved model configs/partial curves and the queue
campaign log. Do not rank incomplete epochs or invent objectives for them.
The repeated expensive region and failure-streak stop limited exploration.
This result does **not** show that all ResNet-18 configurations are inferior,
nor that 50-trial tuning was completed.

Conservative cumulative wall charge is **30,510.021 seconds (8.475 hours)**,
including the carried-forward pre-restart spending. The earlier intentional
coordinator-only restart is separate from these five budget-stopped candidates.
No new test read was performed.

## Reproduce and handoff

The compact, versioned [final measurement cohort](../../metrics/resnet18-final.json)
retains the unchanged normalization and links to the raw artifacts:

```bash
PYTHONPATH=src .venv/bin/python -m cifarq.efficiency docs/metrics/resnet18-final.json
```

Multiply the CLI scores by 100. The report's aggregated result was independently
recomputed from the three confirmation JSON files and matched the driver's
summary. `summary.json` SHA256:
`bb3c80cf60b8d8051d24f96748287899e9852b723ef027c907696e59438dede2`.

Dependent **job 48 started automatically** after 47 succeeded. It uses the best
completed ResNet recipe's encoder where requested, counts that encoder and all
feature/transform/tree costs, and remains capped at four hours within the unused
original 24-hour allowance. Do not restart or extend E1 merely to fill the cap.
The proposed FoldGrid architecture is still unrun and has no separate compute
approval.
