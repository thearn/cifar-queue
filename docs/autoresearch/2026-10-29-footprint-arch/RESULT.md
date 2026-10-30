# Result: a flip-equivariant airbench at half B0's parameters beats it by 0.85 pp

Session `2026-10-29-footprint-arch`, closed 2026-10-30. Plan: `PLAN.md` (§9 is the pivot). Evidence: `ledger.tsv`.

## The answer to the session's original goal

The finalist is a depth-3 airbench at widths 128/112/144, with flip-equivariant convolutions in its last two groups
(method MT1). It is trained 40 epochs on the 45,000 training images with a recipe found by search, plus light
distillation from one airbench96 (weight 0.12). Against B0 (the project's ResNet-9 champion, 3.71 M parameters), with
5 seeds each on the RTX 4060:

| | Finalist, 6 TTA views | Finalist, 1 view | B0 (2 views) |
|---|---|---|---|
| Test accuracy (5 seeds) | **95.45 ± 0.17 %** | **95.00 ± 0.08 %** | 94.60 ± 0.11 % |
| Δ test vs B0 (one-sided 95 % lower bound; Welch p) | +0.85 pp (+0.68; 2e-5) | +0.40 pp (+0.28; 1.5e-4) | |
| Validation accuracy | 95.60 ± 0.27 % | 95.11 ± 0.20 % | 94.68 ± 0.15 % |
| Paired bootstrap, test images, one model each | +0.73 pp [+0.33, +1.13] | | |
| CIFAR-10.1 v6 (one look; paired CI vs B0) | 89.10 % (+1.10 [−0.10, +2.30]) | 88.35 % (+0.35 [−0.95, +1.60]) | 88.00 % |
| Stored parameters | 1,944,024 | 1,944,024 | 3,710,611 |
| Bytes on disk | 7.8 MB | 7.8 MB | 14.9 MB |
| MACs per image | 559 M | **93 M** | 450 M |
| GPU p50, batch 1 / throughput, batch 1024 | 10.0 ms / 5.0k img/s | **1.89 ms / 32.3k img/s** | 2.46 ms / 13.9k img/s |
| CPU p50, 1 thread, batch 1 | 89.7 ms | **13.4 ms** | 13.8 ms |
| CPU resident memory, batch 1 | 28 MB | 28 MB | 43-48 MB |
| GPU peak memory, batch 1024 | 588 MB | 571 MB | 552 MB |
| Footprint class (§2.2) | P | **S** | |

Both variants beat B0 by the session's win rule: ≥ 5 seeds, Δ ≥ 0.20 pp, Welch p < 0.05, the bootstrap interval
above zero, and the same sign on validation. The single-view variant also passes the strict class: no larger than B0
on any footprint axis and faster than B0 on every runtime row. The six-view variant buys a further 0.45 pp with 6× the
MACs. Its GPU peak at batch 1024 is 6 % over B0's, so it is in the parameter class only.

On CIFAR-10.1, all three models lose about 6.5 pp against CIFAR-10 test, and they keep the same order. With 2,000
images, the intervals include zero, as expected.

## Target A (96.0 % at ≤ 2 M parameters and ≤ 40 epochs): not met

The finalist is 0.55 pp short on test and 0.4 pp short on validation. The plan's stop rule (§9) ended the attempt:
the levers screened after the recipe search added +0.23 pp of validation over the 95.40 % baseline, against the rule's
0.3 pp. For scale, airbench96 itself (9.9 M parameters, 37 epochs) reaches 95.87 % on the same 45,000 images.

What was tried, each with 3 seeds unless noted (the full record is in the ledger):

- **Recipe search on the G4** (mode 2, ~420 trials, two phases). The airbench family saturated at 95.6 % single-seed
  validation, which was 95.2-95.4 % over 3 fresh seeds: the search's best was 0.2-0.5 pp lucky.
- **PyramidNet search** (23 trials): no better than its starting point (95.28 %).
- **M2 screens**:
  - DenseNet-BC, ConvMixer, WRN, ResNet-9 and Shake-Shake all came in below airbench.
  - ConvMixer at patch 1 was too slow for the 4060.
- **Distillation**:
  - From one airbench96: +0.1-0.3 pp, kept.
  - From the 3-model ensemble (96.30 % validation): no gain at 40 epochs (−0.1 to −0.3 pp).
- **MT1 flip-equivariant convolutions**: +0.32 pp validation and +0.16 pp test at matched parameters, and +0.58 pp
  test without TTA (the flip is built in). This is the session's one method that added accuracy, and it is the
  reason the single-view variant is in class S.
- **MT2 trunk-and-tails averaging** at equal charged epochs: −0.56 pp. The shorter trunk costs more than the averaging
  gains.
- **TTA with 18 views** instead of 6, on the same networks: no gain (−0.04 to −0.25 pp test).

Goal B (Shake-Shake + AutoAugment, a multi-day local run) was not started.

## Cost

- **Colab:** 22.1 compute units of the 50-unit cap, measured as the fall in the balance (884.46 to 862.40). This
  covered three G4 studies plus about 0.5 units on two failed launches.
- **G4 rate:** the G4 burned 10.05 units an hour against the 8.90 that solve-queue had recorded. solve-queue now
  samples the rate (`solveq colab rates`).
- **Local:** everything else ran on the RTX 4060.
- **Teacher:** one airbench96 trained for 37 epochs at 9.9 M parameters (E1.1). Its forward passes during
  distillation are not charged to the student, per §2.1.

## Reproduce

```bash
# the finalist over 5 seeds (saved models), then the comparison with B0 and the footprint table
python pipeline/tools/seeds.py M4-fe2-t158 --params results/autoresearch/P2/fe2-t158.json --seeds 1 2 3 4 5 --save --targets local --gpu-mb 7000
python pipeline/tools/compare_b0.py results/autoresearch/M4-fe2-t158/<time>.json
python pipeline/tools/footprint_bench.py --ref results/models/B0 results/jobs/train/<job>/model
```

The configuration in `results/autoresearch/P2/fe2-t158.json` has two parts:

- **Architecture:** depth 3, widths 128/112/144, `flip_eq` 2.
- **Recipe:** 39.9 epochs, batch 1024, lr 9.37, momentum 0.78, weight decay 0.0073, label smoothing 0.17, translate 4,
  cutout 12, TTA level 2. Distillation from `results/jobs/train/00023-train/model` at weight 0.12 and temperature 5.5.

The single-view variant is the same network with `tta_level` 0 in its `config.json`. Saved models:
`results/jobs/train/00075-00079-train/model`; seed 5 (`00079`) is the median-validation one that was benched and looked at.

## Infrastructure that came out of the session

- **cifar-queue:**
  - `cifarq.airbench` gained precision `amp`, ensemble distillation, TTA level 3 and MT1-MT6.
  - New pieces: the PyramidNet, DenseNet, ConvMixer, Shake-Shake and WRN builders, the family search spaces with
    warm starts from an earlier study, a teacher shipped to VMs, `fit_rungs.py`, `watch_session.py` and
    `compare_b0.py`.
- **solve-queue:**
  - Branch `colab-rates`: the sampled burn rate per card and region.
  - Branch `gpu-reservations`: local GPU reservations, served in waiting order, with the stall check skipping jobs
    that are waiting. These stop two 7000 MB trainings from being admitted together on the 8 GB card. Neither branch is
    merged into main.
