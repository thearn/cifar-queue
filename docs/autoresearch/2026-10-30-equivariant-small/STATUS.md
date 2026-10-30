# Status: 2026-10-30-equivariant-small

Updated 2026-10-30. Phase: **M2 done: gate hit (no budget reaches 95.7 % at one view). Waiting for the user's target revision.**

## Objective

≥ 96.0 % CIFAR-10 test, ≤ 2 M stored parameters, one forward pass per image (no TTA), ≤ E* epochs on the 45k split (E* from M2, 80-150)
(`PLAN.md` §1), with the equivariance-vs-TTA measurement (H1) as the method result. Start point: the last session's
finalist at 95.00 ± 0.08 % (one view) / 95.45 ± 0.17 % (6 views), 1.94 M, 40 epochs.

## Baselines carried over (not recomputed)

- B0: 94.60 ± 0.11 % test, 94.68 ± 0.15 % val (E0.3, 5 seeds); footprint row in `results/footprint/20261030-085535.json`.
- airbench96: 95.87 ± 0.04 % test on 45k (3 seeds, saved in `results/jobs/train/00023-00025-train/model`), 96.01 % on 50k.
- Finalist fe2-t158: 5 saved seeds in `results/jobs/train/00075-00079-train/model`; configuration
  `results/autoresearch/P2/fe2-t158.json`.

## Exposure of the evaluation data

- Test: reported for every run of the last session (regression evidence, not a hold-out). **Sealed from this
  session's first run** (`test='sealed'`, M1); opened once per final model in M6. Looks taken this session: none.
- CIFAR-10.1: three looks on 2026-10-30 (finalist 6 views, 1 view, B0), `data/holdout/looks.jsonl`. This session: none.

## Budget

- Colab: 75 units (the user, 2026-10-30); spent 0. Balance at the last reading 862.40 (2026-10-30 ~06:25). Local unlimited.

## Jobs

None on the G4 (M2 done, 3.9 units; balance 858.46). M1 part 2 committed on `eqsm/M1` (fdc8cdf, 971a96b, ab172e5), not
yet merged. M3 drafted (`experiments/M3-h1-curves/protocol.md`).

## M2 readout (val, 2 seeds; finalist D1 D1 1.94 M / plain twin 1.91 M)

| Epochs | D1 1 view | D1 6 views | plain 1 view | plain 6 views |
|---|---|---|---|---|
| 40 | 95.18 | 95.59 | 94.29 | 95.22 |
| 80 | 95.58 | 95.93 | 94.67 | 95.57 |
| 100 | 95.56 | 96.08 | 95.02 | 95.74 |
| 120 | 95.59 | 96.01 | 95.24 | 96.14 |
| 150 | 95.47 | 96.02 | 95.13 | 96.04 |

- One view plateaus at ~95.6 % from 80 epochs: more epochs alone will not reach 96 % at one view. The PLAN §1 gate
  (95.7 % by 150) fails: the target needs the user's revision before M3.
- Six views reach 96.0-96.1 % at ≥ 100 epochs for both arms.
- H1's predicted shape appears already: equivariance's one-view lead over plain shrinks from +0.89 pp (40) to
  +0.34 pp (150); at six views it leads at 40 (+0.37) and ties from 120.

## Findings so far

- M0: Weiler & Cesa Table 5 verified (D1 D1 D1 took 0.5 of the 0.6 pp gain at long schedules); the airbench
  successor (hiverge/cifar10-speedrun) holds the 94 % record at 1.98 s and gates TTA on confidence (our MT6). Weiler
  & Cesa's CIFAR parameter counts and epochs: not in the fetched text (framing only; left).
- **Correction to the last session:** `count_macs` (a3) skipped FlipConv; the finalist is 237 M MACs a view, not 93 M
  (fixed in a4; RESULT.md corrected, 7160285).
- Regression gate: airbench training is not bit-reproducible on the 4060 (cudnn.benchmark); gate is statistical.

## Standing instructions

- `SOLVEQ_SESSION=eqsm` on every launch; `pipeline/tools/watch_session.py eqsm` running whenever jobs exist; `--gpu-mb
  7000` on heavy 4060 jobs; absolute paths for data and teacher directories; a changed argument on any relaunch.
- Push notification to the user when the target is Validated (M6).
- No G4 job before M2's readout is agreed and the budget confirmed.

## Next action, in order

1. (Done) Budget 75 units; epoch cap set by M2 within 80-150.
2. M2 readout: E* (PLAN §1).
3. M1 part 2 while M2 runs; M3 protocol (D1/D4 patterns, BlurPool, matched params and MACs) frozen at M2's readout.
