# Status: 2026-10-30-equivariant-small

Updated 2026-10-30. Phase: **plan agreed (75 units, E* ≤ 150); M0 and M1 next.**

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

None.

## Standing instructions

- `SOLVEQ_SESSION=eqsm` on every launch; `pipeline/tools/watch_session.py eqsm` running whenever jobs exist; `--gpu-mb
  7000` on heavy 4060 jobs; absolute paths for data and teacher directories; a changed argument on any relaunch.
- Push notification to the user when the target is Validated (M6).
- No G4 job before M2's readout is agreed and the budget confirmed.

## Next action, in order

1. (Done) Budget 75 units; epoch cap set by M2 within 80-150.
2. M0: the four open items in `literature.md` §6.
3. M1: `test='sealed'`, the two-seed objective, the p4m layer, BlurPool, the MAC-count test; the regression gate.
4. M2 on the G4: the epoch curve (`experiments/M2-epoch-curve/protocol.md`, to freeze at launch).
