# Status: 2026-10-30-equivariant-small

Updated 2026-10-30. Phase: **closed 2026-10-30.** The user confirmed completion; merged into `master` (d038811) and pushed; the session branches
(`eqsm/*`, and the previous session's `fparch/*`) deleted, all merged, so their commits stay reachable from `master`. The
53 `refs/autoresearch/*` snapshot refs (exact trees the result files cite) are kept locally.

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
- Test looks this session: one per finalist, 15:08 (T39) and 15:12 (FIN), `data/holdout/test_looks.jsonl`.
- CIFAR-10.1: three looks on 2026-10-30 (finalist 6 views, 1 view, B0), `data/holdout/looks.jsonl`. This session: none.

## Budget

- Colab: 75 units (the user, 2026-10-30); spent 0. Balance at the last reading 862.40 (2026-10-30 ~06:25). Local unlimited.

## Jobs

- None running. **Session spend ≈ 52 units** of 75 (M6 ~9.5 by VM hours on top of the ≈ 43 below): 25.2 to 12:25 by the balance (only our VM up), then ~17.6 for M5 by our
  VM's hours at 10.05/h (a second session's G4, `snapclip-g4`, has been up since ~12:25, so the balance no longer measures us).

## Run-to-run noise (M6)

The same configuration and seeds retrained give different results: M4's SRF (seeds 1-3) 95.73 / 96.09 % at 1 / 6 views,
M6's 95.45 / 95.80 %; one seed moved 0.54 pp. Airbench training is nondeterministic on the GPU (cudnn.benchmark), so a
seed does not pin a run; per-run noise is ~0.2 pp and 3-seed means move by up to ~0.3 pp. Screens' small effects
(|Δ| < 0.2 pp) are inside it.

## M4 readout (val at 1 / 6 views, 3 seeds, paired with M3's D1-e100: 95.62 / 96.02)

| Lever | 1 view | Δ (paired) | 6 views | Outcome |
|---|---|---|---|---|
| SRF (160/136/176, 1.94 M, 353 M MACs) | 95.73 | +0.11 (+0.22/+0.12/0.00) | 96.09 | Inconclusive; crosses 95.7 |
| KD weighted by disagreement | 95.55 | −0.07 | 95.98 | Null |
| Weight-tied depth | 95.48 | −0.14 | 95.92 | Null |
| SE (field-shared) | 95.39 | −0.23 | 95.89 | Rejected |
| Proxy filtering (keep 0.5) | 95.19 | −0.43 | 95.78 | Rejected |
| Muon (untuned) | 94.93 | −0.69 | 95.53 | Rejected |

M3's repaired control: flips beat a plain net with the same MACs a view (2.60 M parameters) by +0.39 / +0.40 / +0.32 pp at
one view (20 / 40 / 100 epochs).

## M3 readout (val, 3 seeds 1-3; 1 / 2 / 6 views)

| Arm | 20 epochs | 40 epochs | 100 epochs |
|---|---|---|---|
| P plain 1.91 M, 177 M MACs | 93.41 / 94.24 / 94.27 | 94.50 / 95.11 / 95.31 | 95.27 / 95.68 / 95.89 |
| **D1 flips g2-3, 1.94 M, 237 M** | **94.35** / 94.64 / 94.72 | **95.27** / 95.54 / 95.69 | **95.62** / 95.79 / **96.02** |
| D1B + BlurPool | 94.17 / 94.45 / 94.55 | 94.99 / 95.29 / 95.39 | 95.44 / 95.63 / 95.79 |
| D4a flips + quarter turns g3, 260 M | 93.53 / 94.11 / 94.29 | 94.55 / 95.03 / 95.25 | 94.99 / 95.39 / 95.49 |
| D4b flips + quarter turns g2-3, 597 M | 93.48 / 93.91 / 94.09 | 94.83 / 95.20 / 95.38 | 95.23 / 95.51 / 95.75 |

- H1: D1 at 1 view vs plain at 2 views (354 M MACs): +0.11, +0.16, −0.06 pp at 20, 40, 100 epochs. Equivariance buys
  the flip view's accuracy at a third fewer MACs, with a margin at short schedules that closes by 100 epochs, the
  shape PLAN §1 predicted and the tie Adly 2026 predicts at equal test-time compute.
- Rotations hurt (a misaligned group on CIFAR); BlurPool hurts. The matched-MAC control (PW) was invalid (pruned by
  the inherited 2 M cap) and is repaired in the M4 grid.

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

None. Open options for a later session are in `RESULT.md`.
