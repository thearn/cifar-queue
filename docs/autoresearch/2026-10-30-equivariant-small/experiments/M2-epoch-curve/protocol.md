# M2-epoch-curve: where a 2 M equivariant airbench crosses 96 % as epochs grow
DRAFT: to freeze at launch with the commit of `eqsm/main` after M1's `test='sealed'` and the matched-parameter twin.
Dataset dataset-6e01ae0c4036.

- **Hypothesis and mechanism:** the last session's gap (95.00 % one view, 95.45 % six views at 40 epochs) is
  epoch-limited; validation at one view rises monotonically with epochs under the airbench schedule (triangular
  lr, Lookahead) and reaches 95.7-96.0 % somewhere in 80-150 epochs.
- **Failure or uncertainty addressed:** the epoch cap E* in PLAN §1 is not yet known; the target could be unreachable
  at 2 M at any short schedule.
- **Decision it informs:** the session's epoch cap E* (PLAN §1: the smallest of 80/100/120/150 whose one-view
  validation mean reaches 95.7 %; if 150 does not, the target is revised with the user); whether
  equivariance's one-view advantage over plain + TTA persists at longer schedules (a first look at H1 before M3).
- **Baseline / candidate; the variable changed:** candidate = fe2-t158 (D1 D1, 128/112/144, 1,944,024 parameters,
  recipe `results/autoresearch/P2/fe2-t158.json` with its one-teacher KD); baseline = the plain depth-3 airbench at
  128/160/192 (1,912,824; p2-t158's recipe, the matched-parameter twin). Variable: `epochs` ∈ {40, 80, 100, 120,
  150}, the warm-up and final-lr fractions unchanged (the schedule scales).
- **Matched baselines:** parameters matched (above). MACs: the twin at 6 views is 6 × 93 M vs the candidate's
  1 × 93 M at one view; both arms are reported at 1 and 6 views, so the matched-MAC pairs are (candidate 6 views,
  twin 6 views) and (candidate 1 view, twin 1 view), with (candidate 1 view, twin 2 views) the equal-cost pair
  nearest to inference parity.
- **Fixed inputs:** 45k/5k split; seeds 0 and 1; `test='sealed'`; teacher `results/jobs/train/00023-train/model`
  (absolute path, sent as data); charged epochs = `epochs` (no filtering).
- **Measurements and gates:** val at `tta_level` 0 and 2 per run (the evaluator computes both, `val_acc_notta`);
  mean over 2 seeds per grid point; the curve plotted and its crossing read. Tolerance: 2-seed sd ≈ 0.1-0.2 pp, so
  a crossing is called only where the mean is ≥ 96.0 % and the lower seed ≥ 95.8 %.
- **Compute:** `colab:cifar-g4`, one training at a time (stored calibration), 20 runs × 20-80 s ≈ 15-20 min; cap
  4 units; `--stall-min 15`; balance before: recorded at launch.
- **GPU-use check:** ≥ 90 % busy as the last session's airbench jobs; under 80 %: stop and diagnose.
- **Artifacts and recovery:** one `seeds.py`-style job per grid point and arm, or a single `tune.on_vm` grid study;
  results under `results/autoresearch/M2-epoch-curve/`; the ledger gets one row per arm with the crossing noted.
