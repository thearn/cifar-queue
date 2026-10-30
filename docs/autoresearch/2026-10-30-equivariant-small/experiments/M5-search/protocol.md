# M5-search: the flip-equivariant airbench family, searched on two seeds at one view
Frozen: 2026-10-30, commit dd54ee8 (eqsm/main). Dataset dataset-6e01ae0c4036. Balance before: 838.95 units.

- **Hypothesis:** the finalist's recipe was tuned for the plain 40-epoch network with six views; tuning widths, the
  equivariance pattern, SRF and the recipe for one view at 100 epochs lifts the two-seed one-view validation to
  ≥ 95.9 % (a level whose 5-seed confirmation can plausibly reach 96.0 % test, given the finalist's val ≈ test gap
  of ~0.1-0.15 pp last session).
- **Space:** `families.eqairbench_space`: flip_eq {1, 2}; srf {off, on}; w1 96-176, w2 96-192, w3 112-224 (steps of 8);
  batch {512, 1024}; lr 4-14 (log); momentum 0.7-0.9; weight decay 0.004-0.015 (log); bias scaler 24-96; label
  smoothing 0.1-0.3; warm-up 0.05-0.25; final lr 0-0.08; bn momentum 0.035-0.12; translate {2,3,4}; cutout {8,12,16};
  logit scale 1/25-1/8; distillation weight 0-0.4 and temperature 2-8 from airbench96 45k seed 0. Epochs fixed at
  E* = 100; parameter cap 2,000,000 (over: pruned at step 0).
- **Objective:** `families.train_two_seeds` (PLAN §2.1): seed 0's per-epoch curve to the pruner, survivors again at
  seed 1, the value the mean one-view validation (`tta_level` 0); test sealed.
- **Sampler and pruner:** multivariate TPE with constant liar, seed 0, 8 startup trials (the budget allows ~30-35
  trials); warm starts: the finalist (128/112/144) and its SRF variant (160/136/176). Rungs [50] keep 2/3 (fitted on
  the last session's airbench study; the rank correlation there was weak before 70 %, so the cut is light).
- **Compute:** `colab:cifar-g4`, 2 trials at once (the grids ran 2 per VM at 100 % busy), 100 minutes (~17 units of
  the ≤ 20 planned). `55_tune_on_vm.py --family eqairbench --seeds 2 --max-epochs 100 --kd ... --concurrent 2`.
- **Saturation / stop:** the minutes budget; a study still improving at the end may be resumed once from the
  reserve (≤ 8 units) if its best moved in the last third.
- **Next:** the top 5 by two-seed value retrained over seeds 1-3 at one view (M6), the best over 5 seeds.
