# M6-confirm: retrain the candidates, pick the finalist, confirm over 5 seeds, open the test set once
Frozen: 2026-10-30, commit 003f84e (eqsm/main). Dataset dataset-6e01ae0c4036.

- **Stage 1 (retrain, G4):** the search's three best distinct configurations (trials 14, 33, 39: flips in groups 2-3,
  SRF, wide first group, batch 512), M4's SRF variant (160/136/176) and the finalist, each over seeds 1-3 at 100
  epochs, models saved, validation at 1/2/6 views, test sealed (grid `results/autoresearch/eqsm/m6-grid.json`).
  Seeds 1-3 differ from the search's 0-1, so the search's selection does not bias these numbers.
- **Selection rule (validation only):** the configuration with the best 3-seed mean at **one view** is the one-view
  finalist; the best at **six views** is the six-view finalist (often the same network).
- **Stage 2 (confirm):** seeds 4 and 5 added for each finalist (5 seeds 1-5, same card type: all on the G4), saved.
- **Stage 3 (test opened, once per final model):** `compare_b0.py`-style statistics on the 5 saved models: test at
  1 view and 6 views from the same networks (one look), mean, sd, one-sided 95 % lower bound; the footprint bench
  on the median-validation model locally (4060); CIFAR-10.1 once.
- **Targets judged (PLAN §1):** the one-view target (mean ≥ 96.00 %, lower bound ≥ 95.80 %, val ≥ 96.0 %); if it
  fails, the six-view fallback with the same thresholds (the user's decision after M2). Push notification if either
  is Validated.
- **Compute:** stage 1, 15 trainings (~25 min, ~4 units); stage 2, up to 4 trainings (~1.5 units). Spend before:
  ≈ 43 units (per-VM accounting).
