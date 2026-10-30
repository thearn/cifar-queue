# M3-airbench: the airbench family's mode 2 study under the 2 M cap (frozen 2026-10-30)

**Hypothesis.** The airbench family (depth 2 or 3, free widths) with its full recipe tuned at up to 40 epochs reaches
≥ 96.0 % validation under 2,000,000 stored parameters; M2's best untuned point is 95.48 % (depth 3, 64/192/192).

**Decision.** Whether airbench is the Goal A finalist family and which top-5 configurations go to the 3-seed retrains
on the 4060; whether the readout rule (best 3-seed val mean < 95.5 %) fires.

**Candidate.** `cifarq.families.airbench_space` at the branch `fparch/M3-airbench` commit: depth {2, 3}, w1/w2/w3
categoricals, the shared recipe (epochs 20-40, batch, lr 2-30 log, momentum, wd, bias scaler, label smoothing,
schedule, bn momentum, translate, cutout, policy augmentation, whiten bias epochs, logit scale, TTA level, lookahead);
over the cap a trial is pruned at step 0. Warm starts: M2's two depth-3 points and depth 2 64/256/256, each at the
airbench96 recipe and 40 epochs. No distillation in this study (teacher shipping to a VM untested): a deviation from
§M3, to be added as a resumed study once tested.

**Evaluator.** Fixed 5000-image validation (rng 1234), val_acc with the trial's TTA level; test recorded, never used
to select. Sampler: multivariate TPE, constant liar, 16 startup trials, seed 0.

**Phases.**
1. Report-only: `--pruner rungs --pruner-settings '{"rungs":[10],"keep":1.0,"min_trials":1}'`, 40 minutes on
   `colab:cifar-g4`, concurrency calibrated in the job (1, 2, 3, 4, 6) and stored as `cifar-airbench`.
2. `pipeline/tools/fit_rungs.py` on the study (≥ 20 complete curves), then resume with the fitted rungs.
3. Stop at saturation (best val gained < 0.10 pp over the last 40 % of trials and top 10 within 0.3 pp) or 12 units.

**Gates.** GPU busy ≥ 90 % over the job (under 80 %: stop, diagnose, resume). Units tracked from `solveq colab hours`
(cifar-g4 at 17.9 before this study).
