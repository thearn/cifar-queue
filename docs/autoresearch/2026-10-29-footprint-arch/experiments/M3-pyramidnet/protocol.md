# M3-pyramidnet: the PyramidNet family's mode 2 study under the 2 M cap (frozen 2026-10-30)

**Hypothesis.** PyramidNet (M2: depth 110, alpha 48, 1.77 M, 95.18 % val / 95.05 % test with the airbench96 recipe,
untuned, at batch 256) tuned at up to 40 epochs passes the airbench family's plateau (~95.6 % val single-seed, M3-airbench
phases 1 and 2) under 2,000,000 stored parameters.

**Decision.** Whether PyramidNet replaces airbench as the Goal A finalist family (top 5 to 3-seed retrains).

**Candidate.** `cifarq.families.pyramidnet_space` at the `fparch/M3-pyramidnet` commit: bottleneck {no, yes}, blocks a
stage 3-21 (depth 6n+2 or 9n+2), alpha {24 .. 160}; the shared recipe (epochs 20-40, batch 256-2048, lr 2-30 log, ...);
over the cap pruned at step 0. Warm start: M2's A4 point. No distillation (it gave nothing for airbench: phase 2).

**Evaluator.** As M3-airbench: fixed 5000-image validation, val_acc at the trial's TTA level; test recorded only.
Sampler multivariate TPE, constant liar, 16 startup, seed 0.

**Run.** `colab:cifar-g4`, report-only pruner (rungs [10] keep 1.0) for the family's own curves, concurrency calibrated
in the job (deep narrow nets are launch-bound: more at once expected), 50 minutes (~8.5 units at 10.05 an hour).

**Gates.** GPU busy >= 90 %. Session spend before: 12.0 units (balance 884.46 -> 872.46).
