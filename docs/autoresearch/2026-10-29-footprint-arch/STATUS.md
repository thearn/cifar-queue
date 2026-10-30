# Status: 2026-10-29-footprint-arch

Updated 2026-10-29 21:10. Phase: **M0 closed; M1 infrastructure built and tested on the CPU; waiting for the GPUs.**
Goals (PLAN.md §1): Goal A, 96 % at ≤ 2 M stored parameters in ≤ 40 epochs (replicate airbench96 first); Goal B
(stretch), beat Shake-Shake 26 2x32d + AutoAugment (97.5 %) at ≤ 2.9 M; the method track MT1-MT7 alongside.

## Resources

The local 4060 freed at ~21:20 (af2's `ll-cal/run.py` ended); af2's G4 VM `af2-g4` still holds one of the
account's two VM slots.

## Built (CPU-tested: `pytest` 55 pass, 2 skipped as before; 36 of them new; B0's paths unchanged)

| Piece | Where | Checked |
|---|---|---|
| airbench port, every choice a hyperparameter, `AIRBENCH94`/`AIRBENCH96` presets, 50k mode, distillation from a saved teacher, AdamW, policy augmentation, a parameter cap (over it: pruned at step 0) | `src/cifarq/airbench.py` | contract tests; airbench94 1,972,792 params (paper 1.97 M), airbench96 9,918,776 |
| families on the same trainer: ConvMixer, ResNet-9, PyramidNet, DenseNet-BC, Shake-Shake, WRN | `src/cifarq/nets.py` | each trains, saves, reloads and predicts what training measured; counts match the papers (`literature.md`, M0 §3) |
| TrivialAugment (wide) and RandAugment on the GPU, per image | `src/cifarq/augment.py` | range and shape per operation; visual check of all 14 operations |
| footprint and runtime bench, classes S/P/out | `src/cifarq/footprint.py`, `pipeline/tools/footprint_bench.py` | static counts equal the trainer's; fresh-process measurement; negative controls (larger model, training-set-storing model → out) |
| CIFAR-10.1 v6 hold-out stage, every look logged | `src/cifarq/data/holdout.py`, `data/manifests/holdout.json` | fetched at d9982ab, hashes pinned; 2000 images, 200 a class; not yet looked at |
| search spaces for the families (airbench, resnet9, convmixer) with warm points | `src/cifarq/families.py` | warm points under the cap; airbench space 66 % over the cap at random (pruned free), convmixer 6 %, resnet9 0 % |
| mode 2 driver generalised (`--family`, `--cap`, `--max-epochs`, `--kd`, `--resume-from`; per-family slot counts) | `src/cifarq/tune.py` (`on_vm`), `pipeline/55_tune_on_vm.py` | `b0` path unchanged (tests); family path not yet run on a GPU |
| seeds runner with a source snapshot per experiment (`refs/autoresearch/<id>`) | `pipeline/tools/seeds.py` | end to end through `direct` on the CPU |
| method track, each a hyperparameter off by default: MT1 `flip_eq` (flip-equivariant groups, pair-pooled), MT2 `tails`/`tail_epochs` (trunk-and-tails averaging, every tail's epochs charged), MT3 `pca_init`, MT4 `kd_weighting` (`disagree`, `hard`), MT5 `tie` (weight-tied depth), MT6 `tta_gate` (confidence-gated TTA, average views reported) | `src/cifarq/methods.py`, `airbench.model`/`train` | each trains, saves and reloads; MT1 exact pair-swap equivariance and pooled invariance on even maps; MT5 adds only batch-norm parameters; MT2 charges 3 + 1 epochs for 2 tails of 1 |

## Deviations from the plan (recorded, by the agent's judgement)

- **Branches and commits** (the user, 2026-10-29): the restructure and the session's start committed on `layout`
  (5e3440a, 4eb77ee); the session continues on `fparch/main` with a branch per code-changing experiment
  (PLAN.md §3). Earlier: no commits, provenance by `git stash create` refs only.
- **No account-wide `solveq budget 50`.** The limit is account-wide and af2 had already spent 114 units that day; a
  50-unit limit would stop af2's work. The session's spend is tracked per executor (`solveq colab hours`) from these
  starting values: cifar-g4 17.9, cifar-a100 8.8, cifar-a100x 1.2 units (all from earlier sessions). New executors
  added for the session start at 0.
- **The account holds two GPU VMs**: while af2 keeps one, this session has one VM slot, not two (§4's two-family
  pairing halves until af2's VM is released).
- **airbench96 is 128/384/512, 37 epochs, 9.9 M** (the source file), not the 128/512/512 at 40 the plan first said;
  the plan and literature are corrected.
- **A10 (FastBoost) dropped:** below 96 %, no code; see `literature.md`.

## Baselines

- **The bench is frozen** (E0.1, revision a3, merge 1b6540c). B0's row: 3,710,611 stored params (3,709,882 trainable),
  14.86 MB, 450 M MACs an image (flip TTA); GPU bf16 p50 2.51 / 20.2 / 82.3 ms at batch 1 / 256 / 1024, peak GPU
  17.8 / 149.4 / 552.3 MB; CPU p50 15.2 ms (1 thread, b1), 7.7 ms (all, b1); resident 43 MB at b1.
- **B0** (local footprint reference): `results/g4-best.json`; 3,709,882 trainable params. On the 4060 (E0.3, 5 seeds):
  **94.60 ± 0.11 % test**, 94.68 ± 0.15 % validation (G4, 3 seeds: 94.57 %). Saved network (seed 4, median
  validation): `results/models/B0`.
- **airbench96**: 96.05 % (paper) / 96.03 % (script header, 400 runs) at 37 epochs, 9.92 M. E1.1.
- **Shake-Shake 26 2x32d**: 96.45 %; 97.5 ± 0.1 % with AutoAugment (verified). E6.1.

## Exposure of the evaluation data

- Validation (5,000): used by every search since 2026-10-27. The selection set; not trusted at face value.
- Test (10,000): reported for every run since 2026-10-27; never used for selection. Regression evidence.
- CIFAR-10.1 v6: fetched, **not looked at** (`data/holdout/looks.jsonl` absent).

## Budget

- Colab cap 50 units for the session (tracked per executor, see above); spent 0 before M3. Local unlimited.
- 2026-10-30 ~03:45, `colab usage`: balance 884.46 units, burn 10.05 units/h with one VM up (cifar-g4, ours; no other
  session's VM live). The queue's card record says 8.90 for a G4, so `solveq colab hours` understates G4 spend by
  ~13 %. Session spend is tracked as the balance's fall from 884.46 (no other VM up meanwhile), the hours table a check.

## Jobs

PLAN §9 pivot (2026-10-30): stronger teachers (P1), method screens at matched parameters (P2), one combined study.

- **Baseline (airbench top 5, seeds 1-3, done but p2-t30):** best p2-t158 (128/160/192, 1.91 M, kd from one
  airbench96 at w 0.12): 95.40 % val, 95.41 ± 0.11 % test. p1-t157 95.35 / 95.25; p1-t236 95.25 / 95.10; p1-t84 (the
  search's best) 95.17 / 95.06. Winner's curse 0.2-0.5 pp. Under the 95.5 % readout: the pivot stands.
- **P1 teacher:** the ensemble of the three airbench96 45k models reaches 96.30 % val (singles 95.78-96.08). Screens
  queued after MT1: ens3 at p2-t158's kd settings and at w 0.5 T 4, seeds 1-3 (`scratchpad/p1.sh`).
- **P2 MT1** flip-equivariance at p1-t84's recipe and matched parameters (fe1 128/128/144 1.68 M; fe2 128/96/144
  1.76 M), seeds 1-3, running (`scratchpad/p2.sh`); compared with p1-t84's retrain.
- **M3-pyramidnet**, G4 50 min (job via `scratchpad/m3p.log`): 16 trials at 06:00, best the warm point 95.28 %.
- Session spend 12.0 units before the PyramidNet study.
- Watcher over all of them.

## Standing instructions

- Push notification to the user when Target A or B is Validated (PLAN.md §7).
- Whenever the session has queue jobs, `pipeline/tools/watch_session.py fparch` runs in the background (stall 15 min,
  idle 20 min), so a failure, a stall or a dry queue wakes the agent without the user asking (the user, 2026-10-29,
  after two jobs thrashed the 4060's memory for an hour unnoticed). Heavy jobs on the 4060 go with `--gpu-mb 7000`.

## Next action, in order

1. E0.1's GPU rows and repeatability on a quiet machine, with E0.3 (B0 × 5 seeds, local) first since the bench
   needs B0's saved network.
2. E1.2 (airbench94: 2 × 50k + 3 × 45k, local, ~1 min each), then E1.1 (airbench96: 1 × 50k + 3 × 45k, ~5 min each).
3. M2 screens on the 4060 (A1-A9 at 40 epochs) while the first G4 calibration + airbench-family search runs.
4. Method screens (MT1-MT6 on A1 at 20 epochs, 3 paired seeds, local) once E1.2 fixes the A1 recipe. MT7 (the
   MAC-capped joint search) needs a cap in the airbench space: not built yet.
5. R1 teachers: shipping a teacher directory to a VM job as data (the `--kd` path) is untested; test it with the first
   distillation search.
