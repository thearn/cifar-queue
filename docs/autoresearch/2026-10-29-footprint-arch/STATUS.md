# Status: 2026-10-29-footprint-arch

Updated 2026-10-29 21:10. Phase: **M0 closed; M1 infrastructure built and tested on the CPU; waiting for the GPUs.**
Goals (PLAN.md §1): Goal A, 96 % at ≤ 2 M stored parameters in ≤ 40 epochs (replicate airbench96 first); Goal B
(stretch), beat Shake-Shake 26 2x32d + AutoAugment (97.5 %) at ≤ 2.9 M; the method track MT1-MT7 alongside.

## Blocked on

Both GPUs busy with the af2 project at the session's start (the local 4060 by a process outside the queue,
`ll-cal/run.py`; a G4 VM `af2-g4`). The user will say when they are free. Nothing of this session is running.

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

- **No branch, no commits.** The working tree carries the user's uncommitted `layout` restructure, which the campaign
  code depends on; committing it on a new branch would move it out of their working tree when they switch back.
  Provenance instead: `git stash create` snapshots kept as `refs/autoresearch/<experiment>` (exact code, nothing
  stashed or changed), and the results records' commit + dirty flag.
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

- **B0** (local footprint reference): `results/g4-best.json`; 3,709,882 trainable params; 94.57 % test (3 seeds, G4).
  E0.3 re-measures on the 4060.
- **airbench96**: 96.05 % (paper) / 96.03 % (script header, 400 runs) at 37 epochs, 9.92 M. E1.1.
- **Shake-Shake 26 2x32d**: 96.45 %; 97.5 ± 0.1 % with AutoAugment (verified). E6.1.

## Exposure of the evaluation data

- Validation (5,000): used by every search since 2026-10-27. The selection set; not trusted at face value.
- Test (10,000): reported for every run since 2026-10-27; never used for selection. Regression evidence.
- CIFAR-10.1 v6: fetched, **not looked at** (`data/holdout/looks.jsonl` absent).

## Budget

- Colab cap 50 units for the session (tracked per executor, see above); spent 0. Local unlimited.

## Jobs

None. `SOLVEQ_SESSION=fparch` for the session's jobs.

## Next action (when the GPUs are free), in order

1. E0.1's GPU rows and repeatability on a quiet machine, with E0.3 (B0 × 5 seeds, local) first since the bench
   needs B0's saved network.
2. E1.2 (airbench94: 2 × 50k + 3 × 45k, local, ~1 min each), then E1.1 (airbench96: 1 × 50k + 3 × 45k, ~5 min each).
3. M2 screens on the 4060 (A1-A9 at 40 epochs) while the first G4 calibration + airbench-family search runs.
4. Method screens (MT1-MT6 on A1 at 20 epochs, 3 paired seeds, local) once E1.2 fixes the A1 recipe. MT7 (the
   MAC-capped joint search) needs a cap in the airbench space: not built yet.
5. R1 teachers: shipping a teacher directory to a VM job as data (the `--kd` path) is untested; test it with the first
   distillation search.
