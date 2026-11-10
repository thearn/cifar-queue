# The move to JAX: where it stands and what comes next

Started 2026-11-09. The project's training code moved from PyTorch to JAX (Flax NNX for the networks, Optax for the
optimisers), so that the ML examples share one engine with designloop's adjoint and geometry code, can train several
configurations at once under `vmap` when that pays, and can be differentiated through training. The ML template
(`templates/ml-runq-template`) and the `ml-project` skill moved with it.

## Done

- `src/cifarq/jx.py`: the shared pieces (PyTorch-matching initialisation, batch norm with PyTorch's momentum
  convention, frozen filters, the per-image affine resample, the optimisers with weight decay masked off biases and batch
  norm, the weight average, host checkpoints with typed PRNG keys, the persistent compilation cache).
- `model.py`, `train.py` (`train.Run`: an epoch is one jitted `lax.scan`; float hyperparameters, the step count
  included, are traced arguments, so trials that differ only in them share a compiled program), `evaluate.py`,
  `hub.py` (`ModelHubMixin`, safetensors), `pbt.py`, the app, CI, the model card, the champion stage
  (`train_seeds`: seeds one after another in one process, `--separate` for a job a seed).
- The airbench family: `airbench`, `methods` (Muon as an Optax transformation), `nets`, `augment`, `families`,
  `footprint`. Parameter and MAC counts equal the Torch records (airbench94 1,972,792; airbench96 9,918,776 and
  451,967,264 MACs a view; the flip-equivariant finalist 1,944,024 and 237,405,888). Numerical differences are listed in
  each module's docstring.
- Tests: the whole suite on the CPU (`JAX_PLATFORMS=cpu`), style checks included.

## Measured (RTX PRO 6000 Workstation Edition on Vast, held at 400 W by its host)

| | Torch (Colab G4, 2026-10-29) | JAX (2026-11-09) |
|---|---|---|
| Champion configuration, test over 3 seeds | 94.57 % (94.48-94.65) | **94.58 +- 0.13 %** |
| One 30-epoch training, steady | 22 s | about 18-20 s (0.61-0.66 s an epoch) |
| Compilation before the first epoch | none | 11 s cold, 4 s from the cache |
| Cold 45-minute search, seed 0: trials (pruned) | 155 (13) | 168 (33) |
| Seed 0: best validation (test) | 94.86 (94.60) | 94.90 (94.29) |
| Seed 0: first 92 % / 93 % | 2.4 / 3.5 min | 2.1 / 8.1 min |
| Seed 0: configurations at 92 % or more | 116 | 127 |
| Cold search, seed 1: trials (pruned) | 177 (21) | 124 (23) |
| Seed 1: best validation (test) | 95.12 (94.59) | 94.96 (94.61) |
| Seed 1: first 92 % / 93 % | 1.9 / 9.1 min | 6.2 / 8.5 min |
| Seed 1: configurations at 92 % or more | 124 | 90 |

Seed 1 trailed because its calibration chose 3 trainings at once (it stopped measuring at 4, where the gain from 3
was small) where seed 0's chose 6 (0.98 runs a second at 6 against 0.79 at 3): the calibration is noisy at the top,
so the next searches on this card pin `--concurrent 6`. Seed 1 also had 4 failed trials (seed 0: 3).

Two findings changed the code: seeds vmapped into one program ran slower a seed than alone (1.02 s an epoch against
0.61 s: XLA batches the seeds' convolutions less well), so `train_seeds` trains them in turn; and the schedule-free
batch-norm refresh, unrolled over 25 batches, compiled as long as a training epoch, so it is a scan (cold compilation
36.6 s -> 11.5 s).

## Decided (2026-11-10): JAX, not Torch

Full trainings on the local RTX 5070 Ti, the Hugging Face dataset (`6e01ae0c`), seeds 0-2, one fresh process each,
the arms interleaved: Torch at `f7ab6f5`, JAX at `2876eea`, and JAX with the training set passed to the compiled
programs as an argument (below). Warm call: seeds 1-2, the persistent compilation cache filled by seed 0.

| | Steady epoch, s (Torch / JAX / JAX+arg) | Warm call, s | Test % |
|---|---|---|---|
| Defaults, 10 epochs | 0.871 / 0.819 / 0.814 | 13.7 / 14.5 / 12.4 | 85.26 / 85.08 / 85.15 |
| Page, 24 epochs | 2.219 / 2.192 / 2.192 | 59.0 / 60.2 / 57.3 | 93.16 / 93.11 / 93.16 |
| Champion, 30 epochs | 1.958 / 1.615 / 1.611 | 65.2 / 58.1 / 56.3 | 94.52 / 94.54 / 94.66 |

Accuracy is the same within the seeds' spread; the steady epoch equal or faster (18 % on the champion). JAX's one
deficit, compilation, was a defect: the epoch and the schedule-free batch-norm refresh closed over the device training
set, which XLA baked into each program as a 138 MB constant (persistent cache entries of 124 MB; 0.5 MB with the data
as an argument). Passed as an argument, a whole training is 4-15 % shorter than before (most on short ones, i.e. a
search's trials) and shorter than Torch's in all three. The Torch harness (`torch_prof.py`) is removed.

The same comparison carried a patch from a session on a Tesla T4. Adopted: the data as an argument (extended to the
refresh, which the patch missed), the augmentation's key count (11 keys drawn with every augmentation on, 10 split:
`StopIteration` while tracing, an error with no message, which fits item 3 below), and the search summary's
`campaign_minutes_to` and `campaign_seconds` (from the job's start). Not adopted: FP16 with dynamic loss scaling, 2x on
the T4 (no native BF16 there) and a tie on the 5070 Ti (0 skipped updates in 9 runs); the T4-only timing tools; and a
switch of the data source to the Toronto archive, which changed the dataset's hash while its comment said it did not.

## Next

1. **JAX variants on the 5070 Ti** (`pipeline/tools/profile/jax_prof.py`), each kept only for 5 % or more with accuracy
   unchanged: XLA command buffers, Triton GEMM, a scan unrolled by 2-4, bf16 batch norm and weights, a crop-and-flip
   path when rotation and scale are off (in place of the bilinear resample), a hand-written bilinear gather, evaluation
   batch size, fewer host syncs; and 1, 2 and 6 trainings at once on the card. A card without native BF16 (T4) would
   want the FP16 path above behind a capability check, if such cards are ever used for real.
2. **The airbench family on a GPU** (Torch references from the session ledgers): `pipeline/tools/seeds.py JX-E1.2-45k
   --preset airbench94 --seeds 0 1 2` (Torch 93.70 % test, 6 views); `JX-E1.1-45k --preset airbench96 --seeds 0 1 2`
   (95.87 +- 0.04 %); airbench94 on train+val, 2 seeds (94.01-94.11 %); `tests/test_equivariance.py::
   test_levers_run_in_bf16_on_the_gpu` on a GPU machine; the flip-equivariant finalist with an airbench96 teacher
   (95.40 +- 0.20 % at one view).
3. **The failed trials** (seed 0: 4, 21, 39, batch 512-1024, six at once; seed 1: four more; all with an empty error):
   the augmentation's key count (above) is the likely cause; confirm on the next search, and keep a trial process's
   stderr in the job's directory if any still fail.
4. **Concurrency on the RTX PRO 6000:** pin `--concurrent 6` for searches there (calibration picked 3 one run and
   6 the other), and if wanted, repeat seed 1 that way for a like-for-like second seed (about $1.10 on Vast).
5. **Documents:** the README (the hardware, the stack, the timings) and `docs/RESULTS.md` (the table above, both
   seeds).
6. **Publishing:** `70_publish.py --dry-run` with the JAX champion (`results/models/20261109-102621`); after the
   user confirms, it replaces the Torch model in `thearn/cifar10-resnet9` and the demo's requirements (the Torch
   version stays in the repository's history).
7. **runq findings to fix there:** a Vast allocation that was still in progress when its job was cancelled rented a VM
   that nothing held (destroyed by hand after about 10 minutes); `runq exclusive` cannot wait for an idle card without
   pausing everyone else's queue. `runq budget` showed Vast at $0.00 after about $3.30 of Vast VMs that day (the
   search VM in Sweden $1.42 an hour for 105 minutes, two parity VMs, the stray one).

## Not done on purpose

- Torch compatibility: none (the user's rule: direct changes, no shims). The last Torch commit is `f7ab6f5`.
- The dataset stays out of git: it is rebuilt from the hub's files at a pinned revision, its hashes in
  `data/manifests/`.
