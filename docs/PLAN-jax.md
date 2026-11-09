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

## Next

1. **Torch against JAX profiling on the local 5070 Ti** (`pipeline/tools/profile/`, untried on a GPU yet): end to end (4 epochs), component timings at the same batch (augment, forward,
   forward-backward, train step, evaluation), kernel traces (`torch.profiler`, `jax.profiler` via Perfetto) for the
   champion, Page's recipe and the defaults; then JAX variants, each kept only for 5 % or more with accuracy unchanged:
   XLA command buffers, Triton GEMM, a scan unrolled by 2-4, bf16 batch norm and weights, a crop-and-flip path when
   rotation and scale are off (in place of the bilinear resample), a hand-written bilinear gather, evaluation batch size,
   fewer host syncs; and 1, 2 and 6 trainings at once on the card. **Scheduling:** the GPU is shared with another
   session's solves; `runq exclusive` pauses the whole `gpu` group while it waits (it held back that session's next
   solve on 2026-11-09 and was cancelled), so either wait for that queue to drain, slip the jobs between two of its
   solves with its owner's agreement, or rent a 5070 Ti on Vast (about $0.16 an hour).
2. **The airbench family on a GPU** (Torch references from the session ledgers): `pipeline/tools/seeds.py JX-E1.2-45k
   --preset airbench94 --seeds 0 1 2` (Torch 93.70 % test, 6 views); `JX-E1.1-45k --preset airbench96 --seeds 0 1 2`
   (95.87 +- 0.04 %); airbench94 on train+val, 2 seeds (94.01-94.11 %); `tests/test_equivariance.py::
   test_levers_run_in_bf16_on_the_gpu` on a GPU machine; the flip-equivariant finalist with an airbench96 teacher
   (95.40 +- 0.20 % at one view).
3. **The failed trials** (seed 0: 4, 21, 39, batch 512-1024, six at once; seed 1: four more; all with an empty error):
   keep a trial process's stderr in the job's directory, reproduce with six at once, and see whether it is memory.
4. **Concurrency on the RTX PRO 6000:** pin `--concurrent 6` for searches there (calibration picked 3 one run and
   6 the other), and if wanted, repeat seed 1 that way for a like-for-like second seed (about $1.10 on Vast).
5. **Documents:** the README (the hardware, the stack, the timings) and `docs/RESULTS.md` (the table above, both
   seeds).
6. **Publishing:** `70_publish.py --dry-run` with the JAX champion (`results/models/20261109-102621`); after the
   user confirms, it replaces the Torch model in `thearn/cifar10-resnet9` and the demo's requirements (the Torch
   version stays in the repository's history).
7. **runq findings to fix there:** a Vast allocation that was still in progress when its job was cancelled rented a VM
   that nothing held (destroyed by hand after about 10 minutes); `runq exclusive` cannot wait for an idle card without
   pausing everyone else's queue.

## Not done on purpose

- Torch compatibility: none (the user's rule: direct changes, no shims). The last Torch commit is `f7ab6f5`.
- The dataset stays out of git: it is rebuilt from the hub's files at a pinned revision, its hashes in
  `data/manifests/`.
