# Session 2026-11-10: T4 efficiency and the CIFAR-10 accuracy/size/compute frontier

A one-day research session on a local Tesla T4 (16 GB, compute capability 7.5), FP16 throughout, on the local
Toronto-build split (`dataset-56cff0f67835.npz`). This is the summary. The per-milestone reports, ledger, frozen
protocols and experiment harnesses stayed with the session and are not part of the repository.

## Outcome

| Outcome | Status |
|---|---|
| FP16 training path for cards without native bfloat16 (`precision='auto'` / `'f16'`) | **Adopted as an option.** About 2.2x faster end to end on the T4. |
| Conditional-view inference policy (confidence-gated TTA) | **Validated on test.** Same accuracy, weights and bytes, with 66.3% less measured wall time. Not yet realized in the library (see below). |
| MACs as a cost model on this card | **Rejected.** 1.278x the MACs ran in 1.011x the time. |
| Architecture changes (subspace compression, reparameterization, distillation, width allocation, task-sensitive low-rank fitting) | **None promoted.** |

## What this commit adopts

On landing (2026-11-11) `'auto'` became airbench's default and takes the summed-loss question from
`airbench.summed_loss` rather than an optimiser list of its own. The evaluation batch became a fixed 500
(`airbench.EVAL_BATCH`) with no hyperparameter: 2000 also ran out of memory on the 5070 Ti (airbench96), and from 500
up the time was flat within 4 %. The rest is as below.

- `cifarq.precision` holds `compute_dtype` and `native_bf16`. `precision` gains `'f16'` (true float16) and `'auto'`
  (bfloat16 where the card runs it natively, which is compute capability 8.0 and up, and float16 where it does not).
  `'auto'` uses float16 only under a summed loss (`sgd`, `muon`). That sum is about `batch_size` times a mean loss,
  so it works as a static loss scale. `'fp16'` and `'amp'` still mean bfloat16, and airbench's default is still
  `'fp16'`, so every earlier record reproduces as it was measured. On an Ampere or newer card, `'auto'` behaves the
  same as `'fp16'`.
- The new `eval_batch` hyperparameter (`precision.eval_batch_size`) exists because the evaluation batch was fixed at 2000. A six-view pass then holds
  12,000 images and needs about 4 GiB of activations, which ran out of memory on the 16 GB T4. It is now 2000 on
  cards with at least 24 GB and 500 on smaller ones, and can be overridden. Each image's logits depend only on that
  image, so the batch size never changes an accuracy.
- `airbench.param_counts` counted a `jx.Frozen` array as trainable. It is now stored but not trainable.
- `whiten_init` moved to `methods`, and `crop`/`cutout` moved to `augment`, which keeps `airbench.py`
  under the module line limit.
- The tests now run on a GPU (`JAX_PLATFORMS=cuda,cpu pytest`; leaving out `cpu` breaks the tests that
  deliberately run on it). The device assertions compare against `jx.device_name()`. On a GPU, the vmapped-versus-
  plain loss is compared to `rel=5e-3`, because vmapping changes which kernels XLA picks; the CPU check is still
  exact. `tests/test_precision.py` covers the dtype choice on stand-in cards.

The ResNet-9 path (`train.py`, `jx.COMPUTE`) is unchanged. Its loss is a mean, so float16 would need real loss
scaling.

## Key measurements (T4)

| | Value |
|---|---|
| Matmul throughput: fp32 / bf16 / fp16 | 4.87 / 5.14 / **26.28** TFLOP/s |
| airbench94, 8 epochs, 3 paired seeds: bf16 vs f16 training time | 52.9 s vs **24.2 s**; test 0.9329 vs 0.9323 |
| airbench94, full recipe, 5 seeds, `auto` | test **0.9378** (six-view), val 0.9368, 24.9 s training; test seed SD 0.065 pp |
| FP16 vs FP32 inference throughput, batch 256 | **2.26-2.82x** |

The speedup comes from using the card's float16 tensor cores. The measured accuracy difference is small, but this is
evidence for practical use, not a proof that the two precisions are numerically equivalent.

## The inference result

The method: run one view for every image. Images whose top-two softmax margin is below a threshold set on validation
(`tau = 0.477431` for the tested checkpoint, about 15% of images) get six views of the same network.

| | Policy | Six views for all |
|---|---:|---:|
| Test accuracy | **0.9399** | 0.9399 |
| Paired bootstrap lower bound | -0.03 pp (margin -0.20) | - |
| Expected MACs per prediction | **122.8M** | 414.4M |
| Measured wall time, 5,000 images | **0.0995 s** | 0.2956 s |

The policy matched or beat six views on six more checkpoints from two training recipes. The threshold depends on the
checkpoint and has to be re-derived on validation for each model. At batch 1, dispatch overhead dominates, so most
of the benefit is in batched serving. This is the repository's own MT6 confidence-gated TTA.

**Not yet realized in the library.** `methods.gated_logits` computes full TTA for the whole batch and then selects
with `jnp.where`, so its `views_per_image` reports a saving it never makes. The fix, verified in a session harness:
1. Run the cheap pass and compute the margins.
2. Gather the low-margin images.
3. Run six views on that subset, padded to a fixed bucket so XLA compiles only a few shapes.
4. Scatter the results back.

The gathered version gave predictions identical to the masked one. A production version needs a test asserting that
equality, and a `views_per_image` that counts the work actually done. That is the next engineering task. It is not
in this commit.

## Why the compression work found nothing

The first half of the session shrank the network's MACs inside a fixed one-shot graph. The best tradeoff was
`(42,168,168)`: 21.8% fewer MACs, but a five-seed validation delta of -0.09 pp with a lower bound of -0.38 pp, which
fails the -0.20 pp non-inferiority margin. Measuring cost directly explained why: on this card, small models are
dispatch-bound.
- At batch 1, every small model ran in 0.70-0.85 ms whatever its MACs.
- The 8.4x-larger teacher cost only 1.42x the time.
- A 27.8% MAC cut gave no measurable speedup.

The levers that turn into time here are **precision**, **how many passes run**, and **which images get the
expensive pass**.

Other negatives, each limited to the recipes, ranks and seeds tested:
- Donor-assisted rank-8 compression lost 0.42-0.44 pp against the same compact graph trained from scratch.
  Compact and plain narrowing tied.
- Fixed PCA and class-mean subspace bases gave no advantage over random or learned bases.
- Structural reparameterization (RepVGG-style) gave -0.83 pp at about 10 epochs and +0.03 pp at 40, for about
  1.65x the training cost.
- Logit KD from a 95.9% teacher gave no gain at weights 0.1 or 0.3. Early-heavy and late-heavy width allocations
  were both worse than balanced.
- Task-sensitive low-rank fitting preserved donor logits much better (KL 1.42 to 0.18) but gave lower adapted
  validation accuracy. A better immediate imitation was not a good selection proxy here.
- A teacher cascade reached 95.22% against the teacher's 95.28% in 57.8% less time, but the bootstrap lower bound of
  -0.22 pp missed the -0.20 pp gate.
- Long-schedule augmentation gave +0.49 pp at one view, but +0.01 pp under the deployed policy. Train-time and
  test-time augmentation largely overlap.

## Caveats

- Everything was measured on one T4. Rejecting MACs as a cost model is specific to this hardware.
- Exploratory screens early in the session used the test set. Later test reads are regression evidence, not a fresh
  holdout. Policy and threshold selection used only train and validation.
- About 47 training attempts were recorded for the compression and regularization tranches (about 67 minutes).
  Earlier exploratory runs are not fully accounted for. There was no cloud spend.

## Follow-ups

1. A gather-based `methods.gated_logits`, so the inference saving actually happens.
2. `footprint._measure` hardcodes bfloat16 on GPU. Route it through `precision.compute_dtype` so the deployment
   bench is correct on Turing cards.
3. Not authorized during the session: INT8 inference on the T4's integer tensor cores, distilling six-view behaviour
   into a one-view student, and input-adaptive resolution as a cascade variant.
