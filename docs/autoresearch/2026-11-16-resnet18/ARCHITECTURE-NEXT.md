# After boosting: FoldGrid, a score-oriented JAX CNN proposal

**Status: researched and analytically checked, not implemented or trained.**
Prepared 2026-11-16. This is a proposed successor, not a change to the running
ResNet-18 study or queued boosting study. No additional GPU budget is assumed.

## Recommendation

Keep a conservative track (remeasured compact airbench + one-view distillation)
and test one genuinely structural track alongside it: **FoldGrid**, a dense
residual CNN on even-sized grids, with a foldable patch-whitening parameterization
and training-only knowledge/translation-consistency losses.

The hypothesis is specific: move expensive early computation onto a 16x16 grid,
retain useful later-stage capacity, and recover accuracy through training rather
than extra inference views. Use ordinary GPU-friendly convolutions, not a new
attention or spectral operator. Whether this actually improves the score is open.

The working name describes this combination; it does not assert that its
primitives, combination, or name are novel in the literature.

## What the repository's evidence actually says

| Evidence | Implication for this attempt |
|---|---|
| [Capacity study](../2026-11-12-readout-recipe-spectral/RESULT-W.md): depth and width improved accuracy; the historical policy-accuracy slope was about +0.625 pp per parameter doubling. | Do not assume the smallest graph is best. Include a near-d3-capacity point and let the scalar price capacity. The historical slope is not a forecast for the new graph or one-view protocol. |
| [Distillation study](../2026-11-12-readout-recipe-spectral/RESULT-K.md): `kd_w030` gained **+0.320 pp at one view**, but only +0.073 pp under the old conditional-TTA policy, over three seeds. | KD deserves reconsideration now that deployment is explicitly **one view**. The old rejection was not a universal KD null, and three seeds do not settle the effect. |
| [Earlier compression/reparameterization](../2026-11-10-fp16-t4/SUMMARY.md) and [long-schedule reparameterization](../2026-11-11-operator-structure/SUMMARY.md) produced no promoted architecture gain. | Do not present low rank or extra train-time branches as an unexplored main mechanism. Standard BN folding is still useful deployment engineering. |
| [T4 operator gates](../2026-11-12-readout-recipe-spectral/RESULT-G.md): BlurPool cost +164.9% runtime despite identical counted MACs; a width change produced a 55% latency cliff. | MACs are diagnostics. Measure the actual complete pipeline and test neighboring widths, rather than assuming depthwise, fewer operations, or aligned channels guarantee a win. |
| [Equivariant study](../2026-10-30-equivariant-small/RESULT.md): SE, tied depth and several other additions did not transfer; run-to-run variation obscured small gains. | Avoid a collection of already-failed add-ons. Use paired controls and fresh-seed confirmation. |

Two corrections matter. The earlier larger-stem arms were stopped by a historical
latency cap **without training**; that is not evidence that their accuracy was
worse. Also, the source's odd-grid pooling does not mean all border pixels are
ignored: the preceding SAME convolutions can propagate their influence inward.

Historical reports selected under conditional TTA and sometimes exposed test.
Those are scoped research/regression evidence, not fresh measurements under the
current complete-pipeline primary protocol.

After this proposal was prepared, [E1](RESULT-E1.md) and [E2](RESULT-E2.md)
finished at confirmed scores 55.93 and 51.27 respectively. Their selected
ResNet backbone averaged 93.72%, below d3's 94.934%: it is **not the default
stronger teacher** for this track merely because its checkpoint is available.
Verify the historical higher-accuracy airbench teacher's train-only provenance
and deployed one-view accuracy before using it. No FoldGrid result follows
from either completed comparison.

## Public sources and the usable ideas

These are primary papers or official documentation. Hardware and task differences
make them design motivation, not predictions of CIFAR/T4 gains.

| Source | What informs this design | Boundary |
|---|---|---|
| [Jordan, airbench (2024)](https://arxiv.org/abs/2404.00498), [paper body](https://arxiv.org/html/2404.00498v2) | Patch whitening and a simple dense CNN are strong starting points. | Optimizes training speed, principally on A100. Its overlapping stride-1 whitening stem is **not** our proposed stride-2 stem. |
| [Chen et al., FasterNet (CVPR 2023)](https://arxiv.org/abs/2303.03667) | Memory access and achieved throughput explain why FLOPs alone mislead; favor efficient dense work before exotic cheap-looking operators. | Does not establish that its partial convolution helps tiny CIFAR maps on T4; we are not adopting PConv by default. |
| [Ma et al., ShuffleNet V2 (ECCV 2018)](https://arxiv.org/abs/1807.11164) | Direct target-platform speed is the relevant quantity. | Neither old GPU nor ARM timings transfer numerically to this host. |
| [Sunkara and Luo, SPD-Conv (2022)](https://arxiv.org/abs/2208.03641) | Review downsampling as a representation choice on small inputs. | Rearrangement alone preserves entries; subsequent learned projection/nonlinearity need not preserve information. Their non-strided spatial convolution is not our pointwise projection. |
| [Zhang, Making Convolutional Networks Shift-Invariant Again (ICML 2019)](https://arxiv.org/abs/1904.11486) | Downsampling motivates explicitly testing one-pixel shift sensitivity. | Does not validate our consistency loss. Local BlurPool results argue against assuming an inference-time blur is free. |
| [Beyer et al., A Good Teacher Is Patient and Consistent (CVPR 2022)](https://arxiv.org/abs/2106.05237) | Distillation is recipe-sensitive; teacher/student view agreement and adequate schedules need attention. | Not a demonstrated gain for this graph, split, or teacher. Teacher provenance is a prerequisite. |
| [Ding et al., RepVGG (CVPR 2021)](https://arxiv.org/abs/2101.03697) | Training and deployment graphs can differ when transformations are algebraically valid. | Not grounds to repeat locally null branch reparameterization. Here the proposed fold is a linear stem and ordinary inference BN. |
| [NVIDIA, Convolutional Layers User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html) | NHWC, FP16-friendly channel alignment and tile/wave effects guide the initial width choices. | A100 examples are not T4 measurements. Multiples of 8 are a starting prior, not a correctness restriction or proof of speed. |

For the actionable KD details, Beyer et al.'s [full text, sections 3.1.1 and
3.1.2](https://arxiv.org/html/2106.05237v2) explicitly studies consistent
teacher/student views and patient training. Its datasets are Flowers102, Pets,
Food101, SUN397 and ImageNet, not CIFAR-10. NVIDIA's sections 1 and 4.4.3 give
the FP16 multiple-of-8 recommendation; its separate 64/256 tiling advice and
A100 wave examples are not a T4 kernel prescription. XLA's actual padding and
kernel selection still need local measurement.

## Exact proposed deployed graph

Input is decoded host RGB uint8, 32x32. Normalize using the declared training
recipe, then use NHWC convolutions. All listed convolutions are dense.

| Operation | Output, excluding batch |
|---|---|
| 2x2 convolution, stride 2, VALID, RGB -> 48; activation | 16x16x48 |
| Residual block: 3x3 48 -> 48, SAME; activation; 3x3 48 -> 48, SAME; add input; activation | 16x16x48 |
| 2x2 convolution, stride 2, VALID, 48 -> 112; activation | 8x8x112 |
| Same two-convolution residual block, width 112 | 8x8x112 |
| 2x2 convolution, stride 2, VALID, 112 -> 160; activation | 4x4x160 |
| Same two-convolution residual block, width 160 | 4x4x160 |
| Global spatial max, then bias-free 160 -> 10 linear head | 10 logits |
| Host class IDs, including synchronization/transfer | One prediction per input |

Activation is GELU. At training time, put BN after each convolution, before its
activation or residual addition, with learned bias and frozen unit scale.
There are no skip projections: each residual stays at constant resolution/width.
The stem and two transitions are outside those residuals. Use the existing
airbench logit-scale convention initially; scale does not change argmax.

Fold BN into each preceding convolution at deployment. Thus the exported graph
has nine convolutions, their folded biases, three local residual additions and
one head; no BN, whitening operator, materialized space-to-depth, teacher, extra
view, squeeze/excitation, or test-time consistency pass.

The 16/8/4 grids and earlier downsampling differ from d3's 31/15/7/3 path.
They reduce high-resolution intermediates, but also change overlap, receptive
fields, nonlinearities and the final pooling region. A win would initially be a
**bundle result**, not proof that odd grids were the problem.

### Training parameterization of the stem

Pack each nonoverlapping 2x2 RGB patch into a length-12 row `p`. Fit its mean `mu`
and regularized whitening matrix `A` **on train images only**. Learn `P`, a
12-by-48 projection:

```text
training:   p -> (p - mu) @ A -> @ P -> BN -> GELU
deployment: 2x2 stride-2 convolution, folded bias -> GELU
```

Before BN folding, the kernel is `reshape(A @ P, (2, 2, 3, 48))` and its bias is
`-mu @ A @ P`. For BN with mean `m`, variance `v`, bias `beta` and unit scale:

```text
q = 1 / sqrt(v + eps)
deployed_kernel = kernel * q
deployed_bias   = (bias - m) * q + beta
```

This is an affine identity, not a new layer primitive. Space-to-depth followed by
a linear 1x1 convolution is exactly a 2x2 stride-2 convolution. Full-rank
whitening only changes the optimization parameterization, not that linear
layer's representable functions.

**Do not fold the existing airbench whitening through its GELU.** That graph has
an activation between whitening and the next learned convolution. Our proposal
deliberately removes/reorders that basis nonlinearity; losing its useful
positive/negative feature pairing is a genuine accuracy risk. Compare a
non-folded, signed-whitening/GELU stem on the same even grid as a control.

### Capacity family, not a single tiny model

Analytic counts for one two-convolution residual block per stage:

| Widths | Conv/head weights | Folded biases | Deployment scalars | Conv/head MACs per image | FP16 tensor payload |
|---|---:|---:|---:|---:|---:|
| 32/96/128 | 542,336 | 768 | 543,104 | 21,726,464 | 1,086,208 bytes |
| **48/112/160** | **823,424** | **960** | **824,384** | **35,112,512** | **1,648,768 bytes** |
| 48/144/192 | 1,219,008 | 1,152 | 1,220,160 | 48,809,856 | 2,440,320 bytes |
| 64/160/224 | 1,625,024 | 1,344 | 1,626,368 | 67,930,304 | 3,252,736 bytes |

Payload assumes every exported weight/bias is FP16. It excludes file headers,
configuration and other necessary metadata; **actual serialized deployment bytes
must replace it for scoring**. MACs exclude bias, activation, normalization,
pooling, transfers and training-only work. No latency has been measured.

For widths `a,b,c`, the weight count is
`12*a + 18*(a*a+b*b+c*c) + 4*(a*b+b*c) + 10*c`; folded biases add `3*(a+b+c)`.
MACs are
`256*12*a + 18*(256*a*a+64*b*b+16*c*c) + 4*(64*a*b+16*b*c) + 10*c`.
The train-only whitening arrays and BN statistics disappear only after a
verified fold.

## Training bundle and attribution

Start from the established 200-epoch strong-augmentation airbench recipe, not an
undertrained speed proxy. Retune its learning rate/normalization for the new
parameterization rather than assuming the old optimizer numbers transfer.

The coupled hypothesis adds two training-only signals:

1. **KD:** a frozen, provenance-checked stronger teacher predicts the exact same
   augmented image as the student. Use temperature-scaled KL plus label CE.
   The teacher must have trained only on the train split; a `train+val` teacher is
   ineligible. Save teacher identity, weights hash and training-data identity.
2. **Phase consistency:** pair an augmented image with a reflected-boundary,
   one-pixel horizontal or vertical shift of that image. Penalize disagreement
   between their class distributions. This targets a plausible patch-boundary
   sensitivity without adding an inference operation.

One concrete starting loss is
`(1-alpha)*mean_pair(CE) + alpha*T*T*mean_pair(KL(teacher_T || student_T))`
`+ beta*tau*tau*JS(student_0/tau, student_1/tau)`, where the JS inputs denote
softmax distributions. Suggested initial values are `alpha=.3, T=4, beta=.05,
tau=1`; these are proposed starting settings, not measured optima.
Compute probability losses stably in FP32 and retain the established FP16
training loss-scaling conventions. Do not silently use unsupported BF16 on T4.

Concatenate paired student views for one BN update. Disclose the effective batch
and loss normalization; the no-consistency control must receive the same paired
views, CE/KD and BN handling. Otherwise a consistency claim could just be extra
augmentation or altered BN statistics. Charge both student views and teacher
computation to the training budget; they are not free training.

First screen the coherent bundle at multiple capacities with a no-KD/no-consistency
control, then use paired leave-one-out ablations: no whitening parameterization,
no KD, and no consistency. Include the signed-whitening stem control and the
current-stem compact-CNN + KD fallback. This tests the structural idea without
requiring every component to win alone on a starved model.

## Evaluation and decision sequence

1. **Establish the real incumbent.** Remeasure the historical compact baseline,
   d3 and the final ResNet/boosting contenders under the same complete-pipeline
   protocol. The frozen d3 normalization stays fixed; it is not automatically
   the champion. Record JAX, XLA, CUDA/cuDNN, device, shapes and precision.
2. **Give incumbents the same deployment treatment.** Test honest FP16 storage,
   legal BN folding and removal of unnecessary output work where applicable.
   Existing airbench whitening/GELU remains unless a valid transformation exists.
   Report export gain separately from architectural gain.
3. **Correctness before training.** Check all shapes and finite gradients, saved
   model round-trip and legal fold parity. Verify a negative control rejects a
   fold across GELU. Ensure evaluation reads only train/validation members of
   the hashed dataset, never test arrays. Verify the deployed graph has only
   one view and its measured artifacts include everything needed to predict.
4. **Probe the real hardware.** After compute approval, benchmark candidate
   complete pipelines at batch 256 in fresh processes. Probe nearby channel
   widths: alignment and MAC reductions are not guarantees. Never reject a
   quality hypothesis solely for missing an arbitrary latency-saving threshold.
5. **Screen and tune.** Paired initial seeds can prioritize capacity/bundles, not
   establish significance. Optuna maximizes the unchanged primary score over
   stage widths/block counts, learning rate, regularization and loss strengths,
   including zero. Measure accuracy and timing on the same deployed graph.
   Do not substitute early accuracy or theoretical MACs for the objective.
6. **Confirm before promotion.** Reserve fresh training seeds and repeated
   timing processes; choose replication from measured paired variability and
   the available budget, not a claim that three or five seeds always suffice.
   Compare recipe-level mean error, timing and actual bytes, and uncertainty in
   score ratios. A marginal unresolved gain remains inconclusive. Keep the
   Pareto diagnostics, failures and negative results.

No new test read is part of this design iteration. Historical exposed test
results are regression evidence; a final locked reporting read, if authorized,
cannot erase those exposures or supply a fresh independent holdout.

Before submitting anything, freeze a protocol with a separately approved total
and per-run local-T4 budget, reserve for confirmation, and explicit stopping
conditions. No cloud. Finish the currently authorized ResNet/boosting sequence
first; this proposal does not spend or extend its original 24 hours.

## Score arithmetic, not promised performance

With the current frozen reference (94.93375%, 3.704758 ms, 7,210,007 bytes),
a **hypothetical** deployed model at 94.70%, 2.50 ms and 1,655,000 actual bytes
would score **122.19**. At 94.50% with the same costs it would score **117.75**.
Neither accuracy, latency nor serialized size has been achieved by FoldGrid.

For perspective, halving d3's bytes alone, with identical accuracy and latency,
would already yield `100 * 2**.10 = 107.18`. Thus part of a candidate's apparent
gain could be export precision, not architecture. An optimized incumbent must
receive that opportunity too. If latency does not improve, the smaller payload
can still pay for some error increase; judge the full scalar, not a hard speed
gate or MAC target.

## Checks completed during preparation

CPU-only JAX/NumPy checks independently calculated the four count rows and
compared the centered-whitening/projection/BN stem with its single stride-2
convolution on random FP32 inputs. Maximum absolute error was
`2.384185791015625e-7`; inserting GELU before the projection correctly broke the
purported linear fold. These check arithmetic and algebra only, **not training,
GPU speed or accuracy**. FP16 reassociation/rounding can change logits and class
IDs, so accuracy must be remeasured on the actual exported implementation.

The strongest unresolved risks are early loss of overlap/nonlinear feature
capacity, patch-phase sensitivity despite existing translation augmentation,
regularizers overlapping or overconstraining each other, and T4/XLA kernels
failing to turn smaller intermediates into lower complete-pipeline latency.
