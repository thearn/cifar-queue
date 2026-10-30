# Autoresearch session: an undocumented CIFAR-10 result at a small inference footprint

Session `2026-10-29-footprint-arch`, opened 2026-10-29 at commit `e980ead` (branch `layout`, dirty: the pipeline
restructure); goal revised the same day from "beat our own model at its footprint" to the two goals below. Records
here: this plan, `STATUS.md`, `ledger.tsv` (append only), `literature.md` (the review behind the candidate list),
and `experiments/<ID>/` (each experiment's frozen protocol, logs and results).

## 1. Objectives

Two goals, each a documented result to replicate first and then a target that, as far as the review in
`literature.md` can tell, nobody has published. Goal A is the main campaign; Goal B is a stretch taken up only if A
is settled or stuck with budget left.

### Goal A (main): 96 % at 2 M parameters in 40 epochs

- **Replicate:** airbench96 (Jordan 2024): 96.05 % test with TTA (95.6 % without), 37 epochs, about 9.9 M
  parameters (widths 128/384/512, three convs a block with a residual over the last two, from the repository's
  `legacy/airbench96.py`; the count is ours, the script does not print it).
- **Target A:** test accuracy **≥ 96.0 %** with **≤ 2,000,000 stored parameters** (trainable or frozen: a frozen
  dictionary or whitening bank counts, batch-norm running statistics do not), trained in **≤ 40 epochs** over the
  45,000 training images, with an inference footprint in B0's strict class (§2.2). Test-time augmentation is
  allowed and is counted in the footprint and latency.
- **Why it is open (checked 2026-10-29, `literature.md`):** the ≥ 96 % results under 2 M parameters are
  ConvMixer-256/12 and 256/16 (1.0-1.3 M, 96.4-96.7 %) at 200 epochs with RandAugment, mixup, cutmix and random
  erasing; DenseNet-BC-100 and PyramidNet-164 (0.8-1.7 M) reach 95.4-95.8 % at 300 epochs. The short-schedule
  results are airbench94 (1.97 M, 10 epochs, 94.0 %), the ConvMixer CIFAR repository (0.6 M, 100 epochs, > 94 %) and
  airbench96 itself (9.9 M, 37 epochs, 96.05 %). Nothing found combines ≤ 2 M, ≤ 40 epochs and ≥ 96 %. What M0
  still owes: FastBoost's full text (0.85 M at 95.57 %, epochs unknown) and scaled-down NAS cells.
- **Stretch within A:** the same at ≤ 1.5 M parameters, if 2 M falls quickly.

### Goal B (stretch): beat Shake-Shake 26 2x32d at its size

- **Replicate:** Shake-Shake 26 2x32d (Gastaldi 2017), 2.9 M parameters, 1,800 epochs: 96.45 % (S-S-I) or 96.53 %
  (S-E-I) with standard augmentation, verified from the paper's Table 1; **97.5 ± 0.1 % with AutoAugment**, verified
  from AutoAugment's Table 2 (2.5 ± 0.1 % error; 3.6 % baseline, 3.0 % cutout). The AutoAugment figure is the bar.
- **Target B:** test accuracy above the replicated AutoAugment baseline by the win rule in §2.1 (at least +0.20 pp,
  significant over seeds), at **≤ 2.9 M trainable parameters**. The footprint class is parameter-matched (Class P):
  Shake-Shake runs two branches a block at about 450 M MACs an image, twice B0's, so it cannot be a strict-class
  candidate and the comparison is on parameters.
- **Why it is open (checked):** RandAugment and TrivialAugment each report Shake-Shake only as 26 2x96d (TA: 98.21 %
  at 26 M); neither paper has a 2x32d row, and no later result for the small model was found.
- **Why it is the stretch:** one training is 5-8 hours on the 4060 (estimated from the MACs, not measured), so
  there is no search at this scale, only two or three hand-chosen bets, and every 5-seed confirmation is a day or
  more of local GPU time.

### Method novelty (both goals)

Both targets can in principle be reached with known tools (airbench's recipe, distillation, multi-crop TTA, a
search). That would be a new point on the frontier, not a new method. The campaign keeps room for the second kind
of result through the **method track** (§5, "Methods"): hypotheses about *how* to train or run a small network
better, each tested as an ablation on the strongest known-tools recipe, so a gain can be attributed to the idea
rather than to tuning. A validated method joins the search space as a lever; a null result is reported as one. The
known-tools recipe is established first (M2), because without it no method gain is attributable; after that the
method track runs alongside the searches and has its own budget share (§7).

### Shared non-goals

Changing the validation or test split or the accuracy function. Publishing anywhere (hub, airbench repository)
without the user's explicit approval. Quantization as the route to the parameter or byte limits (fp32 or bf16
weights only; int8 is an optional appendix). Training-time records in seconds: epochs are the budget here, the
conversion to seconds on a named card is reported once at the end for Goal A.

### Reference points (local, measured)

| | B0 (width 48, whitened) | Published model (width 64) |
|---|---|---|
| Trainable parameters | 3,709,882 | 6,573,130 |
| fp32 weights | 14.9 MB | 26.3 MB |
| MACs per image | 225 M (450 M with flip TTA) | 379 M |
| Test accuracy | 94.57 % (3 seeds, G4, 30 epochs) | 93.18 +- 0.16 % (5 seeds) |

B0 (`results/g4-best.json`) stays the local footprint reference: its strict class is the memory bound for Goal A,
and its accuracy is the first rung on the way to 96 %.

## 2. Evaluation contract

### 2.1 Accuracy and noise

- Test accuracy on the hub's 10,000 test images (`evaluate.accuracy`, unchanged) is the primary metric.
- Selection uses validation only (the fixed 5,000 images, `configs/data.toml [split]`). Test accuracy is computed
  for every run, as the project always has, but no search, ranking or early stop reads it. Its exposure history is
  recorded in `STATUS.md`: it has been reported for every run since 2026-10-27, so it is regression evidence, not an
  untouched hold-out.
- **Untouched hold-out:** CIFAR-10.1 v6 (Recht et al. 2019, 2,000 images; E0.4). Evaluated only for finalists, once
  each, every look logged.
- Seed noise: B0's 3-seed test sd is 0.09 pp; binomial standard error on the test set is about 0.2 pp at 96 %.
  Paired comparisons on the same images remove most of the latter. A search's best validation value is never
  compared against a bar directly (B0's 95.12 % val became 94.57 % test); candidates are retrained over seeds first.

**Target A is met when (all must hold):**
1. At least 5 seeds of the frozen recipe on one card type; mean test accuracy ≥ 96.00 % and the one-sided 95 %
   lower confidence bound over seeds ≥ 95.80 %.
2. Each seed trains for ≤ 40 epochs of the 45,000 training images. An epoch is one pass of the training images
   through any network that ends up in the final model, summed over branches and members: four tails of 3 epochs
   each cost 12, an ensemble of two 20-epoch members costs 40. Distillation uses precomputed teacher logits, so the
   teacher's forward passes do not count against the student; the teacher's own training cost is reported with the
   result.
3. Stored parameters ≤ 2,000,000 at inference (counted on the saved network, frozen weights included); the strict
   footprint class (§2.2).
4. Mean validation accuracy ≥ 96.0 % as well (a check against a test-only fluke).
5. CIFAR-10.1 accuracy is within its 95 % interval of airbench96's on the same images (a check against a model
   overfitted to CIFAR-10's test distribution; not a win condition, since 2,000 images carry about 0.5 pp of noise).

**Target B is met when:** the win rule against the replicated Shake-Shake baseline holds: ≥ 5 paired seeds each,
mean Δ ≥ 0.20 pp, one-sided Welch t at p < 0.05 (5 more seeds each if p is between 0.05 and 0.2), a paired
bootstrap over test images whose 95 % interval excludes 0, the same sign on validation, and ≤ 2.9 M parameters.
Given the cost, the seeds come from the local 4060 over days, and the ledger records partial seed counts as they
accumulate.

**Record-format run (Goal A, after the target is met on 45k/5k):** the frozen recipe is trained once per seed on
all 50,000 images, with no selection of any kind, and its test accuracy reported beside the 45k figure. That is the
convention of the results Goal A is compared with. It is reported, not used to decide anything.

### 2.2 Footprint and runtime (the evaluator built in E0.1, frozen before candidates are measured)

`src/cifarq/footprint.py` and `pipeline/tools/footprint_bench.py` measure a candidate through one interface:
`predict(uint8 (n,32,32,3)) -> probs (n,10)`, which includes normalisation, any TTA and any feature extraction.
Measurements run on the local machine (RTX 4060 8 GB, 16 threads, WSL2), never on a VM, so timings compare.

| Axis | Measurement |
|---|---|
| Stored parameters | every weight tensor of the saved network, frozen ones included, batch-norm running statistics excluded; trainable count reported beside it; an ensemble's is the sum |
| Stored bytes | the serialized artifact as shipped (safetensors), the sum for an ensemble |
| Peak GPU memory | `torch.cuda.max_memory_allocated` in a fresh process: weights loaded and predict at batch 1 and 256, bf16 autocast |
| Peak CPU RAM | peak RSS above the interpreter's floor in a fresh subprocess: load and predict at batch 1 and 256, fp32 |
| Compute | MACs per image, counted analytically; TTA views multiply it |
| Latency | p50 and p95 at batch 1: GPU (bf16), CPU 1 thread, CPU all threads; 20 warm-up runs, 200 timed |
| Throughput | images a second at batch 1024 on the GPU and 256 on the CPU |

**Footprint classes:**
- **Class S (strict):** at most B0 on trainable parameters and stored bytes, and at most 1.05 × B0 on every peak
  memory reading (5 % absorbs measurement noise). Goal A requires this, with the parameter bound tightened to 2 M.
- **Class P (parameter-matched):** at most the reference on trainable parameters and stored bytes; memory and
  latency reported. Goal B's class, with Shake-Shake 2x32d as the reference.

Latency and MACs are not gates. A candidate more than 2x slower than B0 at batch 1 on any row is reported as a
trade-off beside its accuracy, and 6-view TTA will do that to most candidates: the result table shows every
candidate with and without TTA.

**Evaluator acceptance (E0.1):** five repeat runs on B0 give a coefficient of variation under 5 % on latency and
under 1 % on memory; the width-64 model is flagged as outside both classes against B0; a deliberately leaky
candidate (one that reads `x_test` when built, or keeps the training set for inference) is caught, the bench
predicting on images the candidate was never given.

### 2.3 Attribution

A 2 M-parameter model reaching 96 % can owe it to architecture, to distillation, to recipe, or to a method from
§5. All count for Goal A, but the report separates them with ablations on the final recipe: no distillation; no
TTA; the airbench tricks removed one at a time; each method of §5 removed in turn (3 seeds each, local). For
Goal B, the replicated baseline is retrained with each added ingredient in turn, as far as the local GPU allows.

**What counts as a method result.** A method is written up as a finding (positive or null) only when:
1. it was frozen as a hypothesis with a mechanism before its first run, with the prior work it has to be
   distinguished from named;
2. its gain is measured as a paired-seed ablation against the strongest known-tools recipe at the time, with the
   epoch budget and footprint held equal (a method that spends more epochs or parameters is compared at equal
   budget, not credited for the extra);
3. **Promising** at ≥ +0.15 pp mean over 3 paired seeds on one family; **Validated** at 5 paired seeds, ≥ +0.20 pp,
   p < 0.05, and the gain reproduced (same sign, ≥ +0.10 pp) on a second family; **Null** when 3-5 seeds put the
   gain inside ±0.10 pp; the null is recorded with the same care as a gain;
4. it can be described in a paragraph and reimplemented from that paragraph.

A validated method enters every later search as a lever; its contribution to the final model is one of the
ablations above.

## 3. Mutation boundary

**Editable:**
- `src/cifarq/model.py` (an architecture registry; the default `arch='resnet9'` must stay bit-identical)
- New modules under `src/cifarq/arch/`
- `train.py`, extended without changing defaults: `arch` and its hyperparameters, distillation from stored logits,
  multi-crop TTA, Lookahead, alternating flip, dirac init, AutoAugment/TrivialAugment, Shake-Shake's schedule, and
  a generic `save` for non-ResNet9 networks; an `epochs` cap enforced in the search space, not by the trainer
- `tune.py`, extended: per-family spaces, a parameter cap (counted before training, pruned at step 0 over the cap)
  and an epoch cap
- New scripts under `pipeline/tools/`, new configuration files, and a teacher-logits artifact under `data/`

**Protected:**
- `evaluate.accuracy`, the dataset stages and split, `data/manifests/*`
- Earlier `results/`, `docs/RESULTS.md`'s past entries, and the published model
- The user's uncommitted restructure: campaign code is committed on a branch from it (`autoresearch/footprint-arch`)
  without rewriting their changes

**Regression gate for every code change touching training:** `pytest` (CPU, synthetic data) plus B0 at seed 0,
4 epochs, on the 4060, matching the pre-change run's validation accuracy exactly (bf16 on the same card and software
has been bit-reproducible here).

## 4. Compute: solve-queue, and keeping GPUs busy

All training and search runs go through solve-queue (`SOLVEQ_SESSION=fparch`); no VM is allocated by hand.

- **Local RTX 4060 (`local`, unlimited):** smoke tests, regression runs, the airbench96 replication, every Goal A
  family screen, the top-5 × 3-seed retrains, every 5-seed confirmation (one card type for every number in a
  comparison), the ablations, and all of Goal B unless the agent decides otherwise.
- **G4 (`colab:cifar-g4`, ~9.3 units an hour):** Goal A's exhaustive searches, as mode 2: a whole Optuna study inside
  one VM job (`55_tune_on_vm.py`, generalised to `--family`), trials as processes on the card, early stopping
  through the study, the study checkpointed so `--resume` survives a lost VM. The account holds two GPU VMs, so two
  families search at once on two G4 executors (add `cifar-g4b`); teacher training (R1) can share a VM.
- **Local CPU:** analysis only.

**GPU use is measured, not assumed.** "One training at a time" was measured for ResNet-9 at width 48 and batch 256.
A 1-2 M-parameter network at 40 epochs leaves a 102 GB RTX PRO 6000 mostly idle, and the deep narrow families
(PyramidNet-110, DenseNet-BC-100) are launch-bound. Before each family's first search there is a **calibration
job** on the G4, about 10 minutes, as in Result 9 of `docs/PLAN-mode2.md`:
- 1, 2, 4 and 8 trainings at once (`gpu.device_turn`, `SOLVEQ_DEVICE_TURNS`)
- batch 256 to 2048 with the learning rate scaled
- `torch.compile` on and off, channels_last, bf16
- recorded for each: trainings an hour, images a second, GPU busy share and memory peak from the job's `usage.json`

The slot count with the most trainings an hour at no loss in validation accuracy over 2 short seeds is stored with
`cards.set_slots` for that family.

**Targets:**
- GPU busy ≥ 90 % over each search job (B0's searches reached 97 % on the G4). A search under 80 % is stopped,
  diagnosed and relaunched; its partial study is kept and resumed.
- VM idle between jobs ≤ 2 minutes: the next search is submitted before the running one ends; the executor has
  `idle_minutes = 1`; analysis happens on the local copy, never on a VM kept up for it.
- Trials report `best_value`, `trials` and a GPU busy share (`make_job(..., report=[...])`), so `solveq wait` lines
  show utilisation as jobs end.
- `solveq budget 50 --days 14` is set before the first VM job; `solveq submit ... --check` runs before every one.

## 5. Candidates

Evidence and status (verified against the source, computed here, or recalled) are in `literature.md`. Parameter
counts are for 32x32 input and 10 classes; published accuracies are at their authors' epochs on all 50,000 images.

### Goal A: families under 2 M parameters

| Family | Variant | Params | Published | Concern |
|---|---|---|---|---|
| **A1** airbench, 2 convs a block | 64/256/256 (airbench94's net) | 1.97 M | 94.01 % at 9.9 epochs with 6-view TTA (verified) | the natural start: same recipe, more epochs |
| **A2** airbench96 layout, 3 convs a block with a residual over convs 2-3 | 64/192/256; 64/160/320; 48/224/224 | 1.6-1.9 M | 96.05 % at 128/384/512 (9.9 M, 37 epochs; verified) | scaled down, unpublished |
| **A3** ResNet-9 (B0's family) | width ≤ 35; width 32 | 2.0 M; 1.65 M | 94.6 % at width 48 here | the local control |
| **A4** PyramidNet-110 α=48 (basic) / -164 α=48 (bottleneck) | as published | 1.7 M | 95.38 % / 95.79 % at 300 epochs (verified) | 110-164 layers: slow at batch 1, launch-bound |
| **A5** DenseNet-BC L=100 k=12 | as published | 0.8 M | 95.49 % at 300 epochs (verified) | activation memory: may fail Class S |
| **A6** ConvMixer | 256/8 to 256/16, patch 1-2, kernel 5-9 | 0.6-1.34 M | 256/12: 96.39 %, 256/16: 96.74 % at 200 epochs with RandAug + mixup + cutmix + erasing (verified); 256/8: 92.5 % at 25 epochs, > 94 % at 100 (repository) | patch 1 is 4x the MACs of patch 2; its 200-epoch recipe leans on augmentation that needs many epochs |
| **A7** Shake-Shake 26 2x24d | widths 24/48/96 | ~1.6 M | 2x32d: 96.5 % at 1,800 epochs (verified) | 40 epochs is far from its regime |
| **A8** Ensembles | 2 × A3 width 24; 2 × A6 256/8 | ≤ 2 M | ensembles typically +0.5-1 pp over a member | latency is the sum |
| **A9** CCT-6/3x1 narrowed, WRN-16-3 | ≤ 2 M | ≤ 2 M | CCT-7/3x1 96.53 % (3.76 M, 300 epochs); WRN-16-4 ~95 % | high MACs; long training |
| **A10** FastBoost (arXiv 2511.01026) | as published, if code exists | 0.85 M | 95.57 % (abstract; epochs unknown) | M0 reads the full text first |

**Recipe levers shared by every family** (so attribution is fair): distillation from the teacher pool (R1);
multi-crop TTA (1, 2 or 6 views) with its cost counted; airbench's tricks (2x2 whitening first layer, dirac init,
BN without affine scale and its bias at 64x the learning rate, Lookahead, alternating flip, label smoothing 0.2);
AutoAugment, RandAugment or TrivialAugment; cutout, mixup/cutmix; the existing 53-parameter space with schedule-free
and scheduled SGD/AdamW; EMA; progressive resizing (train at 24-28 px early, 32 px late) and channels_last.

**Teacher pool R1:** airbench96 as replicated (9.9 M, M1), the same net at 80 epochs if it gains, and an ensemble
of the two. Each teacher's logits on the 45,000 training images under the fixed augmentations are
stored once, hashed, under `data/teachers/`, so distillation trials load logits rather than run a teacher and the
GPU stays on the student. Logits under random augmentation (the usual KD setting) need the teacher in the loop; that
costs roughly 6x the student's forward pass, and the calibration job measures whether it is affordable on the G4
before it enters the space.

### Methods: hypotheses about training and inference, not architectures

Each is a candidate for a method result (§2.3). They are chosen for this regime: small networks, 40 epochs, a
footprint that counts TTA. The prior work column is what the write-up must distinguish the idea from; part of
each method's protocol is a check that the prior work did not already test the same thing in a comparable setting.
Order is by expected gain per hour; the agent reorders as evidence arrives.

| ID | Hypothesis | Mechanism | Prior work to distinguish from | Cheapest discriminating test |
|---|---|---|---|---|
| **MT1** Flip-equivariant convolutions in place of mirror TTA | mirror TTA is worth ~0.5-0.8 pp in airbench at 2x inference MACs; building horizontal-flip equivariance into the network (weights shared across the mirror pair, the pair pooled at the end) gives the invariance at equal stored parameters and, if applied only to the last stage, at far less than 2x MACs | G-CNNs (Cohen & Welling 2016) and flip-invariant CNNs; those report gains at 200+ epochs with rotations, not the replace-TTA trade in a 40-epoch regime | A1 at 1.97 M: equivariant last stage vs plain + 2-view TTA, equal params, 3 paired seeds at 20 epochs; success = TTA's accuracy at < 1.5x MACs |
| **MT2** Trunk-and-tails weight averaging | one trunk for ~70 % of the epoch budget, then K short tails with different augmentation seeds and strengths, weights averaged (same basin, so the average is valid); the tails' epochs count against the budget | SWA (one trajectory), snapshot ensembles (an ensemble, Kx the inference), model soups (fine-tuning), branch-train-merge (language models); the question here is whether late branching beats spending the same epochs on the single run | A1: 28-epoch trunk + 4 × 3-epoch tails averaged, vs one 40-epoch run; 3 paired seeds |
| **MT3** Data-dependent whitening at every layer | airbench whitens only the first layer; initialising each conv in turn from the PCA of its input patches (then training, or keeping the lowest-variance directions frozen) should cut the epochs a small net needs to find good early filters | LSUV init, Thiry et al.'s patch dictionaries (frozen, one layer, kernel methods), Page's and Jordan's first-layer whitening | A1 at 10 epochs: whiten-all vs dirac init, 3 paired seeds; a gain at 10 that persists at 40 |
| **MT4** Disagreement-weighted distillation from stored logits | with teacher logits stored (R1), per-example teacher-student KL is free; weighting the loss by it, or keeping the top half of each batch by it, spends the epoch budget on the examples the student still gets wrong, at no extra forward passes (airbench96_faster buys the same with a trained proxy net) | airbench96_faster's proxy-loss filtering, focal loss, learning-to-reweight, curriculum KD | A1 + plain KD vs A1 + weighted KD at equal epochs, 3 paired seeds |
| **MT5** Weight-tied depth | repeating a block's weights r times adds depth at zero stored parameters; under a parameter cap that is depth for free, paid in MACs (counted) | recursive CNNs (Kim et al. 2016), ShaResNet, iterative inference; not tested under an airbench-style short schedule | A2 at 1.6 M with each block's last conv applied twice vs the 1.9 M untied net, equal epochs |
| **MT6** Confidence-gated TTA | run one view; add the mirror and shifts only when the top-2 margin is under a threshold set on validation; average MACs fall below full TTA at the same accuracy (the bench reports average and worst case) | adaptive TTA and early-exit literature; GPS (Lyzhov et al. 2020) learns view policies, not gating | any finished model: accuracy against average views, threshold swept on validation, read on test once |
| **MT7** Joint search over width and view set under a MAC cap | widths and TTA views trade accuracy for MACs on different curves; letting one search choose both under a cap finds a better point than fixing views first | hardware-aware NAS (cap on MACs or latency); TTA is never in those spaces | a MAC-capped variant of one M3 study, compared with the same family's uncapped study at matched MACs |

Rules for the track:
- Each method first runs on **A1** (1.97 M, about 20 s a training on the G4, a minute on the 4060), locally, as a
  3-paired-seed ablation at a short budget (10-20 epochs), then at 40 if promising. Only then on a second family.
- A method's screen is frozen like any experiment, with the prior-work check written before the run.
- Validated methods enter the M3 search spaces; promising ones enter with a switch so the search can reject them.
- The track also serves Goal B: MT2, MT4 and MT6 apply directly to Shake-Shake (B2 is MT4's plain form; MT6 is a
  cheaper B3), and are the bets most likely to be a finding rather than a tuning.
- Budget: local GPU for all screens; about 6 units of M3 for method-inclusive searches, plus the reserve at the
  agent's discretion if a method validates and the search gains from re-running with it.

### Goal B: Shake-Shake 26 2x32d and the bets against it

Baseline: the published recipe (SGD, lr 0.2, cosine without restart over 1,800 epochs, batch 128, flip and 4-px
translation, otherwise fb.resnet.torch's defaults; Shake-Shake-Image, or Shake-Even-Image which the paper's Table 1
puts 0.08 pp higher at 2x32d), then with AutoAugment's CIFAR policy (its Table 2 baseline row is 3.6 %, so the
AutoAugment paper's own Shake-Shake training is the one to match). The bets, each a single change
on the AutoAugment baseline, in the order of expected gain per hour:

| ID | Change | Rationale |
|---|---|---|
| B1 | TrivialAugment (wide) in place of AutoAugment | TA matched or beat AA on every model it was tested on, at no search cost |
| B2 | Distillation from the R1 teacher pool, soft targets at T = 4, α = 0.5 | KD is worth 0.3-1 pp on small models in the literature and is rarely combined with 1,800-epoch schedules |
| B3 | 6-view TTA at inference | Shake-Shake papers report no TTA; counted in latency, free in parameters |
| B4 | SAM (ρ = 0.05) or ASAM | +0.2-0.5 pp on CIFAR ResNets at 2x the training cost (the most expensive bet) |
| B5 | A weight EMA across the last 10 % of training | cheap; sometimes +0.1-0.2 pp |

B3 can be applied to any finished run at no training cost, so it is evaluated on every Goal B checkpoint. B1 and B2
are the bets worth seeds; B4 only if B1-B2 fall short.

## 6. Milestones and experiments

Each experiment is frozen in `experiments/<ID>/protocol.md` before it runs (template in `experiments/README.md`).

### M0: the frontier, cited (no compute; mostly done 2026-10-29)

- The frontier table is in `literature.md`, most rows verified from the papers' tables or the source files:
  airbench94/96 and airbench96_faster, ConvMixer's Appendix B, DenseNet, PyramidNet, Shake-Shake Table 1,
  AutoAugment Table 2, and the RandAugment and TrivialAugment appendices (no 2x32d row in either).
- **Still owed:** FastBoost's full text (epochs and augmentation for its 0.85 M, 95.57 % model); scaled-down NAS
  cells under 2 M and their epochs; the airbench parameter counts confirmed by running the models (E1.1).
- **Decision rule:** if a ≥ 96 % result at ≤ 2 M in ≤ 40 epochs turns up, Target A tightens (≤ 1.5 M, or ≥ 96.5 %).
  Target B's bar is 97.5 ± 0.1 %, verified; it moves only if a published 2x32d result above it is found.

### M1: infrastructure and the airbench96 replication (local; about two days)

| ID | What | Acceptance |
|---|---|---|
| E0.1 | footprint and runtime bench (§2.2) | §2.2 evaluator acceptance |
| E0.2 | `arch` registry, generic save/load, parameter and epoch caps in `tune` | regression gate (§3) |
| E0.3 | B0 over 5 seeds on the 4060; the published model's footprint | the reference numbers; sd recorded |
| E0.4 | CIFAR-10.1 v6 as a data stage (hashed, in the manifest, never used in training or search) | hash recorded; B0 on it once |
| **E1.1** | **airbench96 replicated**: the architecture and recipe of `legacy/airbench96.py` ported into `train` (A2 family at 128/384/512), 3 seeds, 37 epochs, on 45k/5k; one further seed on all 50k for the record-format comparison; the original script run once as-is on the 4060 as the port's reference | 50k run within 0.3 pp of 96.05 %; 45k runs recorded as the local reference; the parameter count confirms ≈ 9.9 M; its footprint measured (expected far outside Class S) |
| E1.2 | airbench94 replicated (A1 at 64/256/256, 9.9 epochs) | within 0.3 pp of 94.01 % on 50k; the first Goal A point: 1.97 M params |

A replication off by more than 0.3 pp is diagnosed before anything else runs: the port has to be right, since every
Goal A trick comes from it.

### M2: Goal A screens (local 4060; no units)

For each family A1-A9: implement, confirm the parameter count against the paper or the computation (within 1 %),
smoke-test on synthetic data, then 1 seed at 40 epochs with the airbench recipe transplanted (and the paper's own
recipe where it differs), with and without 6-view TTA; the bench for each.

**Kill rule:** a family more than 1.0 pp below the best family at 40 epochs on validation, after a 12-trial
learning-rate and weight-decay sweep, is dropped. Outside Class S, it is dropped as a candidate (kept as a teacher if
strong). Expected: A1-A3 and A6 survive; A4-A5 fall to launch overhead or memory; A7 falls to the epoch cap.

### M3: teachers and exhaustive searches (G4 × 2; the bulk of the budget)

- **R1 teachers** first (one G4 job, about 2 units): airbench96 at 40 and 80 epochs, their logits stored.
- **Per surviving family, one mode 2 study**, exhaustive in the sense of run to saturation rather than to a count:
  - **Space:** the family's architectural hyperparameters (widths per stage, depth, kernel and patch size, growth
    rate, as applicable) sampled only where the 2 M cap holds; epochs 20-40; the full training space; every
    recipe lever in §5 including distillation (teacher, temperature, weight) and TTA views. Ranges wide enough that
    the best region is not at an edge; a best trial at an edge widens the range and the search resumes.
  - **Sampler:** multivariate TPE with constant liar, warm-started with the family's M2 best and the airbench recipe.
    A cold restart with another seed for the two finalist families.
  - **Early stopping:** rungs re-derived per family from its first ~40 complete curves (`hpo.rung_check`); B0's
    rungs are not reused, since 40-epoch airbench schedules are triangular and finish fast.
  - **Saturation:** stop when the best validation gained under 0.10 pp over the last 40 % of trials and the top 10
    lie within 0.3 pp, or at the family's unit cap (§7).
- **Winner's curse control:** the top 5 configurations of each study are retrained over 3 seeds on the 4060; the
  family finalist is the best 3-seed validation mean.
- **Cross-family study** at the end (about 1 unit): the top 2 families plus A8 ensembles under the cap, `family` as
  a categorical.
- **Readout:** if the best 3-seed validation mean is under 95.5 % after the first two families, the agent revises
  the plan before spending further: the likely moves are more teacher capacity, the teacher-in-the-loop KD
  measured in calibration, or relaxing to the ≤ 1.5 M stretch being dropped in favour of the 2 M target.

### M3m: the method track (local, alongside M3)

Starts once M2 has fixed the known-tools recipe for A1. MT1-MT7 in order, each as §5 describes, each with its own
`experiments/MT<n>/protocol.md`. After each pair of results the agent reorders the rest. Methods that validate are
added to the running or next M3 study (a study already past saturation is re-run with the method only if the
agent judges the gain worth the units). A method that reaches Validated is also run on the Goal B baseline when
E6.1 exists, as its first bet.

### M4: confirmation and ablations (local)

For each family finalist, with airbench96 and B0 alongside, paired seeds on the 4060: 5 seeds (10 under the §2.1
rule), the bench, the paired bootstrap, one CIFAR-10.1 evaluation each, and the attribution ablations of §2.3 on the
winner. Outcome labels: Validated, Promising, Rejected, Inconclusive.

### M5: the record-format run and the write-up (local)

The validated recipe on all 50,000 images, 5 seeds; its seconds per training on the 4060 and, if a VM is up anyway,
on the G4 and A100 for the airbench convention; a results section in `docs/RESULTS.md`; a draft note in the airbench
issue format, for the user to decide about.

### M6 (stretch): Shake-Shake 2x32d replicated (local; days)

E6.1: the published recipe, 1,800 epochs, 1 seed, on all 50k (the published setting), then 1 seed on 45k/5k.
Acceptance: within 0.3 pp of the verified figure (96.45 %; E6.2 adds AutoAugment, bar 97.5 ± 0.1 %). The
time of one run decides the rest: if a run is under 6 hours on the 4060, M7 proceeds locally; if over 10, the agent
decides between the G4 (about 9 units a run from the reserve) and stopping Goal B at the replication.

### M7 (stretch): the bets

B1 and B2 first, 1 seed each at 1,800 epochs; B3 on every checkpoint. A bet within noise of the baseline is dropped;
one ahead by 0.3 pp or more gets seeds added one at a time, paired with baseline seeds, until the §2.1 rule
decides or the user stops it. B4 only if B1 and B2 both fall short, and only with the user's say so, since it doubles
the cost of each run.

## 7. Resources and stopping

- **Colab cap: 50 units** (`solveq budget 50 --days 14`), confirmed by the user on 2026-10-29; local compute is
  unlimited. The session agent allocates within these limits at its own discretion and records each reallocation in
  the ledger. Default split:
  - M0-M2, M4-M5: local only (0 units).
  - M3: ~40 units. R1 teachers ~2; Tier 1 searches up to ~6 units a surviving family (about 40 minutes on a G4,
    which at 20-40 s a training is 60-120 trials or more with the slots calibration permitting), the most likely
    families first (§8); the cross-family study ~1; ~6 for method-inclusive searches (M3m); the rest, ~4-8, to the
    two finalist families' cold restarts and widened ranges. A family that saturates early hands its leftover share
    to one still improving; no family passes 12 units.
  - M3m screens: local only.
  - Reserve: ~8 units, for resumes after lost VMs, re-runs after a calibration fix, or a Goal B run on the G4.
- **Per-job limits:** search jobs 45-130 minutes with a study checkpoint every 2 minutes; screens ≤ 15 minutes;
  `--stall-min 10` on waits; Goal B runs checkpoint every 50 epochs and resume from the checkpoint.
- **Stop when:**
  - the cap is reached and the local queue is empty of decisive work;
  - Target A is Validated, the stretch (≤ 1.5 M) has had one search round without reaching 96 %, and every method
    screened so far has a recorded outcome (a method mid-screen is finished, not abandoned);
  - every Goal A family has saturated below 95.5 % on the 3-seed means: 96 % at 2 M in 40 epochs is then reported
    as not reached, with the Pareto points measured, a valid result;
  - Goal B's replication costs more than the reserve and the local GPU can carry;
  - or the user interrupts.

## 8. Order and dependencies

```
M0 frontier ─> M1 (E0.1 bench, E0.2 arch ─> E1.1 airbench96, E1.2 airbench94; E0.3, E0.4 alongside)
            ─> M2 screens A1..A10 (local) ─> calibration per family ─> M3 R1 teachers ─> searches, 2 VMs ─> top-5 × 3 seeds
                                          └─> M3m methods MT1..MT7 on A1 (local, alongside M3) ─> validated ones into the searches
            ─> M4 confirmation + ablations (recipe, tricks, methods) ─> M5 record run + write-up
M6 Shake-Shake replication starts on the 4060 whenever it is otherwise idle for 8+ hours (nights), after M2 ─> M7 (validated methods first)
```

The local 4060 is shared between M2/M3m screens and the retrains; the agent keeps a short local queue so the card
is never idle while a VM search runs, and prefers the method screen that would change the running search soonest.

Search order on the two VMs, most likely gains first: A1 + A2; A3 + A6; then A8/A9 as their screens justify; then
the cross-family study. Re-prioritise after each pair, in the ledger.
