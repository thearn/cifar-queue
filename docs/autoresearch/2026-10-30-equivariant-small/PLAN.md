# Autoresearch session: a TTA-free 96 % at 2 M parameters, and equivariance as the lever

Session `2026-10-30-equivariant-small`, opened 2026-10-30 at commit `3c4eb93` (branch `fparch/main`, the closed
session `2026-10-29-footprint-arch`'s integration branch, which carries all the code). Records: this plan,
`STATUS.md`, `ledger.tsv` (append only), `literature.md` (sources, code and models found, points to consider), and
`experiments/<ID>/` (frozen protocols, results). The previous session's `RESULT.md` is the starting point.

## 0. What the last session established (the premises of this one)

- A depth-3 airbench at 1.94 M parameters with flip-equivariant convolutions in its last two groups (`flip_eq=2`,
  method MT1) reaches **95.45 ± 0.17 % test with 6 TTA views and 95.00 ± 0.08 % with one view**, 40 epochs on the
  45k training split, 5 seeds (`M4-fe2-t158`). airbench96 (9.9 M, 37 epochs, 6 views) reaches 95.87 ± 0.04 % on the
  same split and 96.01 % on 50k.
- Flip equivariance was the only method with a measured gain: +0.58 pp test at one view against the plain network at
  matched parameters, +0.16 pp with TTA. Distillation from a stronger (ensemble) teacher, trunk-and-tails averaging
  and 18-view TTA gave nothing at 40 epochs. Recipe search saturated at ~150 trials; single-seed search winners were
  0.2-0.5 pp lucky.
- The binding constraint is what a 2 M network absorbs per epoch, not teacher quality or parameters.
- Infrastructure: `cifarq.airbench` with MT1-MT6, `amp`, ensemble KD, TTA levels 0-3; family search spaces with warm
  starts; `fit_rungs.py`; `compare_b0.py`; the frozen footprint bench (`footprint.py`, revision a3); solve-queue with
  local GPU reservations and sampled Colab rates (its `main`, 3c26405).

## 1. Objectives

### Target (main): ≥ 96.0 % on CIFAR-10 with ≤ 2 M parameters and one forward pass per image

- **Documented points it sits against** (`literature.md`): airbench96, 96.03 % at 9.9 M with 6 views, 37 epochs;
  airbench96_faster, 96.00 % at 9.9 M, 45 epochs, 6 views, half-batch proxy filtering; ConvMixer-256/12 and /16,
  96.4-96.7 % at 1.0-1.3 M, 200 epochs, RandAugment + mixup + cutmix + erasing; p4m-ResNet44 variants, 1.9-2.6 M,
  93-95 % at long schedules; the previous session's 95.45 % (6 views) / 95.00 % (1 view) at 1.94 M, 40 epochs.
  **Nothing found at ≤ 2 M parameters, under 200 epochs and ≥ 96 % with a single view.** The one-view condition is what
  makes this a different point from airbench96 at a fifth of its parameters and a tenth of its inference MACs
  (93 M against 9.9 M × 6 views).
- **Target is met when** (all hold, §2.1): ≥ 5 seeds on one card type, mean test ≥ 96.00 % with the one-sided 95 %
  lower bound ≥ 95.80 %; ≤ 2,000,000 stored parameters; `tta_level` 0 (one forward pass, no mirror, no shifts);
  ≤ **E\* epochs** of the 45k split, E\* set by M2 within 80-150 (§2.1 defines an epoch); the strict footprint class against B0; validation ≥
  96.0 %; CIFAR-10.1 within airbench96's interval. The record-format run (50k, no selection) is reported beside it.
- **The epoch cap E\* (decided with the user 2026-10-30): set by M2, at most 150.**
  - *The ceiling is set by novelty, not cost.* ConvMixer-256/12 already has 96.39 % at 1.0 M parameters with one view
    at 200 epochs. At ≥ 200 epochs the target is documented, so the cap must stay well below it to be a new point;
    150 leaves a clear 25 % gap and keeps "short schedule" honest (airbench96 uses 37-45).
  - *The floor is set by evidence.* 40 epochs was binding last session; 80 is the smallest budget worth a G4 search.
  - *Within that range, lower is the stronger claim*, and every search trial costs in proportion to epochs (~45 s at
    100 on the G4, ~70 s at 150), so E\* is the smallest grid point in M2 where the finalist's one-view validation
    mean reaches 95.7 % (the level a search plus methods can plausibly close by 0.3 pp), rounded up to 80, 100,
    120 or 150. Frozen once chosen; every later comparison uses it.
  - *If 150 does not reach 95.7 % at one view*, the target is revised with the user before M3 (candidates: six
    views allowed, or ≤ 3 M parameters).
  - **Decided 2026-10-30 after M2 (the user: "methods first, then decide"):** the finalist's one view plateaued at
    ~95.6 % from 80 epochs, so the gate failed. **E\* = 100** (the plateau's onset; six views cross 96 % there).
    M3 and M4 run on the one-view target. If they do not lift one view to ≥ 95.7 % at 100 epochs, the target falls
    back to **≥ 96.0 % at ≤ 2 M parameters, ≤ 100 epochs, six views** (already ~96.08 % val on 2 seeds) without
    asking again.

### Method goal: equivariance as the substitute for augmentation and test-time augmentation at matched inference cost

The claim to test (H1): in airbench-style short schedules, a network with the symmetry built in (flip, then flip +
90° rotations as p4m, then anti-aliased downsampling for shifts) reaches a given validation accuracy in fewer epochs
and at fewer inference MACs than the plain network with the matching TTA views, at matched parameters and at
matched MACs, and the advantage shrinks as the schedule lengthens. A 2026 preprint (Adly, `literature.md`) finds on a
controlled C_n task that augmentation plus test-time orbit averaging matches an equivariant model's per-epoch curve
exactly, so the gap "depends on asymmetric test-time computation". That is precisely the matched-compute comparison
nobody has published on a natural image task in this regime, and either outcome is reportable: equivariance wins
(the method result) or it ties at equal inference compute (which still justifies the TTA-free target by making the
views free). The measurement is M3, before any search.

### Stretch: ≤ 1.34 M parameters above ConvMixer-256/16's 96.74 % at ≤ 200 epochs

Taken up only if the main target is met with ≥ 15 units left. ConvMixer at patch 1 trained at 13 minutes an epoch on
the 4060 and was 1.5 pp behind airbench at 40 epochs (M2 of the last session); this stretch is a long-schedule
contest on the G4 (~30 units a run) and is not planned in detail.

### Non-goals

Training time in seconds as a goal or gate (recorded only; the airbench leaderboard rules are noted in
`literature.md` for the write-up). Quantization. Changing the split or the accuracy function. Publishing or opening
a pull request against airbench without the user's approval. Repeating the last session's rejected levers (ensemble
distillation at short schedules, trunk-and-tails, TTA beyond 6 views, DenseNet/PyramidNet on the 4060, per-family
recipe search past saturation).

## 2. Evaluation contract

### 2.1 Accuracy, noise and the sealed test set

- **Selection uses validation only** (the fixed 5,000 images, `configs/data.toml [split]`, `default_rng(1234)`).
- **The test set is sealed for this session.** The last session reported test on every run; for a claim against
  published numbers it must not inform anything. `airbench.train` gains `test='sealed'` (M1): test accuracy is
  neither computed nor written for any screen, search or retrain. It is computed once per final model in M6, by
  `compare_b0.py`, and that look is logged in `STATUS.md`. B0's and airbench96's test numbers from the last session
  are the references and are not recomputed.
- **Search objective = the mean over two seeds.** Each trial trains seeds 0 and 1 of its configuration; the pruner
  sees seed 0's curve (rungs fitted from report-only curves, as before) and seed 1 runs only for survivors; the
  study's value is the mean. The last session's search values were 0.2-0.5 pp optimistic from one seed; two halves
  that and keeps the top-k retrain honest.
- **Seeds.** Screens: 3 paired seeds (1-3; the searches use 0-1). Confirmations: 5 seeds (1-5), same card type for
  every number in a comparison. Paired on seed and on augmentation draws where the code allows.
- **An epoch** is 45,000 forward-plus-backward passes through a network that ends up in the final model, summed
  over branches and members (as before). New this session: **half-batch filtering counts the images actually
  back-propagated** (airbench96_faster's 50 % keep is 0.5 epoch a pass), and the proxy's or teacher's own forward
  passes are not charged but their training cost is reported and capped at 25 % of the student's charged epochs.
- **Win rules:**
  - *A method against its baseline:* Promising at ≥ +0.15 pp mean validation over 3 paired seeds; Validated at
    5 paired seeds, ≥ +0.20 pp, one-sided Welch p < 0.05 and the same sign at a second epoch budget; Null when
    3-5 seeds put the gain inside ± 0.10 pp. Reported at matched parameters **and** matched MACs (two baselines where
    they differ, §2.2).
  - *The target:* as §1. The lower bound and Welch test come from `compare_b0.py`'s functions (no scipy).
- **Untouched hold-out:** CIFAR-10.1 v6, one look per final model, logged by `holdout.look`. Three looks were taken
  in the last session (`data/holdout/looks.jsonl`); it remains a check, not a selection signal.
- **Record-format run:** the frozen recipe once per seed on all 50,000 images (`train_on='train+val'`), reported
  beside the 45k figure, as the convention of the numbers the target is compared with.

### 2.2 Footprint, MACs and matched comparisons

- The frozen bench (`footprint.py` a3, `footprint_bench.py`) measures every final model against B0, as before:
  parameters, bytes, MACs per image including views, GPU and CPU latency, peak memory; classes S / P / out. B0's
  row and the finalist's rows are in `results/footprint/20261030-0855*.json`.
- **MAC counting of equivariant layers must count what runs**, not the stored weights: FlipConv expands to a plain
  convolution of 2× channels (the hooks already see that); the p4m layer expands 4× or 8×. M1 adds a unit test that
  the counter's MACs for an equivariant layer equal those of the expanded plain convolution.
- **Matched-MAC baselines.** When a method multiplies MACs (group convolutions, BlurPool's extra filter, a Fourier
  block), its comparison includes a plain network widened to the same MACs, under the same parameter cap where
  possible. The method's gain is the smaller of the two deltas (matched parameters, matched MACs), and both are in
  the ledger.

### 2.3 Attribution and the method-result rules

Unchanged from the last session's §2.3: one variable or one declared bundle per experiment; a method is reported
as a method only with its ablation on the strongest known-tools recipe; it enters a search space only after
Promising; a Null is recorded with the same care; the final model's gains are decomposed by ablation in M6.

## 3. Mutation boundary and branches

**Editable:** `src/cifarq/airbench.py` (new levers behind flags whose defaults keep every earlier run bit-identical;
`test='sealed'`), `src/cifarq/methods.py` and new modules under `src/cifarq/` (p4m convolution, BlurPool,
structured receptive fields, the Fourier block, SE, proxy filtering), `src/cifarq/families.py` and `tune.py` (the
two-seed objective, the equivariant space, a multi-objective study), `pipeline/tools/*`, tests.

**Protected:** `evaluate.accuracy`, the dataset stages, split and manifests, `footprint.py`'s measurements (a
revision needs a new attempt id and a repeat of B0's row), the last session's records, `results/` from before this
session, `layout` and `master`.

**Git (agreed with the user 2026-10-30; the autoresearch skill now carries the same rules):**
- **`eqsm/main`**, from `fparch/main` at `3c4eb93`, is the integration branch. The records (plan, status, ledger,
  protocols, results notes) are committed **only here**, never on a side branch (last session some ledger rows
  reached the integration branch only through merges).
- **Branches are per unit of new code, not per experiment,** and hold as many commits as the work needs.
  - `eqsm/M1`: all of M1's infrastructure, one commit per piece (sealed test, two-seed objective, p4m layer,
    BlurPool, structured receptive fields, Fourier block, SE, proxy filtering, the MAC-count test, the matched-MAC
    helper), every lever behind a flag whose default keeps earlier runs bit-identical. Merged `--no-ff` once, after
    the CPU tests and the regression gate.
  - Screens, the epoch curve and searches that run merged code get **no branch**; their protocols and ledger rows go
    straight onto `eqsm/main`.
  - A branch only when an experiment needs new code mid-session (a repair, Q3's separable layer): its protocol first,
    then the code; `--no-ff` merged if Promising or Validated, left unmerged as the record if not.
- **Exact trees:** every run records its commit and dirty state; `seeds.py` keeps `refs/autoresearch/<exp>` from
  `git stash create` at submission.
- **Artifacts outside git** (`results/`: jobs, models, study databases) by path; the final model's weight and config
  sha256s go in `RESULT.md` and the ledger.
- **At completion, when the user confirms the session is done:** delete the merged branches and the dead ones,
  keeping each unmerged tip as `refs/autoresearch/archive/<branch>` with its SHA in the ledger, so the findings and
  the code behind them stay reachable without cluttering `git branch`. Merging into `master`, pushing and publishing
  only with the user's say-so.

**Every launch sets `SOLVEQ_SESSION=eqsm`** so the watcher and `solveq brief` see the jobs (two of the last session's
screens ran under the wrong label).

**Regression gate for code touching training:** `pytest` on the CPU; a saved model (the finalist, seed 5) predicting
bitwise identically under the old and new code; and the finalist's recipe at seed 1 for 4 epochs on the 4060 within
run-to-run noise of the old code (not exactly: `cudnn.benchmark` makes airbench training non-bit-reproducible here;
measured 2026-10-30, three runs 0.8456-0.8496 val). Each run of the new code needs a changed no-op argument, or the
queue returns the old run's result (it de-duplicates identical calls).

## 4. Compute, budget and the GPU-use rules

- **Local RTX 4060:** every screen, every confirmation, the ablations, the bench, the hold-out look. solve-queue's
  reservations (3c26405) now keep 7000 MB jobs one at a time; every heavy submission still states `--gpu-mb 7000`.
- **G4 (`colab:cifar-g4`):** the epoch curve (M2), the H1 curves (M3), the searches (M5), the Pareto study (M7).
  Measured rate **10.05 units an hour** (europe-west4, 2026-10-30; `solveq colab rates` samples it); a 40-epoch
  airbench training took ~20 s there, ~180 an hour, one at a time (calibration found no gain from two). Equivariant
  layers in the last two groups cost ~2-4× those groups' MACs, so expect 25-40 s at 40 epochs and 60-100 s at 100.
- **Budget:** 75 units (the user, 2026-10-30), tracked as the fall of the Colab
  balance while only this session's VM is up, `solveq colab hours` as the check. Split: M2 ~3, M3 ~6, M5 two studies
  ≤ 20 each, M7 ~8, M6's record-format runs local, reserve ~15 (resumes, a cold restart, a longer M2 point). Nothing on the G4 before M2's readout.
- **Targets, as before:** GPU busy ≥ 90 % over each VM job (under 80 %: stop, diagnose, resume the study); VM idle
  ≤ 2 minutes between jobs; `--check` before every submission. A VM job never starts without the watcher running.

## 5. Candidates

### 5.1 The equivariance track (the method goal; H1 and its parts)

| ID | Lever | Mechanism | Evidence and cost | First test |
|---|---|---|---|---|
| **Q1** p4m group convolution, last two groups | kernels shared over the 8 flips and 90° rotations; the group pooled at the end (`PairMaxPool` generalised) | Cohen & Welling: p4m-ResNet44 beats planar at equal parameters, but their numbers were not reproduced by two later papers; Weiler & Cesa: steerable WRN28/10 3.87 → 3.28 % with no extra test cost at equal width. Rotations are not a CIFAR symmetry: their "decreasing equivariance" D8→D4→D1 beat global equivariance on natural images, and our `flip_eq=2` is already the D1 tail of that pattern | p4m in group 3 only, D4 in group 2 + D1 in group 3, and D1 D1 (the finalist), each at matched parameters (widths lowered) and at matched MACs (plain widened), 3 seeds, 40 and 100 epochs |
| **Q2** anti-aliased downsampling (BlurPool) | a fixed low-pass before each `MaxPool2d(2)` makes the features shift-equivariant up to sampling, which is what airbench's translate-TTA views compensate for | Zhang 2019 (`adobe/antialiased-cnns`): consistency gains; no CIFAR accuracy table found, so the accuracy effect is open. Costs one depthwise filter per pooling | finalist ± BlurPool at one view against the finalist with 6 views; 3 seeds |
| **Q3** separable group convolution | the group dimension factored out (SepGrouPy), cutting the MAC multiplier of Q1 | Lengyel & van Gemert; p4m-ResNet44 8.06 vs 13.10 % error on unaugmented CIFAR at ~8× MACs for the full version | only if Q1 is Promising but fails the matched-MAC comparison |
| **Q4** structured receptive fields | each 3×3 (or 5×5) kernel a learned combination of a fixed Gaussian-derivative basis (orders 0-2, one or two scales); fewer parameters per filter and a smooth prior | Jacobsen 2016: gains on small and medium data; N-Jet (Pintea 2021) learns the scale σ; a 2022 fully trainable variant has code. No short-schedule result anywhere; implementation is ~40 lines | all convolutions of the finalist, basis of 6 at 3×3 and 10 at 5×5, matched parameters by width; 3 seeds |

### 5.2 Other operators and blocks (cheap screens, local)

| ID | Lever | Why it might help here | Risk |
|---|---|---|---|
| **O1** Fourier global filter in group 3 | FFT → learned complex filter per channel → inverse FFT (GFNet); global mixing at 4×4 or 8×8 for C×8×5×2 parameters, trivial MACs | GFNet reports ImageNet only; FFT needs fp32 (`precision='amp'`); untested on 32×32 from scratch |
| **O2** squeeze-and-excitation | ~0.3-1 pp on CIFAR ResNets at 400 epochs; nearly free in parameters | no controlled short-schedule result exists; attention may need epochs to pay |
| **O3** weight-tied depth (MT5, built, unscreened) | depth-3 accuracy at depth-2 parameters | MACs rise; the last session's search preferred depth 3 strongly, so the prior is favourable |
| **O4** Muon optimizer | airbench94's current record (2.59 s) uses it; untested at the 96 % scale by anyone we found | interacts with Lookahead and the bias/scale groups; one screen at the finalist's recipe |

### 5.3 The data side

| ID | Lever | Evidence | Note |
|---|---|---|---|
| **D1** proxy filtering (airbench96_faster) | a 32/64/64 proxy trained first; the student back-propagates only the hardest 512 of each 1024 by proxy loss; 96.00 % at 45 epochs and 9.9 M, n = 200 | the documented record trick in exactly this regime; at our size the question is whether a 2 M student benefits from spending its epochs on the hard half. Charged at 0.5 epoch a pass (§2.1), so 100 charged epochs buys 200 passes |
| **D2** disagreement-weighted distillation (MT4, built, unscreened) | the same idea with teacher-student KL as the score and the airbench96 teachers from the last session | the plain KD gain was +0.1-0.3 pp; D1 and D2 are compared head to head |

### 5.4 Not pursued, with reasons

Depthwise-separable and inverted-residual blocks (slow convergence; ConvMixer 1.5 pp behind at 40 epochs; only the
MAC axis would gain). Attention/ViT blocks from scratch (data-hungry at 45k). Scattering/wavelet front ends (cap at
93-94 %). CKConv/FlexConv continuous kernels (92 % at 0.67 M; large receptive fields are not the limit at 32×32).
Ensemble teachers, tails, > 6 views (last session's nulls).

## 6. Milestones, readouts and stop rules

- **M0 Literature close-out (local, 0 units):** fetch Weiler & Cesa's parameter counts and epochs; the Hiverge
  successor records (the airbench README says they are now faster); the N-Jet code; the SepGrouPy layer. Done when
  `literature.md` marks each target-relevant figure verified or unverifiable.
- **M1 Infrastructure (local):** `test='sealed'`; the two-seed objective and racing in `tune.on_vm`; the p4m layer
  (hand-rolled like `FlipConv`, by `rot90` expansion of 3×3 kernels, channels_last and fp16 safe; escnn is not
  used, see `literature.md`); BlurPool; SRF convolution; the Fourier block; SE; proxy filtering; the MAC-count test
  for expanded layers; the matched-MAC widening helper. Each with CPU tests; the regression gate run once at the end.
- **M2 The epoch curve (G4, ~3 units):** the finalist (D1 D1) and its plain matched-parameter twin at 40, 80, 100,
  120 and 150 epochs, 2 seeds each, validation at one view and at 6 views. **Readout:** E\*, the smallest of
  80/100/120/150 where the finalist's one-view validation mean reaches 95.7 % (§1). If 150 does not, the target is
  revised with the user before M3 (candidates: 6 views allowed, or ≤ 3 M parameters).
- **M3 H1 curves (G4, ~6 units):** Q1's three equivariance patterns and Q2, against plain + 2 views and plain + 6
  views, at matched parameters and at matched MACs, at 20, 40 and E\* epochs, 3 seeds. **Readout:** the method
  result stands or is null. If no equivariant arm beats plain + TTA at matched MACs by ≥ 0.15 pp at any budget, H1
  is reported null and the target rests on §5.2-5.3 plus the recipe; the session continues either way.
- **M4 Screens (local, ~2 days):** Q4, O1-O4, D1, D2 on the best M3 arm at the M2 epoch budget, 3 paired seeds each,
  matched parameters and MACs; Promising levers join the space. After every two results the order is revised.
- **M5 Searches (G4, ≤ 24 units):** one noise-aware study on the equivariant airbench family with the Promising
  levers and the recipe (epochs up to the budget, widths under the cap, the equivariance pattern as a categorical,
  Muon/SGD, filtering on/off), warm-started from M3's best; report-only rungs first, `fit_rungs.py`, resume. A cold
  restart with another seed if the first saturates below target. **Saturation rule:** best two-seed validation up
  < 0.10 pp over the last 40 % of trials and the top 10 within 0.3 pp, or the family's unit cap.
- **M6 Confirmation (local):** top 5 × 3 seeds (seeds 1-3); the finalist × 5 seeds with saved models; the record-format
  50k run; the bench against B0; the test set opened once per final model; CIFAR-10.1 once; the ablation of each lever
  in the final recipe (3 paired seeds). **Push notification to the user when the target is Validated.**
- **M7 The MAC/accuracy front (G4, ~6 units, if ≥ 10 units remain):** a two-objective study (validation, MACs at one
  view) over the equivariant family under the cap; the Pareto set retrained over 3 seeds; reported as a table.
- **M8 Write-up:** `RESULT.md` in this session's format; the method result with its curves; nulls listed with settings.

**Stop rules.** The unit cap. M2's readout failing with no agreed revision. M3 null and M4 adding < 0.3 pp over the
finalist (then M5 is not run; the write-up reports the ceiling). Any VM job under 80 % GPU busy twice.

## 7. Order and dependencies

```
M0 ─┐
M1 ─┴─> M2 (G4, epoch curve) ─readout─> M3 (G4, H1 curves) ─readout─> M4 (local screens) ─> M5 (G4 search)
                                                                                              └─> M6 (local confirm, bench, test opened once) ─> M7 (if units) ─> M8
```

M1 can start now; M2 needs M1's `test='sealed'` and the matched-parameter twin only; M3 needs the p4m layer and
BlurPool; M4 needs the rest of M1. Nothing on the G4 before the user confirms the budget.

## 8. Lessons from the last session, applied here

- Search on two seeds; retrain the top k on fresh seeds; never read a one-seed search value as a result.
- Keep every job's label under the session (`SOLVEQ_SESSION`), the watcher up whenever jobs exist, `--gpu-mb 7000`
  on the 4060, teacher and data directories passed as absolute paths, and a changed argument on any relaunch (the
  queue de-duplicates identical calls, including cancelled ones).
- Record the Colab balance at the start and end of every VM job; the card's stored rate is a check, not the truth.
- Matched MACs beside matched parameters in every comparison, since the one validated method traded MACs for views.
- Stop a lever after 3 seeds inside ± 0.10 pp; do not re-run a null at another weighting.
