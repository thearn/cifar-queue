# Plan: mode 2 as efficient as it can be (a whole search on one VM)

Written 2026-10-29, at the user's request ("make sure that's as efficient as we can make it"; "make sure ASHA/hyperband
work as part of that effort"). Mode 2 is one queue job on one Colab VM: Optuna (or Ray Tune) runs there, its trials
as processes on the VM's GPU (`tune.on_vm`, `solve_queue.parallel.GPUProcessExecutor`).

Efficiency here means three things, measured separately:

1. **The card kept busy**: the GPU's busy fraction over the job (`usage.json`, solve-queue's sampler), and training
   time over wall time.
2. **Trials an hour and units a trial** on each card (L4, G4, A100), solve-queue's `cards.py` rates.
3. **The search's yield**: the best validation accuracy reached for a given number of units. Early stopping (ASHA,
   Hyperband) moves this one most: a trial that is plainly worse after 3 of 24 epochs stops there.

## Where it stands

Mode 2 on an L4, 2026-10-27 (an older, narrower space): 60 trials in 40 min, 89 an hour, 97 % GPU busy, best 92.8 %
validation and 91.8 % test. The calibration ran 1 and 2 trainings at once (48 s): 2 gave no gain, so one at a time.
Trials run to their full epochs whatever their curve; nothing prunes. `hpo.py` says so: "pruning on intermediate
values is not passed through (a trial reports when it ends)".

Mode 3 (trials as queue jobs), the same day as this plan: 85 % training time on an L4 in steady state with
solve-queue's persistent worker (69 % without); about 12 s lost a trial (8.5 s inside the job, 3.4 s between jobs).
With two trials in flight on the L4 taking turns at the card (`solve_queue.gpu.device_turn`), 94 % training time, 53
trials an hour (16 trials in steady state; images per second at each width unchanged, so the turns held). Mode 2's
pool can use the same turns (step 3).

## Steps, in order of value

1. **Early stopping: ASHA and Hyperband**, the largest lever.
   - The trial reports validation accuracy after each epoch and asks whether to stop.
   - In mode 2 the trials are processes on the same VM as the study's SQLite file, so a trial opens the study itself
     (`optuna.load_study`, then `trial.report` and `should_prune` through the `Trial` with its number), passed
     `study_name`, `storage` and `trial_number` by the driver.
   - The driver chooses the pruner: `SuccessiveHalvingPruner` (ASHA), `HyperbandPruner`, or the median pruner, with
     the epoch as the step.
   - Ray Tune's own `ASHAScheduler` and `HyperBandScheduler`, with `tune.report` per epoch, as the second driver (Ray
     installed in the cifarq venv, which rebuilds its layer once).
   - Acceptance: the same 60-minute budget on an L4, with and without each pruner. Trials completed or pruned, the best
     validation reached, and units used: pruning should reach mode 2's earlier best in well under half the time.
2. **The search's state survives a lost VM.**
   - `study.db` checkpointed to the bucket as it changes (`job.json` `checkpoint`), restored by the queue before a lost
     VM's rerun.
   - Optuna's heartbeat with `RetryFailedTrialCallback`, so trials in flight rerun.
   - Acceptance: a run whose VM is stopped mid-way continues on a new one with no completed trial lost.
3. **The pool's gaps.**
   - Each trial's set-up (the model, the data to the GPU, the CUDA context of a new process) idles the card.
   - Two pool processes taking turns at the card (`solve_queue.gpu.device_turn`, `SOLVEQ_DEVICE_TURNS=1`), so the next
     trial's set-up overlaps the current one's training.
   - The arrays cached per process (done, `data.load`).
   - Acceptance: GPU busy above 95 % on the L4 with a mixed space, and trials an hour up by the set-up time's share.
4. **The card by cost.**
   - The calibration on a G4 (102 GB, 48 vCPUs): a small model leaves a big card idle, so several trainings at once
     should pay there, unlike the L4.
   - Recorded with `cards.set_slots` and reused, not re-run each time (48 s a run).
   - Units a trial on L4, G4 and A100 compared at the same space; the batch-size range widened if the big cards favour
     it (the queue's GPU plan, step 5).
   - Acceptance: a stored count per card, and the cheapest card for mode 2 named with its units an hour.
5. **What the desktop sees.**
   - `job.json` `report: [best_value, trials, pruned]`, so `wait`, `news` and `brief` show the search's state.
   - One line per finished or pruned trial in the log, for `wait --grep`.
   - `progress.json` kept as it is.
6. **The refined mode 2 on every card**, once steps 1-5 are in: the same search on each card Colab offers.
   - **Cards:** L4, T4, A100 40 GB, A100 80 GB (`--high-mem`), G4, and the H100 if it stops being refused.
   - **Setup per card:**
     - each card's own calibration, recorded with `cards.set_slots`;
     - the best pruner from step 1;
     - the same space;
     - a cold start with two seeds;
     - a fixed budget: 45 minutes of search per card, after the VM is ready.
   - **Measured per card:**
     - trials an hour (completed and pruned);
     - GPU busy from `usage.json`;
     - units a trial;
     - the best validation reached within 5, 10 and 20 units.

     The last is the number that decides which card mode 2 should use.
   - **Cost:** about 40-50 units in all with two seeds at the rates in CARDS.md: G4 ~6.7, A100 80 GB ~5.1, A100 ~4.0,
     L4 ~1.2, T4 ~0.8 a run of 45 minutes, plus start-up.
   - **Order:** two cards at a time (the account holds two GPU VMs).
   - **Results:** added to CARDS.md beside the openEMS figures, the cheapest card for this kind of job recorded in
     `cards.json` notes, and the README's results table.
   - **Acceptance:** a table that names the card with the best validation per unit and the one with the most trials an
     hour, and what each costs.
7. **Optional, later: population-based training as a second stage** (the user, 2026-10-29).
   - **The method.** Ray Tune's `PopulationBasedTraining` and `PB2` (Parker-Holder et al., 2020: a Gaussian-process
     bandit in place of random perturbation, for small populations). Optuna has no equivalent: a trial cannot take over
     another's weights.
   - **What it tunes.** With the architecture fixed by the search's best (members copy each other's weights, so width,
     activation, head and whitening must be the same):
     - learning rate, weight decay, augmentation strengths, label smoothing, mix probability;
     - their schedules over the run.
   - **Where.** On a G4, whose calibration should allow several members at once. Not on an L4, which runs one training
     at a time best, so a population would wait its turn.
   - **Prerequisites:** `train` saving and loading its state (model, optimiser, epoch) for Ray's checkpoints;
     schedule-free training suits it (no fixed end to break when a member's settings change).
   - **Acceptance:** test accuracy over 3-5 seeds against the search's best settings at the same units.
   - **Not across VMs** through the queue: members exchange checkpoints during training, which wants one Ray cluster,
     not jobs on separate VMs.
8. **Later: several machines on one study** (mode 3's successor).
   - Optuna's gRPC storage proxy (`run_grpc_proxy_server`, `GrpcStorageProxy`, in Optuna 5.0 here) runs on the desktop.
   - It reaches each VM through an SSH reverse forward on the queue's one connection to it.
   - Each VM runs one long queue job of N Optuna workers on that storage, with native pruning and no per-trial job.
   - The queue leases and stages the VMs, samples them, and replaces a lost one.
   - Per-trial jobs stay for long trials (hours each), where the queue's per-trial records earn their cost.

## What the first L4 comparison found, and what the literature offers (2026-10-29)

**The finding.** The same 30-minute search on two L4s: no pruning reached 93.58 % validation with 21 trials; Optuna's
ASHA ran 31 trials (8 completed, 23 pruned) and reached 91.42 %. ASHA pruned trial 0, Page's recipe, which completes
at 93.6 %, at 33 % of its training (77.5 % there, against another trial's 82 %). Its schedule holds a high learning
rate through the middle and gains most of its accuracy as the rate decays at the end, so early accuracy ranks it low.
This is the "slow starter" problem of successive halving, and Hyperband and Ray's ASHA use the same signal.

**Fixes, by where they act** (sources at the end):

| Fix | What it does | Fits here because | Cost |
|---|---|---|---|
| A first rung from data | rank correlation of accuracy at each percent of training against final accuracy, from completed curves; no pruning before it is high | we have full curves (the no-pruning run) | small: solve-queue `hpo` |
| Soft ranking (PASHA, ICLR 2023) | ties within ε (0.025 suggested) count as equal at a rung; the maximum budget grows only while the top ranks keep changing | SGD noise between close configurations; PASHA reports ~1/3 of ASHA's time at no accuracy loss | small: a pruner in solve-queue |
| Best-so-far smoothing | report the running maximum of validation accuracy | high-rate epochs jump around | trivial |
| A schedule-free optimizer (Defazio et al., NeurIPS 2024; `schedulefree`) | evaluate the averaged iterate: a model that is good at any point, with no decay phase to wait for | removes the cause: early accuracy then predicts final accuracy | moderate: an optimizer option in `train`, and a check that it trains as well |
| Recurring learning rate (MORL, WACV 2023) | each successive-halving rung trains with a schedule that decays to its own end, re-warmed at the next | made for exactly this slow-starter case with CNNs | larger: trials resumable rung by rung |
| Learning-curve extrapolation (LC-PFN, NeurIPS 2023) | a pre-trained transformer predicts the rest of a curve from its start in one pass; stop a trial when it is unlikely to beat the best | 2-6x speed-ups on 45 datasets at negligible overhead; a pruner like any other | moderate: an Optuna pruner wrapping `lcpfn` |
| Freeze-thaw BO (ifBO / FT-PFN, ICML 2024) | pause and resume trials, choosing which to continue from curve extrapolations | state of the art on deep-learning HPO benchmarks | larger: resumable trials and a different driver |
| PriorBand (NeurIPS 2023, NePS) | Hyperband that samples around an expert prior, recovering if the prior is bad | Page's recipe is a strong prior | moderate: another driver |
| One epoch, then top-K ("Is One Epoch All You Need", 2023) | train all briefly, keep the best few | the baseline every multi-fidelity method should beat; here it should fail on slow starters, which tells us how much the schedule matters | trivial |

**The calibrated pruners on the L4 (2026-10-29, same space, 30-minute budget each).** The best trial of every run
that kept it was Page's recipe (enqueued first, 93.6-93.8 % across runs: seed noise), so the runs are compared on
the best trial other than it and on how many others reached 92 %:

| Run | Trials | Best other than Page | Others >= 92 % |
|---|---|---|---|
| No pruning | 30 | 93.10 % | 4 |
| Optuna successive halving, 10/30/90 %, keep a third | 47 | 92.64 % (Page pruned) | 3 |
| `rungs` (`tune.PRUNE`): 10/30/60 %, keep two thirds, best so far, at least 6 a rung | 35 | 93.30 % | 7 |
| Ray `ASHAScheduler`, 30/60 %, keep half, best so far (**cold: no Page trial**) | 28 | 93.64 % (AdamW, CELU, BN head, whitening) | 6 |

The rung check's choice did what the simulation said: more good configurations than no pruning in the same time,
none of the good ones cut. Single runs: a few tenths between them are within seed noise; the counts are the
firmer sign. The Optuna runs began with Page's recipe as trial 0; the Ray run did not (the Ray driver passed no
starting points), and still found the best configuration other than Page's of any run. From now on both drivers start
from Page's recipe (Ray's `points_to_evaluate`) and summaries say which start they had.

**A new best, and a bug (2026-10-29).** A run meant as Optuna's successive halving at Ray's rungs found trial 30
(AdamW, SiLU, a batch-norm head, 20 epochs): **94.16 % validation, 93.95 % test**, above Page's recipe (93.58 / 93.25)
and the morning's best (93.66 / 93.43); its family (AdamW with a batch-norm head) also gave 93.96 % and 93.50 %, and the
cold Ray run's best. But the trials were pruned at 10-14 % of training: the pruner's settings had not reached the
trial processes, which rebuilt Optuna's pruner at its defaults (fixed: `make_pruner` keeps them). So that run was the
default successive halving again, which this time kept Page's recipe: how much pruning by early accuracy is luck.

**How comparisons are run from here (the user, 2026-10-29).**
- **Start cold:** no Page trial (`--cold`). A warm start's best is the known optimum found in the first minutes,
  which hides differences between methods and cards; a cold start is also what a new problem looks like.
- **Two seeds a configuration** (`--seed 0`, `--seed 1`): a cold start depends on its first random trials.
- **Measured:**
  - minutes (and units) to the first trial at 92 % and at 93 %;
  - the best found;
  - trials at 92 % or more.
- **Production and champion searches start warm** from Page's recipe.

**Order adopted.** Step 1 becomes:
1. Measure the rank correlation from the no-pruning run's curves, and set the first rung from it.
2. Add best-so-far smoothing and a soft-ranking ASHA (PASHA's rule) in solve-queue.
3. Re-run ASHA and Hyperband against no pruning.

Then:
- **Schedule-free training:** a schedule-free optimizer as a `train` option, with the correlation measured again. If early accuracy becomes predictive, pruning becomes safe, and the accuracy itself may improve.
- **LC-PFN:** an LC-PFN pruner, compared with the calibrated ASHA.
- **Freeze-thaw:** only if those leave a large gap, since it needs trials that can be paused and resumed.

**Sources.**
- [In-Context Freeze-Thaw BO (ifBO), ICML 2024](https://proceedings.mlr.press/v235/rakotoarison24a.html), [code](https://github.com/automl/ifBO)
- [PASHA, ICLR 2023](https://arxiv.org/pdf/2207.06940), [schedulers](https://github.com/ondrejbohdal/pasha/blob/master/docs/schedulers.md)
- [LC-PFN, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3f1a5e8bfcc3005724d246abe454c1e5-Abstract.html)
- [MORL, WACV 2023](https://arxiv.org/pdf/2209.12499)
- [Schedule-Free (code)](https://github.com/facebookresearch/schedule_free)
- [PriorBand, NeurIPS 2023](https://arxiv.org/pdf/2306.12370)
- [Is One Epoch All You Need](https://arxiv.org/pdf/2307.15422)
- [Deep Power Laws / scaling laws for HPO, NeurIPS 2023](https://arxiv.org/pdf/2302.00441)
- [DyHPO, NeurIPS 2022](https://arxiv.org/pdf/2202.09774)
- [Hyper-Tune, VLDB 2022](https://www.vldb.org/pvldb/vol15/p1256-li.pdf)
- [Optuna pruners](https://optuna.readthedocs.io/en/stable/reference/pruners.html)

## Results log (2026-10-29)

Every measurement so far, in the order taken. All searches 30 minutes on an L4 (asia-southeast1) unless noted, one
training at a time with two pool processes taking turns at the card.

**1. How well early accuracy predicts the final** (`scripts/rung_check.py`; 161 complete curves from mode 3 trials).

| Point in training | Rank correlation | Top-10 % trials cut by keeping a third (latest / best so far) | Keeping half (best so far) |
|---|---|---|---|
| 10 % | 0.58 | 6 / 6 of 17 | 0 |
| 20 % | 0.65 | 7 / 7 | 2 |
| 30 % | 0.65 | 11 / 6 | 0 |
| 40 % | 0.72 | 8 / 4 | 2 |
| 50 % | 0.77 | 6 / 4 | 1 |
| 60 % | 0.83 | 3 / 2 | 1 |
| 80 % | 0.95 | 1 / 1 | 0 |
| 90 % | 0.99 | 0 / 0 | 0 |

**2. Successive halving replayed on those curves** (`simulate_rungs`; best so far).

| Rungs, share kept | Compute | Top-10 % kept |
|---|---|---|
| 10/30/90 %, a third (Optuna's ASHA as first set) | 24 % | 5 of 17 |
| 10/20/40/80 %, half | 27 % | 5 |
| 20/50 %, half | 49 % | 12 |
| 30/60 %, half (Ray's ASHA as set) | 57 % | 12 |
| **10/30/60 %, two thirds (`tune.PRUNE`)** | 50 % | **15** |

**3. The searches** (warm: Page's recipe as trial 0, unless marked cold).

| Run | Trials | Best (val / test) | Best other than Page | Others >= 92 % |
|---|---|---|---|---|
| No pruning | 30 | 93.58 / 93.25 (Page) | 93.10 | 4 |
| Optuna successive halving, 10/30/90 %, a third | 47 | 92.64 / 92.27 | 92.64 (Page pruned at 33 %) | 3 |
| `rungs` (10/30/60 %, two thirds, best so far, 6 a rung) | 35 | 93.82 / 93.12 (Page) | 93.30 | 7 |
| Ray ASHA, 30/60 %, half, best so far, **cold** | 28 | 93.64 / 93.23 | 93.64 (AdamW, CELU, BN head) | 6 |
| Optuna successive halving meant at 30/60 %, run at its defaults (the settings bug) | 45 | **94.16 / 93.95** (trial 30) | 94.16 (AdamW, SiLU, BN head, 20 epochs) | not counted |
| `rungs`, schedule-free in the space | 35 | 93.32 / 93.17 (Page) | 93.08 | 5 (schedule-free trials: 8, 7 completed, 1 pruned, best 90.02) |

Readings: the calibrated pruners find more good configurations than no pruning in the same time; the default ASHA
cuts the slow starters, and whether it cuts Page's recipe is luck (one run did, one did not); AdamW with a batch-norm
head is the strongest family found (94.16 %, and the best of the cold Ray run).

**4. Schedule-free training against Page's schedule** (Page's recipe, 24 epochs, on the local 4060).

| Optimiser | Final val (test) | At 30 % | At 50 % | At 80 % |
|---|---|---|---|---|
| SGD, Page's schedule | 93.38 (93.23) | 75.8 | 85.1 | 91.5 |
| schedule-free SGD, lr 0.4 | 90.60 (90.04) | 82.7 | 86.5 | 89.8 |
| schedule-free SGD, lr 1.0 | 92.18 (91.88) | 87.2 | 89.4 | 91.5 |

Early accuracy far more telling (87 % at 30 % against 76 %); untuned, 1.2 points below the tuned schedule. In a search
(row 3, last), schedule-free trials were rarely pruned (as their flat curves suggest) and finished lower so far.

**5. Ray's ASHA against Optuna's successive halving, as implemented** (Ray 2.59, Optuna 5.0).

| | Ray `ASHAScheduler` | Optuna `SuccessiveHalvingPruner` |
|---|---|---|
| Rungs | `grace_period x reduction_factor^k` on the reported counter | `min_resource x reduction_factor^k` steps |
| Cut | below the `(1 - 1/rf)` percentile of values recorded at the rung so far | not in the top `1/rf` of trials that reached the rung |
| Early rungs | the first trial passes; the second is judged against one value | the same in effect; `bootstrap_count` prunes rather than waits |
| A minimum of trials before cutting | none | none (`RungPruner` adds one) |

**6. OptunaHub's Conformal Pruner against our rungs** (`scripts/conformal_check.py`; its rule, `surestop.KillRule`,
fitted on 100 of the 161 curves resampled to a 5 % grid of training, both rules scored on the other 61; 200 random
splits).

| Rule | Compute | Share of the test top 10 % cut |
|---|---|---|
| rungs 10/30/90 %, a third | 24 % | 70 % |
| conformal, alpha 0.03 | 41 % | 26 % |
| **rungs 10/30/60 %, two thirds (`tune.PRUNE`)** | **51 %** | **16 %** |
| **conformal, alpha 0.02** | **51 %** | **18 %** |
| conformal, alpha 0.0125 | 62 % | 9 % |
| conformal, alpha 0.05 / 0.1 / 0.2 (80 calibration curves) | 25 / 10 / 5 % | 43 / 88 / 100 % |
| conformal with `hold_until_resolved` 0.9 | 77-86 % | 8-24 % |

The two lie on the same trade-off: at half the compute they cut about the same share of the best trials. The
conformal guarantee bounds good-and-pruned trials over all trials, so protecting a top 10 % needs alpha near 0.02 or
lower (at 0.05-0.2 it cut most of them); it needs one step grid (resampling to percent of training made it work on
our mixed-length trials), ~100 calibration curves at that alpha, and exchangeable trials (TPE voids the guarantee).
Ours stays the default. The resampling and a replay of compute against good trials cut could be offered to its
author, but not yet: nothing is offered upstream (to Optuna, Ray or OptunaHub) until it holds on much larger
problems than this one (the user, 2026-10-29).

**Still running:** the cold comparisons, two seeds of no pruning, `rungs` and Ray's ASHA (schedule-free in the space),
on the two L4s until about 16:15.

## Where each change lives

| Change | Repository |
|---|---|
| per-epoch reports, the pruned trial's result, Ray Tune driver | cifar-queue (`train.py`, `tune.py`, scripts) |
| a trial process's handle on the study (report and prune from a pool process), pruner options in `optuna_optimize` | solve-queue (`hpo.py`): any project's mode 2 needs it |
| `device_turn`, the worker, checkpoints, `cards.set_slots` | solve-queue (done or in place) |
| the gRPC storage over SSH | solve-queue (`hpo.py` and the transport), later |
