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
A two-slot run with turns at the card is being measured.

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
     - the same space and seed;
     - a fixed budget: 45 minutes of search per card, after the VM is ready.
   - **Measured per card:**
     - trials an hour (completed and pruned);
     - GPU busy from `usage.json`;
     - units a trial;
     - the best validation reached within 5, 10 and 20 units.

     The last is the number that decides which card mode 2 should use.
   - **Cost:** about 20-25 units in all at the rates in CARDS.md: G4 ~6.7, A100 80 GB ~5.1, A100 ~4.0, L4 ~1.2, T4 ~0.8
     for 45 minutes each, plus start-up.
   - **Order:** two cards at a time (the account holds two GPU VMs).
   - **Results:** added to CARDS.md beside the openEMS figures, the cheapest card for this kind of job recorded in
     `cards.json` notes, and the README's results table.
   - **Acceptance:** a table that names the card with the best validation per unit and the one with the most trials an
     hour, and what each costs.
7. **Later: several machines on one study** (mode 3's successor).
   - Optuna's gRPC storage proxy (`run_grpc_proxy_server`, `GrpcStorageProxy`, in Optuna 5.0 here) runs on the desktop.
   - It reaches each VM through an SSH reverse forward on the queue's one connection to it.
   - Each VM runs one long queue job of N Optuna workers on that storage, with native pruning and no per-trial job.
   - The queue leases and stages the VMs, samples them, and replaces a lost one.
   - Per-trial jobs stay for long trials (hours each), where the queue's per-trial records earn their cost.

## Where each change lives

| Change | Repository |
|---|---|
| per-epoch reports, the pruned trial's result, Ray Tune driver | cifar-queue (`train.py`, `tune.py`, scripts) |
| a trial process's handle on the study (report and prune from a pool process), pruner options in `optuna_optimize` | solve-queue (`hpo.py`): any project's mode 2 needs it |
| `device_turn`, the worker, checkpoints, `cards.set_slots` | solve-queue (done or in place) |
| the gRPC storage over SSH | solve-queue (`hpo.py` and the transport), later |
