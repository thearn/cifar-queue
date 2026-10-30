# cifar-queue: results

The log of what was measured, newest last. Runs before 2026-10-29 evening were made with the `scripts/` layout (the
old names are in the table in `docs/PLAN-mode2.md`); their records are `results/<mode>/*.json` and the studies
`results/distributed-study.db` and `results/on-vm-study.db`. Runs after it are `results/<stage>/*.json`, each with its
commit and dataset hash. The two layouts train on the same images in the same order (dataset stage acceptance,
below), so their numbers compare.

## 2026-10-27/28: the three modes and the first champion

| Run | Hardware | Trials | Time | Best val | Test |
|---|---|---|---|---|---|
| Defaults, 16 epochs (mode 1) | one L4 | 1 | 37 s training | 88.9 % | 88.3 % |
| Page's recipe | RTX 4060 | 1 | 154 s training | 93.5 % | 93.1 % |
| Mode 2 (an older 44-parameter space, max 20 epochs) | one L4 | 60 | 40 min | 92.8 % | 91.8 % |
| Mode 3 | 4060 + two L4s | 80 | 36 min | **93.7 %** | **93.4 %** |
| Mode 3's best retrained, 5 seeds | RTX 4060 | 5 | 16 min | 93.2-93.8 % | 93.18 +- 0.16 % |

Page's own result is 94 %, trained on all 50000 images; here 5000 are held out for validation. Mode 3's best trial kept his architecture and optimiser and changed mostly epochs (28), learning rate (0.78), logit scale (0.066), and added a weight EMA. It found his region within about 30 trials, and the driver (the local script running Optuna) was killed twice during that run and resumed without losing a trial. Retrained over 5 seeds, the same parameters give 93.18 +- 0.16 % test, so the 0.3-point lead of the single trial over Page's single run is mostly seed noise: the search matched his recipe rather than beat it. Seed 0 reproduced the trial's validation accuracy exactly (93.66 %), and its network, the median seed by validation accuracy, is the published model (93.23 % test).

## 2026-10-29: mode 2 on each card, early stopping, schedule-free training

A cold search (no known good trial to start from), TPE with the pruner chosen from this model's own learning curves
(`tune.PRUNE`), 45 minutes on one VM (details: `docs/PLAN-mode2.md`, "Results log"):

| Card | Trials | Best val (test) | First 92 % | Units |
|---|---|---|---|---|
| G4 (RTX PRO 6000) | 155-177 | **95.12 % (94.59 %)** | ~2 min | ~7.0 |
| A100 40 GB | 81-97 | 94.72 % (94.49 %) | ~6 min | ~4.4 |
| L4 (30 minutes) | 36-38 | 93.80 % (93.29 %) | 13-18 min | ~0.8 |

The best configuration, schedule-free AdamW (CELU, width 48, batch 256, 30 epochs, 22 s a training on the G4),
retrains over three seeds to **94.57 % test** (94.48-94.65 %), above Page's recipe here (93.2 %, row 2 above). Early
stopping needed care: Optuna's and Ray's ASHA at their usual settings cut the best trials, because these schedules
gain most of their accuracy late; rungs chosen by replaying 161 complete curves kept them and found more good
configurations than no pruning in the same time. Population-based training on the best configuration gave nothing
more for runs this short.


## 2026-10-29 evening: the move to the template layout, checked on the 4060

Checks that the new layout runs and trains on the same data, not measurements: the 4060 was shared with another
session's job for part of them, so the times are not clean.

- **Dataset stage**: the raw parquet files' sha256 equal those of the old `data/MANIFEST.json`, and the dataset's six
  arrays (images and labels of train, validation, test) equal, array for array and in order, what the old `train()`
  drew from the old array file.
- **Defaults, 16 epochs** (`results/train/20261029-190348.json`): 88.6 % validation, 88.2 % test, 43 s; mode 1 gave
  88.9 % and 88.3 % on an L4.
- **The champion's seed 0 again** (`results/champion/20261029-190902.json`, from `distributed-study.db`'s best trial):
  93.64 % validation, 93.03 % test, against 93.66 % and 93.23 % on 10-28. Not bit-identical, and it cannot be: the old
  code at master, run the same day on the old array file with the 10-28 job's exact arguments, gave 93.28 % validation
  and 93.33 % test. This training is not bit-reproducible run to run on the 4060 (the first epoch's loss differs in
  the fourth digit), so the 10-28 run's exact match with its search trial was luck. Both runs lie within the
  champion's spread over seeds (93.2-93.8 % validation).
- **Mode 3** (`results/tune/20261029-192346.json`): 2 trials on the 4060 with `--prune rungs --cold`, 91.06 % best.
- **Resume**: a copy of `distributed-study.db` resumed with `--no-schedule-free` (30 trials to 31; the original untouched).
- **Mode 2 on the 4060** (`results/tune_on_vm/20261029-193227.json`): 2 trials, the study, progress and summary back
  with the job.
- **Publish**: `70_publish.py --dry-run` staged the model repo and card; nothing uploaded.
