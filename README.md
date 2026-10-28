# cifar-queue

CIFAR-10 with a small CNN, trained and tuned through the machine's solve queue
([solve-queue](../solve-queue)): the local GPU and Colab VMs as one pool of compute. A test problem for the queue's
machine-learning path: profiles, data, function jobs, and hyperparameter search at scale.

## The problem

- **Data:** CIFAR-10 from the Hugging Face hub ([uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10),
  its parquet files at a pinned revision), decoded once into `data/cifar10-<sha12>.npz` (uint8, 184 MB).
  `data/MANIFEST.json` (committed) records the revision and every file's sha256; the array file is not committed:
  jobs name it as their `data`, so it goes to the bucket once and to each VM once, by its content.
- **Split:** 45000 training images, 5000 held out (a fixed split) as the validation set the search optimises;
  the 10000 test images are reported, never tuned on.
- **Model:** a ResNet-9-style CNN (David Page's DAWNBench design; below) with a width multiplier (0.4-6.6 M
  parameters at width 16-64).
- **Training** (`src/cifarq/train.py`, every choice a hyperparameter, 0 or `none` turning a feature off): SGD
  (momentum, Nesterov) or AdamW (betas), decoupled weight decay, warmup then cosine, linear, exponential or no decay
  to a final fraction, label smoothing, gradient clipping, an EMA of the weights, dropout and the activation; and
  augmentation on the GPU, random per image: affine (translation, rotation, scale), flip, brightness, contrast,
  saturation, cutout, mixup or cutmix. bf16 under autocast, the images resident on the GPU.
- **Search space** (`src/cifarq/tune.py`; the full table below): 53 Optuna parameters, conditional (each optional feature a switch and,
  when on, a strength on a log scale down to almost no effect, so the search turns it down by degrees or off), epochs
  6-30 included. Multivariate TPE with a constant liar for trials in flight; Page's recipe is every search's first trial.

Defaults, 16 epochs, on a Colab L4: 88.9 % validation, 88.3 % test, 37 s of training.

## The model

A **ResNet-9-style convolutional network** (`src/cifarq/train.py`, `model`), the architecture David Page of Myrtle.ai
designed for the Stanford DAWNBench CIFAR-10 training-time competition (his "Custom ResNet 9" entry, November 2018:
94.08 % test accuracy in 75 s on one V100) and described in his "How to train your ResNet" blog series. It became
the standard baseline for fast CIFAR-10 training (Lambda's 94 % in about 100 s on one Turing GPU; the later
speedrun work, airbench, starts from the same family).

Layout, with `w` the width (Page's network is `w` = 64: 64-128-256-512 channels):

| Stage | Layers | Output (channels x size) |
|---|---|---|
| prep | conv 3x3 -> batch norm -> activation | w x 32x32 |
| layer 1 | conv 3x3 -> BN -> act -> max pool 2; residual block (2 x conv-BN-act, added to its input) | 2w x 16x16 |
| layer 2 | conv 3x3 -> BN -> act -> max pool 2 | 4w x 8x8 |
| layer 3 | conv 3x3 -> BN -> act -> max pool 2; residual block | 8w x 4x4 |
| head | global max pool -> dropout -> linear | 10 classes |

Eight 3x3 convolutions in all, each followed by batch normalisation and the activation. Parameters by width: 16:
0.41 M, 24: 0.93 M, 32: 1.65 M (the default), 48: 3.70 M, 64: 6.57 M (Page's size).

Where it differs from Page's network, every difference a hyperparameter whose values include his: the width (his is
64), the activation (relu, gelu, silu, celu; his baseline ReLU, his later versions CELU with alpha 0.075), dropout
before the classifier (his: none), the classifier head and the normalisation of its output (his: linear, logits x
0.125; also a cosine classifier and batch norm on the logits, each with its own scale), and an optional fixed input whitening layer
(from his later versions). His training choices (SGD with Nesterov momentum, the warmup-then-decay learning rate,
label smoothing, random crops, flips, cutout) and later additions (weight EMA, test-time flips) are in the search
space too. Not included from his later versions: ghost batch norm and frozen batch-norm scales.

## The search space

53 Optuna parameters (`src/cifarq/tune.py`, `space`). Many are conditional: a feature's strength is sampled only when
its switch is on, and each optimiser has its own parameters. "log" means sampled on a log scale; a strength's range
reaches down to where the feature has almost no effect, so the search can turn it down by degrees or off.

**Length and model**

| Parameter | Values / range |
|---|---|
| `epochs` | 6-30, step 2 |
| `width` | 16, 24, 32, 48, 64 (channels of the first layer; 0.4-6.6 M parameters) |
| `activation` -> `celu_alpha` | relu, gelu, silu, celu; CELU's alpha 0.05-1.0 (log) |
| `head` -> its scale | the classifier and how its output is normalised: `linear` (`use_logit_scale` -> `logit_scale`: off, or the logits x 0.03-2.0, log), `cosine` (normalised features and class weights, `cosine_scale` 2-50, log), `bn` (batch norm on the logits, no affine part, `bn_scale` 0.3-10, log) |
| `whiten` -> `whiten_eps` | off, or a fixed input whitening layer (27 filters from the 3x3 patch covariance), its epsilon 1e-4-0.1 (log) |
| `batch_size` | 256, 512, 1024 |
| `use_dropout` -> `dropout` | off, or 0.005-0.5 (log) |
| `use_label_smoothing` -> `label_smoothing` | off, or 0.002-0.3 (log) |

**Optimiser** (`optimizer`: sgd or adamw)

| Parameter | SGD | AdamW |
|---|---|---|
| learning rate | `sgd_lr` 0.005-0.8 (log) | `adamw_lr` 5e-5-1e-2 (log) |
| weight decay | `sgd_weight_decay` 1e-6-5e-3 (log) | `adamw_weight_decay` 1e-4-0.5 (log) |
| momentum / betas | `momentum` 0-0.97, `nesterov` on/off | `beta1` 0.7-0.97, `beta2` 0.98, 0.99, 0.999 |

**Gradient clipping and weight averaging**

| Parameter | Values / range |
|---|---|
| `clip` -> `grad_clip` | off, or gradient-norm clipping at 0.25-50 (log) |
| `ema` -> `ema_horizon` | off, or an EMA of the weights over 5-2000 steps (log; decay = 1 - 1/horizon, warmed up) |

**Learning rate schedule**

| Parameter | Values / range |
|---|---|
| `warmup_frac` | 0-0.5 of all steps |
| `warmup_start_frac` | 0.005-1.0 of the peak (log) |
| `decay` | cosine, linear, exponential, none |
| `final_lr_frac` | 0.001-0.3 of the peak (log; not sampled when `decay` is none) |

**Augmentation** (on the GPU, random per image)

| Parameter | Values / range |
|---|---|
| `use_translate` -> `aug_translate` | off, or 0.1-6 px (log) |
| `use_rotate` -> `aug_rotate` | off, or 0.2-25 degrees (log) |
| `use_scale` -> `aug_scale` | off, or +-0.2-30 % (log) |
| `aug_flip` | flip probability 0-0.5 |
| `use_brightness` -> `aug_brightness` | off, or +-0.5-50 % (log) |
| `use_contrast` -> `aug_contrast` | off, or +-0.5-50 % (log) |
| `use_saturation` -> `aug_saturation` | off, or +-0.5-60 % (log) |
| `use_cutout` -> `cutout_size`, `cutout_prob` | off, or a 2-16 px square (step 2) with probability 0.05-1.0 (log) |
| `mix` -> `mix_alpha`, `mix_prob` | none, mixup or cutmix; Beta alpha 0.05-2.0 (log), probability per batch 0.05-1.0 (log) |

**Evaluation**

| Parameter | Values / range |
|---|---|
| `tta_flip` | off, or each image's logits averaged with its mirror's (validation and test) |

**Why the output normalisation matters:** a batch-normalised network's final linear layer starts out producing large
logits, so the softmax saturates and the classifier's gradients are large early in a short run. Scaling the logits
down (Page's 0.125: a softmax temperature, and a lower effective learning rate for the classifier), normalising
features and weights (a cosine classifier) and batch-normalising the logits all keep their magnitude in hand; at 4
epochs each gave about +10 points over an unscaled linear head (70.1 % to 79.8 % linear x 0.125, 81.7 % cosine x 20,
80.9 % batch norm), so the space has the three methods and their scales rather than the one value.

**Page's recipe as a point of the space** (`tune.PAGE`, enqueued as the first trial of both searches): width 64,
24 epochs, batch 512, SGD with Nesterov momentum 0.9, peak learning rate 0.4 (linear warmup over 5 of 24 epochs from
about 0, then linear decay to about 0), weight decay 5e-4, label smoothing 0.2, logits x 0.125, 4-pixel translation
(reflect padding), flip 0.5, 8x8 cutout on every image; no dropout, whitening, EMA, clipping, mixing or test-time
flips. The search then has the published reference measured on this problem before it explores.

**Fixed:** the split (45000 train, 5000 validation, 10000 test), the objective (validation accuracy), bf16, seed 0,
and the sampler: multivariate TPE over the conditional groups with a constant liar, after 16 random trials. Every
parameter of `train.train` has a default (`train.DEFAULTS`), so a single run names only what it changes.

## Three ways to run it

```bash
uv venv ~/.venvs/cifarq --python 3.12          # a uv (or conda) Python: a VM cannot rebuild the system's
uv pip install --python ~/.venvs/cifarq/bin/python -e . -e ../solve-queue[gcs]
~/.venvs/cifarq/bin/python scripts/prepare_data.py
~/.venvs/cifarq/bin/python scripts/1_single.py                     # one point, fixed hyperparameters, a Colab L4
~/.venvs/cifarq/bin/python scripts/2_optuna_on_vm.py --trials 60    # Optuna inside one job on an L4
~/.venvs/cifarq/bin/python scripts/3_optuna_distributed.py --trials 80   # Optuna here; this GPU and two L4s
```

1. **One point** (`1_single.py`): `QueueExecutor(target='colab:cifar-l4a').submit('cifarq.train:train', ...)`. The
   executor's VM is allocated by the job, staged with this venv's profile (cached layers), sent the code and the
   data, and released when idle.
2. **Optuna on the VM** (`2_optuna_on_vm.py`): one queue job runs `cifarq.tune.on_vm` on an L4. It first measures
   how many trainings the GPU runs at once before its total throughput stops growing
   (`solve_queue.parallel.calibrate`), then keeps that many trials in flight as processes on the GPU
   (`GPUProcessExecutor`). The study (`study.db`) comes back in the job's directory.
3. **Optuna here, trials everywhere** (`3_optuna_distributed.py`): the study runs on this machine; each trial is a
   queue job, spread over `targets={'local': 1, 'colab:cifar-l4a': k, 'colab:cifar-l4b': k}`: each GPU runs as many
   at once as raise its throughput (`k` from mode 2's calibration; the 4060 gains 5 % from two, so one). Colab lets
   the account hold two GPU VMs, so run it after mode 2; a trial whose VM cannot be had moves to another target.

## Results (2026-10-27)

| Run | Hardware | Trials | Search time | Best validation | Its test | Notes |
|---|---|---|---|---|---|---|
| Defaults, 16 epochs (mode 1) | one L4 | 1 | 139 s from submission (37 s training) | 88.9 % | 88.3 % | |
| Page's recipe (reference) | RTX 4060, local | 1 | 154 s training | 93.5 % | 93.1 % | his DAWNBench baseline: 94 % on 50000 training images |
| Mode 2: Optuna on the VM | one L4, 1 trial at a time (calibrated) | 60 | 40 min | 92.8 % | 91.8 % | the earlier 44-parameter space, epochs at most 20 (its best sat at 20) |
| Mode 3: Optuna here | RTX 4060 + two L4s, 1 trial each | 80 (27 on the 4060, 53 on the L4s) | 36 min (84 an hour) | **93.7 %** | **93.4 %** | the 53-parameter space; driver killed twice and resumed, no trial lost |

Mode 3's best (trial 77) kept Page's skeleton (width 64, SGD, batch 512, ReLU, linear head) and changed 28 epochs, a
learning rate of 0.78, logits x 0.066, label smoothing 0.25, a weight EMA, and 4 px cutout. The search converged on his
region of the space within its first 30 trials (four of its top five are width 64, 24-30 epochs, SGD at a high rate,
no mixing) and beat his recipe by 0.5 points of test accuracy: real but within what seeds move, so worth confirming over
several seeds before calling it better. Calibration found one training at a time best on both GPUs: two at once gave
0 % (L4) and 5 % (4060) more throughput.

## Watching a search as it runs

Both searches keep their state in an Optuna study (SQLite), so the same tools read either while it runs:

```bash
~/.venvs/cifarq/bin/python scripts/watch.py            # mode 3: results/distributed-study.db, refreshed every 30 s
~/.venvs/cifarq/bin/python scripts/watch.py --mode2    # mode 2: the running job's study.db
~/.venvs/cifarq/bin/python scripts/watch.py --once     # one report and exit
uvx optuna-dashboard sqlite:///results/distributed-study.db                       # charts in a browser
uvx optuna-dashboard sqlite:///results/jobs/on-vm/<job>/study.db                  # (mode 2's)
```

`watch.py` shows the trials by state (complete, running, failed) and by device, trials an hour, the best trial (its
validation and test accuracy, training time, device and every parameter) and the top five. It reads a copy of the
database (SQLite's backup API), so it never holds a lock the search needs. `optuna-dashboard` runs from its own tool
environment (`uvx`): installing it in this venv would change the venv's cached layer and rebuild it on the VMs.

How the data gets here:

- **Mode 3:** the study is written on this machine as each trial's job ends, so it is always current.
- **Mode 2:** the study lives on the VM until the job ends. The job's `job.json` names `watch` files
  (`progress.json`, `study.db`), which solve-queue brings back from the running job every two minutes
  (`SOLVEQ_WATCH_S`); `cifarq.tune.on_vm` rewrites `progress.json` (trials done, best so far) after every trial, and
  `2_optuna_on_vm.py` prints it as it changes. The job's log also comes back once a minute as
  `results/jobs/on-vm/<job>/campaign.log.tail` (one line per trial: its value, its device, the best so far).
- Every job, either mode: `solveq list` (state, target, label), `solveq colab timings` (transfers and stagings), and
  the job's own directory under `results/jobs/` (its call, log and, once done, result).

Results go to `results/` (not committed): each job's directory (its call, log and result) under `results/jobs/`,
summaries as JSON, the studies as SQLite (`optuna-dashboard sqlite:///results/distributed-study.db`).

## Resuming a search that died

The study (SQLite) is the search's record, and every trial records its job's directory as it is submitted
(`solve_queue.hpo.optuna_optimize`). Start the driver again on the same study and it goes on:

```bash
~/.venvs/cifarq/bin/python scripts/3_optuna_distributed.py --resume --trials 40      # the latest study, 40 more trials
~/.venvs/cifarq/bin/python scripts/3_optuna_distributed.py --resume distributed-20261027-204644 --trials 40
~/.venvs/cifarq/bin/python scripts/2_optuna_on_vm.py --resume --trials 40            # mode 2: a new job, the old study
```

- **Mode 3:** trials the dead driver left in flight are reattached to their jobs (`QueueExecutor.attach`): a job that
  finished meanwhile is told its result, one still queued or running is followed, one lost (with its VM) fails and its
  parameters are queued again, so no point is dropped and no finished work is redone. Tested: the driver killed with
  three trials running on the 4060 and both L4s; the resumed driver told all three their results and went on.
- **Mode 2:** the study lives on the VM; the copy its job brought back last (every two minutes) is frozen into
  `results/resume/` and sent with a new job as its data (`on_vm(resume_from=...)`), which continues it; trials that were
  running on the dead VM run again.
- Completed trials are never re-run: TPE refits from the whole history, so a resumed search samples as the
  uninterrupted one would have.

## References

- D. Page, "How to train your ResNet" (Myrtle.ai blog series, 2018-2019), and his DAWNBench entry:
  [DAWNBench CIFAR-10 training](https://dawn.cs.stanford.edu/cifar10-training),
  [Myrtle.ai's announcement](https://www.cambridgenetwork.co.uk/news/myrtleai-sets-new-deep-learning-standard)
- [Lambda: ResNet9, 94 % CIFAR-10 in 100 seconds on a single Turing GPU](https://lambda.ai/blog/resnet9-train-to-94-cifar10-accuracy-in-100-seconds)
- K. Jordan, [94 % on CIFAR-10 in 3.29 seconds on a single GPU](https://arxiv.org/abs/2404.00498) (airbench, 2024)
- Typical accuracies: a basic CNN without augmentation 70-80 %; a tuned ResNet-9 with augmentation 93-94 % in
  10-25 epochs; speedrun methods 94-96 %; large pretrained vision transformers over 99 %. This repository's defaults
  give 88.3 % test in 16 epochs (37 s on an L4): the remaining gap is mostly epochs and the speedrun methods
  (whitening, test-time flips, alternating flips), not compute.

## How the compute is managed

- **Environment:** this venv is the jobs' profile (solve-queue `profiles.py`): its base Python (uv's, sent once as
  files) and its packages (`pip freeze`) as a cached layer in the bucket, built once, pulled by every later VM; its
  editable installs (solve-queue) sent with the staging. A VM shared with other projects holds each one's venv.
- **Data:** the array file is the jobs' `data`: a content-addressed blob in the bucket, placed once per VM.
- **Code:** `src/` is each job's `send`: changed files only.
- **Search:** `solve_queue.hpo` drives any ask/tell optimiser over any executor (`optuna_optimize`,
  `ray_searcher_optimize` for Ray Tune's searchers): the optimiser stays where it runs, the trials go to the queue
  (`futures.QueueExecutor`) or to a GPU process pool (`parallel.GPUProcessExecutor`).
