# cifar-queue

This repository is a test problem for solve-queue, my compute queue library (not public at the moment). solve-queue is a thin wrapper that takes local and cloud GPUs and CPUs from a mix of sources (here a local RTX 4060 and Colab VMs) and presents them as one compute pool, so that HPC and ML workflows can be submitted as ordinary function calls without caring where they run. It handles environment and code sync, sending only what changed to storage buckets in the same cloud region as the compute to avoid egress costs, and copying between regions only when a job needs it. It is close to a mix of existing queue managers, but handles my own particular needs.

I plan to push and update my test models publicly. The code is on GitHub ([thearn/cifar-queue](https://github.com/thearn/cifar-queue)) and the trained model on the Hugging Face hub ([thearn/cifar10-resnet9](https://huggingface.co/thearn/cifar10-resnet9)), which holds a snapshot of this repository at each publish.

This one is a small CNN trained on CIFAR-10 and tuned with Optuna. It was chosen because it has a well-known reference benchmark, and it exercises everything an ML workload needs from the queue: shipping a Python environment and a dataset to remote GPUs, running training as function calls, and spreading a hyperparameter search over several GPUs at once.

## Using the model

The model and training code need no queue. `pip install` from either location, then:

```python
from cifarq.hub import ResNet9            # pip install git+https://github.com/thearn/cifar-queue
net = ResNet9.from_pretrained('thearn/cifar10-resnet9')
probs = net.predict(images)              # uint8 (n, 32, 32, 3) RGB -> (n, 10); net.classes names the columns
```

To train one model yourself (a CUDA GPU is needed; this example, the defaults at 16 epochs, trains in under a minute on an L4 and reaches about 88 %, while the published settings take about 3 minutes on an RTX 4060):

```bash
pip install -e '.[train]'
python scripts/prepare_data.py           # download CIFAR-10 from the hub into data/
python -c "from cifarq.train import train; print(train(epochs=16, save='model')['test_acc'])"
```

`space/app.py` is a small Gradio demo (upload an image, get the top five classes), run locally with `python space/app.py`.

## Problem

The data is CIFAR-10 from the Hugging Face hub ([uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10), at a pinned revision), decoded once into a 184 MB `.npz` whose hashes are recorded in `data/MANIFEST.json`. Training uses 45000 images, the search optimises accuracy on a fixed 5000-image validation split, and the 10000 test images are only reported, never tuned on.

The model is a ResNet-9 in the style of David Page's DAWNBench entry (94 % test in 75 s on a V100, 2018), with a width multiplier covering 0.4-6.6 M parameters. Every training choice in `src/cifarq/train.py` is a hyperparameter with a default (`train.DEFAULTS`), and `src/cifarq/tune.py` defines a conditional search space of 53 parameters: optimiser (SGD or AdamW), schedule, regularisation, GPU-side augmentation, classifier head and logit scaling, epochs 6-30. Optional features are a switch plus a log-scaled strength reaching down to almost no effect, so the search can turn them down gradually instead of only on or off. The sampler is Optuna's multivariate TPE, with a "constant liar" so that trials running in parallel are not proposed twice, and Page's recipe (`tune.PAGE`) is always the first trial.

Note that the logit scaling matters more than it looks. At 4 epochs, scaling the logits by 0.125, using a cosine classifier, or batch-normalising the logits each gave roughly +10 points over a plain linear head (70.1 % to 80-82 %), which is why the space has all three.

## Running on the queue

The scripts below need solve-queue, so they document how the results were produced rather than something you can run today. A few of its terms are used throughout. A **job** is one Python function call (here `cifarq.train.train` with its hyperparameters) that the queue runs on whichever machine it places it: the local GPU, or a Colab VM it starts and stops as needed (an **executor**; two L4 executors, `cifar-l4a` and `cifar-l4b`, are used here). Each job gets its own directory, and whatever the function writes there comes back when it finishes. The queue copies the submitting Python environment to a VM once and caches it, sends only changed source files with each job, and sends the dataset file once per VM, keyed by its hash.

The same search is run in three ways, numbered by script, to test the different paths through the queue:

- **Mode 1** (`1_single.py`): one training at the defaults, as one job on a Colab L4. This tests the basic round trip.
- **Mode 2** (`2_optuna_on_vm.py`): the whole Optuna (or Ray Tune) search runs inside a single job on one VM. It uses the card's stored concurrency, or measures how many trainings the GPU can run at once before total throughput stops growing and stores it. Because the trials run beside the study, they can be stopped early (`--pruner rungs`, chosen from this model's curves). The study comes back with the job and is checkpointed while it runs. On a G4 it is the fastest and best way to search here (`docs/PLAN-mode2.md`).
- **Mode 3** (`3_optuna_distributed.py`): Optuna runs on the local machine, and each trial is a separate job spread over the RTX 4060 and both L4s at once. This is the main use case: one search over a mixed pool.

Two more scripts follow the search: `4_champion.py` retrains mode 3's best trial over several seeds, saving each network, and `publish.py` pushes the code and the chosen network to the hub.

```bash
uv venv ~/.venvs/cifarq --python 3.12
uv pip install --python ~/.venvs/cifarq/bin/python -e '.[train]' -e ../solve-queue[gcs]
~/.venvs/cifarq/bin/python scripts/prepare_data.py
~/.venvs/cifarq/bin/python scripts/1_single.py                          # mode 1: one training on a Colab L4
~/.venvs/cifarq/bin/python scripts/2_optuna_on_vm.py --trials 600 --minutes 45 --pruner rungs --executor cifar-g4 --gpu G4
                                                                        # mode 2: a whole search in one G4 job, early stopping
                                                                        # (--driver ray, --cold --seed N for comparisons)
~/.venvs/cifarq/bin/python scripts/rung_check.py                        # where early stopping is safe, from complete curves
~/.venvs/cifarq/bin/python scripts/5_pbt_g4.py                          # probe (seeds, batches, concurrency) and PBT on a G4
~/.venvs/cifarq/bin/python scripts/3_optuna_distributed.py --trials 80   # mode 3: trials over the 4060 and two L4s
~/.venvs/cifarq/bin/python scripts/4_champion.py --seeds 5              # mode 3's best retrained, networks saved
~/.venvs/cifarq/bin/python scripts/publish.py --no-space                # code and network to the hub
```

Use a uv or conda Python, since a VM cannot rebuild the system one. One training at a time was best on the L4, the A100 and the G4 (two at once gave 6 % less on the G4); mode 2 runs two processes taking turns at the card instead, so one trial's set-up overlaps the other's training. To follow a run, `solveq wait ID --grep trial` prints each trial as it ends (the solve-queue skill). Colab allows two GPU VMs per account, so run mode 3 after mode 2 has released its VM; `4_champion.py --targets local` uses the local GPU only, for when the VMs are busy with other work.

Trials return only their numbers, so no weights are kept from the search (80 networks at 26 MB would mostly be thrown away). Instead `4_champion.py` trains the best parameters again with `train(save='model')`, which writes the network into the job's directory, and publishes the seed with the median validation accuracy. `publish.py` refuses to run with uncommitted changes, and its hub commit names the local commit. The Gradio app in `space/` is not hosted, since a Gradio Space now needs a paid Hugging Face plan or ZeroGPU.

## Results (2026-10-27/28)

| Run | Hardware | Trials | Time | Best val | Test |
|---|---|---|---|---|---|
| Defaults, 16 epochs (mode 1) | one L4 | 1 | 37 s training | 88.9 % | 88.3 % |
| Page's recipe | RTX 4060 | 1 | 154 s training | 93.5 % | 93.1 % |
| Mode 2 (an older 44-parameter space, max 20 epochs) | one L4 | 60 | 40 min | 92.8 % | 91.8 % |
| Mode 3 | 4060 + two L4s | 80 | 36 min | **93.7 %** | **93.4 %** |
| Mode 3's best retrained, 5 seeds | RTX 4060 | 5 | 16 min | 93.2-93.8 % | 93.18 +- 0.16 % |

Page's own result is 94 %, trained on all 50000 images; here 5000 are held out for validation. Mode 3's best trial kept his architecture and optimiser and changed mostly epochs (28), learning rate (0.78), logit scale (0.066), and added a weight EMA. It found his region within about 30 trials, and the driver (the local script running Optuna) was killed twice during that run and resumed without losing a trial. Retrained over 5 seeds, the same parameters give 93.18 +- 0.16 % test, so the 0.3-point lead of the single trial over Page's single run is mostly seed noise: the search matched his recipe rather than beat it. Seed 0 reproduced the trial's validation accuracy exactly (93.66 %), and its network, the median seed by validation accuracy, is the published model (93.23 % test).

## Results (2026-10-29): mode 2 on each card, early stopping, schedule-free training

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

## Watching and resuming

```bash
~/.venvs/cifarq/bin/python scripts/watch.py            # mode 3, refreshes every 30 s
~/.venvs/cifarq/bin/python scripts/watch.py --mode2    # mode 2's study, as last copied back
uvx optuna-dashboard sqlite:///results/distributed-study.db
```

`watch.py` reads a backup copy of the SQLite study, so it never holds a lock the search needs. Run `optuna-dashboard` through `uvx` rather than installing it in the venv, since a change to the venv makes every VM reinstall it. In mode 2 the study lives on the VM, and solve-queue copies `progress.json` and `study.db` back every two minutes.

To resume a dead search, pass `--resume` (optionally with a study name) to the mode 2 or mode 3 script. In mode 3, trials left in flight are reattached to their queue jobs: finished jobs report their result, running ones are followed, and lost ones are re-queued with the same parameters. In mode 2 the last copied study is sent to a new job, and only the trials that were running on the dead VM are repeated. Completed trials are never re-run, and since TPE refits on the full history the resumed search samples as the uninterrupted one would have.

All output goes to `results/` (not committed).

## References

- D. Page, "How to train your ResNet" (Myrtle.ai, 2018-2019); [DAWNBench CIFAR-10](https://dawn.cs.stanford.edu/cifar10-training)
- [Lambda: ResNet9 to 94 % in 100 s on one Turing GPU](https://lambda.ai/blog/resnet9-train-to-94-cifar10-accuracy-in-100-seconds)
- K. Jordan, [94 % on CIFAR-10 in 3.29 seconds on a single GPU](https://arxiv.org/abs/2404.00498) (airbench, 2024)
