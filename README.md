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

To train one model yourself (the defaults at 16 epochs train in under a minute on an L4 and reach about 88 %, while the published settings take about 6 minutes on an RTX 4060; without a GPU it runs on the CPU, slowly):

```bash
pip install -e '.[train]'
python pipeline/10_ingest.py && python pipeline/20_clean.py --targets direct && python pipeline/30_dataset.py
python -c "from cifarq.train import train; print(train(epochs=16, save='model')['test_acc'])"
```

`app/app.py` is a small Gradio demo (upload an image, get the top five classes), run locally with `python app/app.py` (`MODEL=path/to/model` for a network of your own).

## Problem

The data is CIFAR-10 from the Hugging Face hub ([uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10), at a pinned revision), decoded once into a 184 MB `.npz` by the data stages, whose hashes are recorded in `data/manifests/`. Training uses 45000 images, the search optimises accuracy on a fixed 5000-image validation split (the same images since the first study: `configs/data.toml`, `[split]`), and the 10000 test images are only reported, never tuned on.

The model is a ResNet-9 in the style of David Page's DAWNBench entry (94 % test in 75 s on a V100, 2018), with a width multiplier covering 0.4-6.6 M parameters. Every training choice in `src/cifarq/train.py` (the network in `model.py`) is a hyperparameter with a default (`train.DEFAULTS`), and `src/cifarq/tune.py` defines a conditional search space of 53 parameters: optimiser (SGD or AdamW), schedule, regularisation, GPU-side augmentation, classifier head and logit scaling, epochs 6-30. Optional features are a switch plus a log-scaled strength reaching down to almost no effect, so the search can turn them down gradually instead of only on or off. The sampler is Optuna's multivariate TPE, with a "constant liar" so that trials running in parallel are not proposed twice, and Page's recipe (`tune.PAGE`) is always the first trial.

Note that the logit scaling matters more than it looks. At 4 epochs, scaling the logits by 0.125, using a cosine classifier, or batch-normalising the logits each gave roughly +10 points over a plain linear head (70.1 % to 80-82 %), which is why the space has all three.

## Running on the queue

The pipeline below needs solve-queue, so it documents how the results were produced rather than something you can run today. A few of its terms are used throughout. A **job** is one Python function call (here `cifarq.train.train` with its hyperparameters) that the queue runs on whichever machine it places it: the local GPU, or a Colab VM it starts and stops as needed (an **executor**; `configs/compute.toml` names them: two L4s, a G4 and an A100). Each job gets its own directory, and whatever the function writes there comes back when it finishes. The queue copies the submitting Python environment to a VM once and caches it, sends only changed source files with each job, and sends the dataset file once per VM, keyed by its hash.

The layout is that of [ml-queue-template](../ml-queue-template) (`.copier-answers.yml`): one script per stage in `pipeline/`, where each stage runs set in `configs/compute.toml` (`--targets` overrides it for one run), and every run writes a record to `results/<stage>/<time>.json` with its commit and dataset hash.

| Script | Does |
|---|---|
| `10_ingest.py`, `20_clean.py`, `30_dataset.py` | the hub's parquet files at a pinned revision, decoded, split; manifests in `data/manifests/` (committed) |
| `40_train.py` | one training (mode 1: `--targets colab:cifar-l4a --set epochs=16`) |
| `50_tune.py` | mode 3: Optuna here, each trial a job on the 4060 and the L4s at once |
| `55_tune_on_vm.py` | mode 2: a whole Optuna (or Ray Tune) search inside one job on one card, its trials as processes there; it uses the card's stored concurrency or measures and stores it, stops trials early through the study (`--pruner rungs`, chosen from this model's curves), and checkpoints the study as it runs. On a G4 the fastest and best way to search here (`docs/PLAN-mode2.md`) |
| `60_champion.py` | the best trial trained again over seeds, the median seed by validation kept |
| `70_publish.py` | the code at HEAD, the champion's network and the model card (`pipeline/card.md`) to the hub; `--dry-run` first |
| `tools/` | `watch.py` (follow a search), `rung_check.py` (choose early stopping), `conformal_check.py`, `pbt_g4.py`; for the autoresearch sessions below, `seeds.py` and `grid.py` (configurations over seeds as jobs), `fit_rungs.py`, `footprint_bench.py` (parameters, bytes, MACs, latency and memory against a reference), `compare_b0.py` (the win rule), `open_test.py` (one logged look at the test set), `watch_session.py` |

```bash
uv venv ~/.venvs/cifarq --python 3.12      # a uv or conda Python: a VM cannot rebuild the system one
uv pip install --python ~/.venvs/cifarq/bin/python -e '.[train,sf,test]' -e ../solve-queue[gcs]
~/.venvs/cifarq/bin/python pipeline/10_ingest.py && ~/.venvs/cifarq/bin/python pipeline/20_clean.py && ~/.venvs/cifarq/bin/python pipeline/30_dataset.py
~/.venvs/cifarq/bin/python pipeline/40_train.py --targets colab:cifar-l4a --set epochs=16          # mode 1
~/.venvs/cifarq/bin/python pipeline/55_tune_on_vm.py --target colab:cifar-g4 --trials 600 --minutes 45 --pruner rungs
                                           # mode 2: a whole search in one G4 job (--driver ray, --cold --seed N for comparisons)
~/.venvs/cifarq/bin/python pipeline/tools/rung_check.py         # where early stopping is safe, from complete curves
~/.venvs/cifarq/bin/python pipeline/tools/pbt_g4.py             # probe (seeds, batches, concurrency) and PBT on a G4
~/.venvs/cifarq/bin/python pipeline/50_tune.py --trials 80 --targets local colab:cifar-l4a colab:cifar-l4b # mode 3
~/.venvs/cifarq/bin/python pipeline/60_champion.py --seeds 5
~/.venvs/cifarq/bin/python pipeline/70_publish.py --dry-run && ~/.venvs/cifarq/bin/python pipeline/70_publish.py --no-space
~/.venvs/cifarq/bin/python -m pytest       # CPU tests on a tiny synthetic CIFAR; RUN_QUEUE_TESTS=1 adds one job through the daemon
```

One training at a time was best on the L4, the 4060, the A100 and the G4 (two at once gave 6 % less on the G4); mode 2 runs two processes taking turns at the card instead, so one trial's set-up overlaps the other's training. To follow a run, `solveq wait ID --grep trial` prints each trial as it ends (the solve-queue skill). Colab allows two GPU VMs per account, so run mode 3 after mode 2 has released its VM. Trials return only their numbers, so no weights are kept from the search; `60_champion.py` trains the best parameters again with `train(save='model')`. `70_publish.py` refuses uncommitted changes and a champion trained on another dataset than the manifest names, and its hub commit names the local commit. The Gradio app in `app/` is not hosted, since a Gradio Space now needs a paid Hugging Face plan or ZeroGPU.

To follow a search, `pipeline/tools/watch.py` (mode 3's study) or `--mode2` (the running mode 2 job's study, copied back every two minutes), or `uvx optuna-dashboard sqlite:///results/tune-study.db`. To resume a dead search, pass `--resume`: mode 3 reattaches the trials left in flight to their jobs, and mode 2 sends the last copied study to a new job. Completed trials are never re-run. All output goes to `results/` (not committed).

## Results (2026-10-27/28; the full log: `docs/RESULTS.md`)

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

## Results (2026-10-29/30): two autoresearch sessions on small, fast networks

Two agent-run campaigns (plans, ledgers and write-ups in `docs/autoresearch/`) searched for a better network than the
ResNet-9 above at a smaller inference footprint, using airbench's architecture and recipe (`src/cifarq/airbench.py`)
with flip-equivariant convolutions and other levers behind flags (`src/cifarq/methods.py`). All numbers are on the
same 45000/5000 split, 5 seeds, test opened once for the final models in the second session.

| Model | Params | Epochs | Views | Test | MACs/image |
|---|---|---|---|---|---|
| ResNet-9 (B0, the search's best above) | 3.71 M | 30 | 2 | 94.60 +- 0.11 % | 450 M |
| airbench96, replicated (96.01 % on all 50000 images, paper 96.03 %) | 9.92 M | 37 | 6 | 95.87 +- 0.04 % | 2712 M |
| Flip-equivariant airbench, 40 epochs (`2026-10-29-footprint-arch`) | 1.94 M | 40 | 1 / 6 | 95.00 +- 0.08 % / 95.45 +- 0.17 % | 237 M / 1424 M |
| The same, 100 epochs (`2026-10-30-equivariant-small`) | 1.94 M | 100 | 1 / 6 | 95.40 +- 0.20 % / **95.87 +- 0.17 %** | 237 M / 1424 M |

At one view the 1.94 M network beats B0 by 0.4-0.8 points and is smaller and faster on every footprint axis (2.0 ms
against 2.6 ms at batch 1 on the 4060, 32k against 14k images a second at batch 1024). At 100 epochs it matches
airbench96 on this split at a fifth of the parameters, though airbench96 stays about a point ahead on CIFAR-10.1.
The goal of 96 % at 2 M parameters was not reached (95.43 % at one view, 95.87 % at six). The method finding: at one
forward pass, flip equivariance buys what the plain network gets from its mirrored test-time view, ahead of it at
20-40 epochs and tied at 100 with a third fewer MACs, and 0.3-0.4 points ahead of a plain network with the same MACs.
Adding 90-degree rotations hurt, as did anti-aliased pooling, squeeze-and-excitation, proxy data filtering, Muon and
ensemble distillation. Training here is not bit-reproducible (cuDNN chooses its algorithms per run), so identical runs
differ by about 0.2 points, which is the floor for any effect claimed above.

The published model on the hub is still the ResNet-9.

## References

- D. Page, "How to train your ResNet" (Myrtle.ai, 2018-2019); [DAWNBench CIFAR-10](https://dawn.cs.stanford.edu/cifar10-training)
- [Lambda: ResNet9 to 94 % in 100 s on one Turing GPU](https://lambda.ai/blog/resnet9-train-to-94-cifar10-accuracy-in-100-seconds)
- K. Jordan, [94 % on CIFAR-10 in 3.29 seconds on a single GPU](https://arxiv.org/abs/2404.00498) (airbench, 2024); [cifar10-airbench](https://github.com/KellerJordan/cifar10-airbench)
- T. Cohen, M. Welling, [Group equivariant convolutional networks](https://arxiv.org/abs/1602.07576) (2016); M. Weiler, G. Cesa, [General E(2)-equivariant steerable CNNs](https://arxiv.org/abs/1911.08251) (2019)
- A. Adly, [Measuring the symmetry-data exchange rate](https://arxiv.org/abs/2606.01090) (2026)
- B. Recht et al., [Do CIFAR-10 classifiers generalize to CIFAR-10?](https://arxiv.org/abs/1806.00451) (CIFAR-10.1, 2018)
