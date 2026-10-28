# cifar-queue

This repository is a test problem for solve-queue, our compute queue library (not public at the moment). solve-queue is a thin wrapper that takes local and cloud GPUs and CPUs from a mix of sources (here a local RTX 4060 and Colab VMs) and presents them as one compute pool, so that HPC and ML workflows can be submitted as ordinary function calls without caring where they run. It handles environment and code sync, sending only what changed to storage buckets in the same cloud region as the compute to avoid egress costs, and copying between regions only when a job needs it. It close to a mix of different queue managers but handles my own unique needs.

I plan to push and update my test models publically.

This one is a small CNN trained on CIFAR-10 and tuned with Optuna. It was chosen because it has a well-known reference benchmark, and it exercises the queue's ML path: environment profiles, data blobs, function jobs, and distributed hyperparameter search using solve-queue's simple functional callback structure.

## Problem

The data is CIFAR-10 from the Hugging Face hub ([uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10), pinned revision), decoded once into a 184 MB `.npz` whose hashes are recorded in `data/MANIFEST.json`. The array itself is not committed; jobs name it as their `data`, so it is uploaded to the bucket once and copied to each VM once. Training uses 45000 images, the search optimises accuracy on a fixed 5000-image validation split, and the 10000 test images are only reported.

The model is a ResNet-9 in the style of David Page's DAWNBench entry (94 % test in 75 s on a V100, 2018), with a width multiplier covering 0.4-6.6 M parameters. Every training choice in `src/cifarq/train.py` is a hyperparameter with a default (`train.DEFAULTS`), and `src/cifarq/tune.py` defines a conditional search space of 53 parameters: optimiser (SGD or AdamW), schedule, regularisation, GPU-side augmentation, classifier head and logit scaling, epochs 6-30. Optional features are a switch plus a log-scaled strength reaching down to almost no effect, so the search can turn them down gradually instead of only on or off. The sampler is multivariate TPE with a constant liar, and Page's recipe (`tune.PAGE`) is always the first trial.

Note that the logit scaling matters more than it looks. At 4 epochs, scaling the logits by 0.125, using a cosine classifier, or batch-normalising the logits each gave roughly +10 points over a plain linear head (70.1 % to 80-82 %), which is why the space has all three.

## Running

```bash
uv venv ~/.venvs/cifarq --python 3.12
uv pip install --python ~/.venvs/cifarq/bin/python -e '.[train]' -e ../solve-queue[gcs]
~/.venvs/cifarq/bin/python scripts/prepare_data.py
~/.venvs/cifarq/bin/python scripts/1_single.py                          # one training on a Colab L4
~/.venvs/cifarq/bin/python scripts/2_optuna_on_vm.py --trials 60         # whole search inside one L4 job
~/.venvs/cifarq/bin/python scripts/3_optuna_distributed.py --trials 80   # study here, trials on the 4060 + two L4s
~/.venvs/cifarq/bin/python scripts/4_champion.py --seeds 5              # best trial retrained over seeds, models saved
~/.venvs/cifarq/bin/python scripts/publish.py --private                 # model repo and Space on the Hugging Face hub
```

Use a uv or conda Python, since a VM cannot rebuild the system one. Mode 2 first calibrates how many trainings the GPU can run at once before throughput stops growing; on both the L4 and the 4060 the answer was one (two at once gave 0 % and 5 % more). Colab allows two GPU VMs per account, so run mode 3 after mode 2 has released its VM.

The package alone (`pip install .`, torch and the hub client) is enough to load and run a trained model, the `train` extra adds the data preparation and the search, and the run modes need solve-queue. `train(save=dir)` writes the network into the job's directory, which comes back with the job, so `4_champion.py` retrains the best trial over several seeds and keeps the median one by validation accuracy rather than saving all 80 trials' weights (26 MB each at width 64). `publish.py` pushes the committed repository plus those weights to [thearn/cifar10-resnet9](https://huggingface.co/thearn/cifar10-resnet9), with a model card linked to the dataset. `space/` is a Gradio app that loads it with `ResNet9.from_pretrained`; it runs locally (`python space/app.py`) and is not hosted yet, since a Gradio Space now needs PRO or ZeroGPU (`publish.py --no-space` skips it).

## Results (2026-10-27/28)

| Run | Hardware | Trials | Time | Best val | Test |
|---|---|---|---|---|---|
| Defaults, 16 epochs | one L4 | 1 | 37 s training | 88.9 % | 88.3 % |
| Page's recipe | RTX 4060 | 1 | 154 s training | 93.5 % | 93.1 % |
| Mode 2 (older 44-parameter space, max 20 epochs) | one L4 | 60 | 40 min | 92.8 % | 91.8 % |
| Mode 3 | 4060 + two L4s | 80 | 36 min | **93.7 %** | **93.4 %** |
| Mode 3's best retrained, 5 seeds | RTX 4060 | 5 | 16 min | 93.2-93.8 % | 93.18 +- 0.16 % |

Mode 3's best trial kept Page's architecture and optimiser and changed mostly epochs (28), learning rate (0.78), logit scale (0.066), and added a weight EMA. It found his region within about 30 trials, and the driver was killed twice during that run and resumed without losing a trial. Retrained over 5 seeds (`4_champion.py`), the same parameters give 93.18 +- 0.16 % test, so the 0.3-point lead of the single trial over Page's single run is mostly seed noise: the search matched his recipe rather than beat it. Seed 0 reproduced the trial's validation accuracy exactly (93.66 %), and its network, the median seed by validation accuracy, is the published model (93.23 % test).

## Watching and resuming

```bash
~/.venvs/cifarq/bin/python scripts/watch.py            # mode 3, refreshes every 30 s
~/.venvs/cifarq/bin/python scripts/watch.py --mode2    # mode 2's study, as last copied back
uvx optuna-dashboard sqlite:///results/distributed-study.db
```

`watch.py` reads a backup copy of the SQLite study, so it never holds a lock the search needs. Run `optuna-dashboard` through `uvx` rather than installing it in the venv, since that would change the venv's cached layer and force a rebuild on every VM. In mode 2 the study lives on the VM, and solve-queue copies `progress.json` and `study.db` back every two minutes (`SOLVEQ_WATCH_S`).

To resume a dead search, pass `--resume` (optionally with a study name) to either driver. In mode 3, trials left in flight are reattached to their queue jobs: finished jobs report their result, running ones are followed, and lost ones are re-queued with the same parameters. In mode 2 the last copied study is sent to a new job, and only the trials that were running on the dead VM are repeated. Completed trials are never re-run, and since TPE refits on the full history the resumed search samples as the uninterrupted one would have.

All output goes to `results/` (not committed).

## References

- D. Page, "How to train your ResNet" (Myrtle.ai, 2018-2019); [DAWNBench CIFAR-10](https://dawn.cs.stanford.edu/cifar10-training)
- [Lambda: ResNet9 to 94 % in 100 s on one Turing GPU](https://lambda.ai/blog/resnet9-train-to-94-cifar10-accuracy-in-100-seconds)
- K. Jordan, [94 % on CIFAR-10 in 3.29 seconds on a single GPU](https://arxiv.org/abs/2404.00498) (airbench, 2024)
