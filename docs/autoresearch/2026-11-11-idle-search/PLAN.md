# CIFAR-10 search on idle GPU time

Kind: model campaign. Goal: use idle local GPU time to find a better one-view CIFAR-10 model under 2 M parameters
than the flip-equivariant airbench finalists. The evaluation contract is `contract.yaml`.

## Why this family and space (2026-11-11)

The ResNet-9 space (b0) was checked first: its best known point scores 94.88 % validation under the JAX code, and it
sits on the space's epoch bound. The flip-equivariant airbench from sessions 2026-10-29 and 2026-10-30 beats it by
0.40 pp test at one view with half the parameters, so idle time goes there. That family's recipe region is already
searched (a 61-trial two-seed search found nothing better than its start) and most add-ons measured negative (SE,
Muon, BlurPool, weight tying, proxy filtering). One-view accuracy saturated past 80 epochs, which points at
regularisation. So the space `families.eqaug_space`:

- fixes flip equivariance in the last two groups; keeps structured receptive fields as a switch (T39, the front's
  best, has them on);
- opens policy augmentation (none, TrivialAugment wide, RandAugment with its magnitude), never measured on this family;
- keeps distillation from airbench96 (45k, seed 0) with its weight free down to none;
- trains a fixed 40 epochs, two seeds a trial (test sealed), scored at one view; over-cap points refused before they run.

Each study: `pipeline/55_tune_on_vm.py --target local --family eqaug --seeds 2 --kd results/models/ab96-45k-s0
--concurrent 1 --pruner rungs --minutes N`, starting from FIN and T39. The driver writes the first starting point's
value as the scoreboard's baseline (`val_1view`) and each new best as it comes; the target is FIN at 40 epochs over
3 seeds, 95.27 %.

## Hypotheses

- H1: policy augmentation or a stronger regulariser lifts one-view validation at 40 epochs above FIN's by more than
  the tolerance.
- H2: a recipe that wins at 40 epochs keeps its lead when retrained at 100 epochs over 5 seeds.

## Gates and order

1. Smoke: a short eqaug study shows in the app; time per trial measured.
2. Idle-time studies.
3. A candidate above the tolerance retrained over 5 seeds at 40 and 100 epochs before it counts.

## Prepared: the accuracy / MACs front with policy augmentation (M4)

`pipeline/57_front.py --space aug --seed-from <the 24 h study's study.db> --trials N` on the local GPU after the 24 h
study: two objectives (two-seed one-view validation up, MACs a view down), `families.frontaug_space` (the front's
widths down to 48/48/64, RandAugment magnitude up to 30, 40 epochs), starting from the 24 h study's completed trials
as known points (70 at 08:30; their front ran 197-367 M MACs) and its best made narrower (3/4 and 1/2 widths).

## Next lever (not built)

Augmentation easing: the last epochs trained on clean images (needs a schedule in `airbench.train`).
