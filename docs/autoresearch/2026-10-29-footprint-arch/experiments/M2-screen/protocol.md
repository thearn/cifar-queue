# M2: Goal A family screens (one seed each, 40 epochs, local)
Frozen: 2026-10-29, fparch/main at the commit adding this file.

- Hypothesis: at 40 epochs under the 2 M cap, some families reach within 1 pp of the best on validation with a
  transplanted airbench recipe (or their paper's), and are worth an M3 search.
- Decision: which families get a G4 search, in which order (PLAN.md §8), and which are dropped.
- Recipes (all `cifarq.airbench.train`, 40 epochs, 45k/5k, TTA level 2 with no-TTA reported, seed 0):
  - `r94`: AIRBENCH94's (lr 11.5, wd 0.0153, warmup 0.23, final 0.07, bn momentum 0.4, translate 2, no cutout)
  - `r96`: AIRBENCH96's (lr 9, wd 0.012, warmup 0.1, final 0, bn momentum 0.1, translate 4, cutout 12)
  - `adamw-ra` (ConvMixer's paper style): AdamW lr 0.01, weight decay 0.01, betas (0.9, 0.99), RandAugment m 12,
    label smoothing 0.1, warmup 0.1 then linear to 0, translate 4, no cutout
- Configurations (stored params, MACs per view):
  A1 airbench 64/256/256 d2 (1,972,792; 118 M): r94, r96
  A2 airbench 64/160/224 d3 (1,866,552): r96; 64/192/192 d3 (1,857,336): r96
  A3 resnet9 width 35 (1,968,655; 114 M): r96
  A4 pyramidnet-110 alpha 48 (1,772,706; 404 M): r96
  A5 densenet-bc 100 k12 (769,162; 288 M): r96
  A6 convmixer 256/8 k5 p2 (594,186; 148 M): r96, adamw-ra; 256/12 k8 p1 (1,005,578; 1007 M): adamw-ra
  A7 shakeshake 26 2x24d (1,647,978; 242 M): r96
  A9 wrn-16-3 (1,549,530; 223 M): r96
  (A8, ensembles, needs ensemble support in the predictor; screened after M3's first families.)
- Measurements: val (TTA) and val no-TTA, test reported; train seconds; diverged runs noted.
- Kill rule (PLAN.md M2): a family more than 1.0 pp below the best family's validation (TTA) at 40 epochs, after a
  12-trial lr/wd sweep (the sweep only for families within 2 pp, the rest dropped on this screen alone), is dropped.
- Compute: local 4060 through the queue, several at once as the queue admits them (accuracy unaffected; seconds not
  comparable across runs and not used).
