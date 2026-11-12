# Status

Prose only: what was learned, what is uncertain, what the next session should know. The structured state (now, next,
the scoreboard, the milestones, the questions) lives in `session.json` and is shown by `runq session show`.

## 2026-11-12: two studies, the best confirmed over 5 fresh seeds

The first study (eqaug, job 4457, 11.5 h, 71 complete) found policy augmentation the lever: RandAugment at magnitude
13-15 in all 15 best trials, best 95.87 % two-seed validation (trial 113), 0.6 pp over FIN's 95.27 %. The second
(eqaug2, job 4458, stopped by the user after 4.7 h, 69 complete counting the 43 seeded) crept to 95.93 % (trial 104):
within the noise of a best-of-60 pick on two seeds. Its useful signal was direction: batch 256 took every late lead,
distillation weight rose to 0.39-0.45, magnitude settled at 14-17 (21 and 30 lost 0.5-0.7 pp).

Confirmation on seeds 2-6 (never seen by either search), one view, `results/autoresearch/confirm*`:

| config | epochs | val mean +- sd | vs control |
|---|---|---|---|
| eqaug 113 (control, batch 512) | 40 | 95.65 +- 0.18 | |
| eqaug2 104 (batch 256) | 40 | 95.85 +- 0.32 | +0.20 pp (p 0.3) |
| eqaug2 76 (batch 512) | 40 | 95.83 +- 0.14 | +0.18 pp (p 0.1) |
| eqaug2 88 (batch 256) | 40 | 95.81 +- 0.36 | +0.16 pp (p 0.4) |
| eqaug 113 (control) | 100 | 96.00 +- 0.08 | |
| eqaug2 104 | 100 | 96.20 +- 0.17 | +0.20 pp (p 0.07) |

What this says:

- The three eqaug2 points agree on a +0.16-0.20 pp gain over the control, below the contract's 0.3 pp tolerance.
  **Counted as the new best by the user's decision (2026-11-12)**, so the next study starts from it instead of
  confirming it again: the three agree in sign at both 40 and 100 epochs, and the tolerance came from 3-seed means.
- Batch 256 is not better than 512: 104 and 76 tie (+0.02 pp) and 256 trains 10 % slower. The common factor of the
  three winners is the stronger distillation (0.39-0.45 against 0.33) and slightly stronger RandAugment (15-16).
- The lead holds at 100 epochs (H2), and 100 epochs adds 0.35 pp to both: 96.20 % one-view validation, five seeds,
  the first configuration over 0.96.
- The search's own two-seed values ran 0.08 pp optimistic for the winner (95.93 vs 95.85): the winner's curse, small.
- The test set was opened by mistake: `pipeline/tools/seeds.py` did not pass `test='sealed'` (it does now, unless
  `--open-test`), so all 30 runs scored test. Logged in `data/holdout/test_looks.jsonl` as this session's look;
  nothing was chosen on it. For the record, 104 at 100 epochs: 96.14 +- 0.10 % test (control 95.94).

## For the next session

- Resume with `eqaug3` (RandAugment and cutout 8 fixed, batch 128/256/512, other ranges unchanged), warm from 104,
  76, 88 and seeded from both studies' git-kept exports. The exports hold pruned trials too (early-stopped with their
  curves, over-cap with none), so TPE knows the losing regions: 136 known points fit eqaug3 (57 complete, 79 pruned,
  repeats skipped), against 43 completes when eqaug2 was seeded.
  `python pipeline/55_tune_on_vm.py --target local --family eqaug3 --seeds 2 --kd results/models/ab96-45k-s0
  --concurrent 1 --pruner rungs --minutes 1440 --seed-from docs/autoresearch/2026-11-11-idle-search/eqaug2-trials.jsonl
  docs/autoresearch/2026-11-11-idle-search/eqaug-trials.jsonl`
- Expect small returns: two studies plateaued at 95.8-95.9 % two-seed at 40 epochs. Larger levers are 100 epochs
  (+0.35 pp, 2.5x cost), the M4 front, or augmentation easing (not built).
- The confirmed point is `candidates.jsonl` id `confirm100-t104-20261112` (train arguments, 100 epochs).
