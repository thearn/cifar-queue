# Result: equivariance substitutes for the flip view; the 96 % one-view target was not reached

Session `2026-10-30-equivariant-small`, run 2026-10-30. Plan: `PLAN.md`. Evidence: `ledger.tsv`. Spend: about 52 of
75 Colab units. That is 25.2 units by the balance until 12:25, then counted by this session's VM hours at 10.05
units an hour once another session's G4 was also up: M5 about 17.6, M6 about 9.5.

## Targets

| Target (PLAN §1) | Required | Result (test, 5 seeds, opened once) | Outcome |
|---|---|---|---|
| ≥ 96.0 % at ≤ 2 M params, **one view**, ≤ 100 epochs | mean ≥ 96.00, lower bound ≥ 95.80, val ≥ 96.0 | **95.43 ± 0.13 %** (T39), lower bound 95.31; val 95.62 | Not met (−0.57 pp) |
| Fallback: the same at **six views** | the same | **95.87 ± 0.17 %** (FIN), lower bound 95.71; val 95.96 | Not met (−0.13 pp) |

**What the finalists do match.** airbench96 (9.92 M parameters, 37 epochs) on the same 45k training split scores
95.87 ± 0.04 % with six views and 95.37 ± 0.11 % with one (last session, E1.1). The flip-equivariant finalist FIN
(1.94 M, 100 epochs) scores **95.87 ± 0.17 % with six views and 95.40 ± 0.20 % with one**. That is parity with
airbench96 at a fifth of its parameters, about half its MACs per view (237 M against 452 M), and 2.7× its epochs.

**On the shifted CIFAR-10.1 images, the larger network is ahead** (one look each):
- airbench96 at six views: 90.15 %
- FIN at six views: 89.15 % (−1.00 pp, paired 95 % interval [−2.10, +0.10])
- FIN at one view: 88.10 % (−2.05 pp, [−3.25, −0.85])

So the parity holds on the CIFAR-10 test set, not under distribution shift.

## The method result (H1): flip equivariance buys the flip view's accuracy at fewer MACs

Validation, 3 seeds (M3, plus M4's repaired control). All arms have the same recipe and 1.9-2.6 M parameters.

| Epochs | Flips at 1 view (237 M MACs) | Plain at 1 view (177 M) | Plain at 2 views (354 M) | Plain, matched MACs, 1 view (234 M, 2.60 M params) |
|---|---|---|---|---|
| 20 | **94.35** | 93.41 | 94.24 | 93.96 |
| 40 | **95.27** | 94.50 | 95.11 | 94.87 |
| 100 | **95.62** | 95.27 | 95.68 | 95.30 |

- **Against the plain network with its flip view:** the equivariant network at one view beats it at 20 and 40
  epochs (+0.11, +0.16 pp) and ties it at 100 (−0.06 pp), using a third fewer MACs. The margin shrinks as training
  lengthens. A 2026 preprint (Adly, arXiv 2606.01090) predicts this tie at equal test-time compute on a controlled
  task; this is the same comparison on natural images in a short schedule.
- **At matched MACs:** it beats a plain network with the same MACs per view (and 34 % more parameters) by 0.32-0.40 pp
  at every budget.
- **The wrong group hurts.** Adding quarter turns (p4m) cost 0.4-0.6 pp at one view at matched parameters, at 1.1×
  (one group) or 2.5× (two groups) the MACs. Rotations are not a symmetry of CIFAR-10. This matches the preprint's
  misaligned-group control and Weiler & Cesa's result that less equivariance does better on natural images.
- **Anti-aliased downsampling (BlurPool) hurt:** −0.2 to −0.3 pp.
- **MAC/accuracy front (M7, from 46 existing points).** From ~190 M MACs up, every point on the front is a
  flip-equivariant network, except the plain network at six views between 475 M and 1,424 M MACs. A
  structured-receptive-field variant at 192 M MACs and 1.49 M parameters reached 95.54 % (2 seeds), matching the
  1.94 M finalist with fewer MACs and parameters.

## Footprint (frozen bench, RTX 4060, median-validation model; MACs by the a4 counter)

| Model | Class vs B0 | Params | MACs/image | GPU b1 | GPU b1024 | CPU 1 thread | GPU peak b1024 |
|---|---|---|---|---|---|---|---|
| B0 (2 views) | ref | 3.71 M | 450 M | 2.60 ms | 13.8k img/s | 13.9 ms | 552 MB |
| **FIN, 1 view** | **S** | 1.94 M | 237 M | 2.01 ms | 32.3k img/s | 11.2 ms | 571 MB |
| FIN, 6 views | P | 1.94 M | 1,424 M | 10.6 ms | 5.0k img/s | 85.3 ms | 588 MB |
| T39, 1 view | P | 1.99 M | 358 M | 3.39 ms | 21.8k img/s | 29.8 ms | 735 MB |
| airbench96, 6 views | out | 9.92 M | 2,712 M | 7.75 ms | 3.5k img/s | 91.3 ms | 603 MB |

## What did not help (all on the finalist at 100 epochs, 3 paired seeds; M4)

- **Structured receptive fields:** +0.11 pp. It did not reproduce: the identical configuration retrained in M6 scored
  95.45 % at one view, 0.17 pp under M3's baseline (see the noise finding below).
- **Disagreement-weighted distillation:** −0.07 pp.
- **Weight-tied depth:** −0.14 pp.
- **Field-shared squeeze-and-excitation:** −0.23 pp.
- **Proxy filtering:** −0.43 pp. This is airbench96_faster's record trick; it did not transfer to a 2 M student.
- **Muon for the filters, untuned:** −0.69 pp.
- **Fourier global filter:** dropped before running. The feature maps are 15/7/3 pixels, so a 3×3 convolution is
  already nearly global.
- **More epochs:** saturated at one view after 80 epochs (M2).
- **A 61-trial two-seed search (M5):** found nothing better than its starting point, the finalist.

## Findings about the method of research

- **The airbench recipe is not reproducible run to run on these GPUs.** `cudnn.benchmark` picks convolution
  algorithms per run. The same configuration and seeds gave 3-seed means 0.3 pp apart, and one seed moved 0.54 pp.
  Effects under ~0.2 pp in 3-seed screens are inside the noise. That is why M4's structured-receptive-field gain
  vanished on retrain. Later sessions should use 5 or more seeds for screens, or set deterministic algorithms when
  resolution matters.
- **A correction to the previous session.** The MAC counter (`footprint.count_macs`, revision a3) skipped
  flip-equivariant layers. The last session's finalist is 237 M MACs a view, not 93 M. This was fixed in revision a4
  and corrected in that session's RESULT.md.
- **Infrastructure added:**
  - `test='sealed'`
  - the two-seed search objective
  - p4m group convolutions, BlurPool, structured receptive fields, field-shared SE, proxy filtering and Muon, all
    behind flags
  - `grid.py` (many configurations submitted from one process)
  - `open_test.py` (one logged test look)
  - teacher shipping in `seeds.py`

## Reproduce the finalist

`results/autoresearch/eqsm/finalist-d1d1.json` with `epochs=100`:
- depth 3, widths 128/112/144, `flip_eq=2`
- batch 1024, lr 9.37, momentum 0.78, weight decay 0.0073, label smoothing 0.17, translate 4, cutout 12
- distillation from airbench96 45k seed 0 at weight 0.12, temperature 5.5

```bash
python pipeline/tools/grid.py M6-confirm --grid results/autoresearch/eqsm/m6b-grid.json --seeds 1 2 3 4 5 --targets colab:cifar-g4
python pipeline/tools/open_test.py --why "..." results/jobs/train/<job>/model ...
```

Saved models (`results/jobs/train/<job>/model`): FIN seeds 1-5 in jobs 00228, 00229, 00230, 00233, 00234; T39 seeds 1-5
in 00222, 00223, 00224, 00231, 00232. The benched median-validation models are FIN seed 1 (00228) and T39 seed 2 (00223):

| Model | `model.safetensors` sha256 | `config.json` sha256 |
|---|---|---|
| FIN seed 1 (00228) | `59bbab1ca3fd03e7bef879df5ac9170651d4c71411b871f229c33c054338216d` | `77a59bf619254437408c7b7da8e3aba584f9f6699b93b05dbde5485beff87194` |
| T39 seed 2 (00223) | `7f2f7895b814b302b89c4dbc4e274b9408ad83e9151b75eda63e83df46b4974e` | `1f665e5748ab8c3c06395aee819d8b17160cf984d31fb95806ca23cb43c22961` |

## Novelty, honestly

The 96 % target was the novel result on offer, and it was not reached. The finalists' accuracy (airbench96's on this
split at a fifth of the parameters) is a useful engineering point, not a record.

The H1 measurement is the candidate contribution: a matched-parameter and matched-MAC comparison of flip
equivariance against flip TTA across schedule lengths on CIFAR-10. It shows a short-schedule advantage that closes
into a tie at equal test-time compute, a consistent win at matched MACs, and a cost for the misaligned rotation group.
We found no published version of it for natural images in this regime. It rests on one architecture family, one
dataset and 3-seed means with ~0.2-0.3 pp run-to-run noise, so it is a short note or a pull request, not a paper,
unless repeated with more seeds and a second dataset.
