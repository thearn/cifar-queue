# Literature, code and models: 2026-10-30-equivariant-small

Searched 2026-10-30 (web search and direct fetches of repositories and papers). Status per figure: **verified** (read
from the source), **secondary** (a review or README quoting it), **unverified** (search summary only). The previous
session's `literature.md` holds the frontier table for small CIFAR-10 models; only what this session adds is here.

## 1. Where the target sits

| Result | Params | Epochs | Views | Accuracy | Status | Source |
|---|---|---|---|---|---|---|
| airbench96 (legacy) | 9.9 M | 37 | 6 | 96.03 % | verified (README) | [cifar10-airbench](https://github.com/KellerJordan/cifar10-airbench) |
| airbench96_faster | 9.9 M | 45 (student sees 50 % of each batch) | 6 | 96.00 %, n = 200, 27.3 s A100, 3.1 PFLOPs | verified (source header) | [airbench96_faster.py](https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench96_faster.py) |
| airbench94_muon | ~2 M | ~10 | 6 | 94.01 %, 2.59 s | verified (README) | same repo |
| ResNet-18, standard training | 11 M | ~200 | 1 | 96.0 % in 7 min, 32 PFLOPs | verified (README's baseline line) | same repo |
| ConvMixer-256/12, /16 | 1.0 / 1.3 M | 200 | 1 | 96.39 / 96.74 % | verified last session (paper Table 4) | previous `literature.md` |
| ConvMixer-cifar10 repo, 256/8 k5 p2 | ~0.6 M | 25 / 100 | 1 | ~92.5 % / > 94 % | verified (README) | [locuslab/convmixer-cifar10](https://github.com/locuslab/convmixer-cifar10) |
| p4m-ResNet44 (Cohen & Welling) | 2.62 M | long | 1 | 93.54 % plain, 95.06 % with aug | verified from the paper's table, **not reproduced**: 90.53 % plain by [1911.07849](https://arxiv.org/pdf/1911.07849) | [1602.07576](https://arxiv.org/pdf/1602.07576) |
| gc-p4m-ResNet44 (learned symmetries) | 1.88 M | long | 1 | 93.28 % plain, 94.57 % aug | secondary | [2106.04914](https://arxiv.org/pdf/2106.04914) |
| Last session's finalist (D1 D1 airbench) | 1.94 M | 40 | 6 / 1 | 95.45 / 95.00 % (45k split) | verified (ours) | `../2026-10-29-footprint-arch/RESULT.md` |
| FlexNet | 0.67 M | long | 1 | 92.2 % | secondary | [rjbruin/flexconv](https://github.com/rjbruin/flexconv) |

**Not found:** any result at ≤ 2 M parameters, ≤ 100 epochs and ≥ 96 % with one view. The Hugging Face hub holds no
small CIFAR-10 checkpoint near this point (searched: convmixer, resnet9, airbench, "cifar10 small"; the hits were a
68k-parameter ResNet20 at 82 %, a 345k model, ConvNeXt and ViT fine-tunes). The hub is not a source for this session.
**Successor (M0, read 2026-10-30):** [hiverge/cifar10-speedrun](https://github.com/hiverge/cifar10-speedrun) holds the 94 %
record, 94.02 % in 1.98 s (A100), with vectorised Muon, brightness/contrast jitter, SiLU, SVD whitening, and TTA
applied only to low-confidence predictions (our MT6). No 96 % track and no stated rules.

## 2. Equivariance: results, code and the points to consider

- **Group convolutions.** [Cohen & Welling 2016](https://arxiv.org/pdf/1602.07576): p4m-ResNet44 6.46 % error plain,
  4.94 % with augmentation, ~2.62 M parameters, against 9.45 / 5.61 % planar at 2.64 M. Two later papers could not
  reproduce the plain figure ([2106.04914](https://arxiv.org/pdf/2106.04914) notes it; [1911.07849](https://arxiv.org/pdf/1911.07849)
  measured 9.47 %). **Point:** the plain-CIFAR gains are large because no augmentation supplies the symmetry; with
  flip augmentation the gap shrinks to ~0.7 pp, and airbench's recipe already has flips and translations. Expect
  tenths, not points.
- **Steerable CNNs.** [Weiler & Cesa 2019](https://ar5iv.labs.arxiv.org/html/1911.08251), Table 5 (verified):
  wrn28/10 3.87 % → D1 D1 D1 3.36 ± 0.08, C8 C4 C1 3.20 ± 0.04, D8 D4 D1 3.28 ± 0.10 (same channel count), D8 D4 D4
  2.91 (wider); with AutoAugment 2.6 → 2.39 (same width) and 2.05 (wider). "At test time steerable CNNs are not more
  expensive than conventional CNNs of the same width"; wider equivariant models cost more. **Points:** (a) restricting
  equivariance towards the output (D8 → D4 → D1) beat global equivariance on natural images, and our `flip_eq=2` is
  that pattern's tail; (b) the flip-only D1 D1 D1 already took 0.5 pp of the 0.6 pp gain at 300-epoch schedules.
- **Separable group convolutions.** [SepGrouPy](https://github.com/Attila94/SepGrouPy) (Lengyel & van Gemert):
  p4m-ResNet44 8.06 % vs 13.10 % on unaugmented CIFAR-10 at ~8× the MACs of the planar net (README, secondary).
  The MAC multiplier is the reason M3 reports matched-MAC baselines.
- **Libraries.** [escnn](https://github.com/QUVA-Lab/escnn) (successor of e2cnn) implements all of this through
  `GeometricTensor` wrappers; its kernels module was refactored in 1.0 and the README's CIFAR example is the 3D
  octahedral case. **Decision:** not used. It would not keep airbench's channels_last fp16 path or `torch.compile`
  speed, and a p4m 3×3 layer is a `rot90`/flip expansion of the stored kernel, the same shape as `FlipConv`
  (`methods.py`). SepGrouPy's layer is the reference for the separable variant (Q3).
- **The matched-compute question.** [Adly 2026, arXiv 2606.01090](https://arxiv.org/abs/2606.01090) (exploratory,
  stat.ME): on a controlled C_n task, augmentation plus test-time orbit averaging matches an equivariant model's
  per-epoch validation curve exactly, a misaligned group is worse than no constraint, and the architecture-vs-
  augmentation gap "depends on asymmetric test-time computation". **Point:** this predicts that plain + flip TTA ties
  flip-equivariant + no TTA at equal epochs, and that equivariance's value is inference compute and the misaligned-
  group risk (rotations on CIFAR). Our last session's numbers are consistent: TTA still added 0.45 pp on top of
  equivariance (translation views). H1 is the matched-MAC version of this on a natural task with a short schedule;
  no such comparison was found for flips or for CIFAR.
- **Anti-aliased downsampling.** [Zhang 2019](https://arxiv.org/pdf/1904.11486), code
  [adobe/antialiased-cnns](https://github.com/adobe/antialiased-cnns): MaxPool → MaxBlurPool etc.; the paper shows
  CIFAR-10 shift-consistency gains for VGG, and no CIFAR accuracy table was found. Accuracy effect open (Q2).
- **Nothing combines equivariant layers with airbench or hlb-CIFAR10** (searched repositories and issues; the hits
  were the papers above and HexaConv). That is the gap M3 fills.

## 3. Other operators and blocks

- **Structured receptive fields.** [Jacobsen et al. 2016](https://arxiv.org/pdf/1605.02971), code
  [jhjacobsen/RFNN](https://github.com/jhjacobsen/RFNN) (framework unchecked; likely Theano-era): filters as
  combinations of Gaussian derivatives; gains on small and medium data. N-Jet ([Pintea et al. 2021](https://arxiv.org/pdf/2106.03412.pdf))
  learns σ; "one code implementation" listed, URL to find in M0. [Penaud-Polge et al. 2022](https://arxiv.org/pdf/2207.08424)
  has a repository. **Point:** none of these report short schedules; the implementation is small enough to write
  here (a fixed basis tensor and a learned coefficient tensor, `F.conv2d` on the composed kernel).
- **Fourier global filters.** [GFNet](https://arxiv.org/pdf/2107.00645.pdf), code
  [raoyongming/GFNet](https://github.com/raoyongming/GFNet): 2D FFT, learned complex filter, inverse FFT, log-linear
  cost; ImageNet 82.9-85.0 %, **no CIFAR-10 result** by the authors. **Point:** `torch.fft` in fp16 is unreliable, so
  the block runs under `precision='amp'` or casts to fp32 internally.
- **Squeeze-and-excitation.** [Hu et al.](https://arxiv.org/pdf/1709.01507): CIFAR-10 ResNet-110 6.37 → 5.21 %,
  ResNet-164 5.46 → 4.39 %, ~400 epochs. No controlled short-schedule study found; O2 is that study at our scale.
- **Continuous kernels.** [FlexConv](https://arxiv.org/pdf/2110.08059.pdf): 92.2 % at 0.67 M (secondary). Not pursued.
- **Muon.** airbench94_muon.py is the current 94 % record (2.59 s). No 96 %-scale result found. O4 screens it.

## 4. The data side

- **airbench96_faster's filtering** (read from the source): a proxy (widths 32/64/64, depth 2, scaling 1/9) is
  trained first for the same 45 epochs at batch 1024; every step it scores the batch by per-example cross-entropy
  (label smoothing 0.2) and keeps the hardest 512 of 1024, back-propagating through those only on every 4th step
  (forward-only otherwise); the masks are stored and replayed for the main model (128/384/512, depth 3), which thus
  back-propagates through half of each batch. The proxy's time is counted in the 27.3 s. **Points:** (a) this is
  the only documented record-setting trick in exactly our regime; (b) at 2 M parameters the student may not have the
  capacity the hard half demands, which is what D1 measures; (c) our MT4 (teacher KL as the score) is the natural
  comparison, and the airbench96 teachers from the last session (`results/jobs/train/00023-00025-train/model`) exist.

## 5. Search under noise

- Overtuning / winner's curse in HPO: a [review](https://www.themoonlight.io/review/overtuning-in-hyperparameter-optimization)
  names repeated resampling and racing as the mitigations; [Bischl et al.](https://ar5iv.arxiv.org/html/2107.05847)
  cover racing; [sequential statistical tests for tuning](https://arxiv.org/pdf/2112.12438) and
  [PyHopper](https://arxiv.org/pdf/2210.04728) (seed-aware pruning, Optuna-like API) are the method references.
  Optuna has no built-in repeated-seed objective. **Decision:** two seeds per trial inside the objective, seed 1
  only for survivors of seed 0's rungs (§2.1); the sampler stays multivariate TPE with constant liar.

## 6. Open items for M0

1. Weiler & Cesa Table 5's parameter counts and epochs (the fetched half of the paper did not include them).
2. The Hiverge successor records and the airbench leaderboard's rules (for the write-up's framing only).
3. The N-Jet code's URL and framework; whether SepGrouPy's layer runs under channels_last fp16.
4. Cohen & Welling's original CIFAR table, to settle which p4m numbers to quote (the search summary converted
   errors to accuracies; the reproduction failures make the exact figure moot for our purposes).
