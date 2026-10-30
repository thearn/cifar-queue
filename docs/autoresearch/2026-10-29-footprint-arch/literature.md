# Literature behind the goals and candidates (2026-10-29)

Source claims are kept apart from local observations. Status: **verified** (read from the paper, its table or its
source file during this review, with the pointer given), **computed** (arithmetic here from verified
architecture details), or **recalled** (memory of the paper; checked in the Phase 1 screen). Every published accuracy
below trains on all 50,000 images; this project trains on 45,000.

## Local observations (this repository)

- B0: ResNet-9 width 48, whitened, CELU, schedule-free AdamW, 30 epochs, flip TTA. 3.71 M trainable parameters,
  225 M MACs (450 M with TTA), 94.57 % test over 3 seeds on the G4 (`docs/PLAN-mode2.md`, Result 9). Batches above
  256 lost accuracy; two trainings at once lost 6 % throughput on the G4; PBT added nothing (Result 10).
- B0's search was capped at 30 epochs; the ≥ 96 % results under 2 M parameters train for 200 epochs or more, so
  epochs are a real axis.
- Logit scaling or normalisation was worth about 10 points at 4 epochs (README), so every family's head gets the
  same options.

## The frontier that Goal A and Goal B are set against

| Model | Params | Epochs | Augmentation | CIFAR-10 test | Source | Status |
|---|---|---|---|---|---|---|
| airbench94 (64/256/256, 2 convs a block) | 1.97 M | 9.9 | flip, 2-px translate; 6-view TTA | 94.01 % (93.2 % no TTA) | Jordan 2024, arXiv 2404.00498 | verified |
| airbench95 (128/384/384) | ~4.6 M | 15 | as above | 95.01 % | same | widths verified; params computed |
| **airbench96** (128/384/512, 3 convs a block, residual over convs 2-3) | **≈ 9.9 M** | **37** | + 12-px cutout; TTA level 2 (mirror + translate) | 96.05 % (95.6 % no TTA) | `legacy/airbench96.py` in the repository | verified from source; params computed (312 + 322,560 + 3,096,576 + 6,488,064 + 5,120 + BN biases) |
| airbench96_faster | same main net | 45, half the examples a batch kept by a proxy net's loss | as above | 96.00 % in 27.3 s | `airbench96_faster.py` | verified |
| hlb-CIFAR10 | > 2 M (widths 32/64/256/512; arithmetic) | 12.1 | flip, translate, cutmix late | 94 %; 95.79 % in ~110 s | tysam-code/hlb-CIFAR10 `main.py` | widths verified; params computed roughly |
| ConvMixer-256/12, patch 1, kernel 8 | 1.006 M | 200 | RandAug, mixup, cutmix, random erasing, random scaling | 96.39 % | Trockman & Kolter 2022, Appendix B Table 4 | verified |
| ConvMixer-256/16, patch 1, kernel 8 | 1.339 M | 200 | as above | 96.74 % | same | verified |
| ConvMixer-256/8, patch 1, kernel 9 | 0.707 M | 200 | as above | 95.88 % | same | verified |
| ConvMixer-256/8, patch 2, kernel 9 | 0.709 M | 200 | as above | 95.00 % | same | verified |
| ConvMixer-256/8, kernel 5, patch 2 (the CIFAR repo's quick runs) | ~0.6 M | 25 / 100 | RandAug (+ jitter, erasing at 100) | 92.5 % / > 94 % | locuslab/convmixer-cifar10 README | verified |
| DenseNet-BC L=100 k=12 | 0.8 M | 300 | flip, translate | 95.49 % (4.51 % error) | liuzhuang13/DenseNet README | verified |
| DenseNet L=40 k=12 | 1.0 M | 300 | as above | 94.76 % | same | verified |
| PyramidNet-110 α=48 (basic) | 1.7 M | 300 | as above | 95.38 % (4.62 %) | jhkim89/PyramidNet README | verified |
| PyramidNet-164 α=48 (bottleneck) | 1.7 M | 300 | as above | 95.79 % (4.21 %) | same | verified |
| FastBoost (progressive attention) | 0.85 M; 0.37 M | not in the abstract | not in the abstract | 95.57 %; 93.80 % | arXiv 2511.01026 | verified from the abstract only |
| Shake-Shake 26 2x32d, S-S-I | 2.9 M | 1,800 | flip, translate | 96.45 % (3.55 %) | Gastaldi 2017, Table 1 | verified |
| Shake-Shake 26 2x32d, S-E-I | 2.9 M | 1,800 | as above | 96.53 % (3.47 %) | same | verified |
| Shake-Shake 26 2x32d + cutout | 2.9 M | 1,800 | + cutout | 97.0 % (3.0 %) | Cubuk et al. 2019 (AutoAugment), Table 2 | verified |
| **Shake-Shake 26 2x32d + AutoAugment** | 2.9 M | 1,800 | AutoAugment CIFAR policy | **97.5 ± 0.1 % (2.5 ± 0.1 %)** | same (baseline row 3.6 %) | verified |
| Shake-Shake 26 2x96d + TrivialAugment | 26.2 M | 1,600 | TA wide | 98.21 ± 0.06 % | Müller & Hutter 2021, Table 2 | verified |

Checked gaps:
- **RandAugment** (Cubuk et al. 2020) and **TrivialAugment** report Shake-Shake only as 26 2x96d (their appendices
  name the configuration); neither has a 2x32d row. Verified from both papers' text. So 97.5 % with AutoAugment is
  the documented frontier at 2.9 M.
- No result found at ≤ 2 M parameters and ≥ 96 % with fewer than 200 epochs. The sub-2 M results at ≥ 96 % are
  ConvMixer at 200 epochs with heavy augmentation; the short-schedule results under 2 M are airbench94 (94.0 %,
  10 epochs) and the ConvMixer CIFAR repository (92.5 % at 25, > 94 % at 100). Two extended web searches on the
  combination returned nothing closer. Not checked: the full text of FastBoost (its epochs), and the sub-2 M region
  of NAS-cell papers (DARTS-style cells are 3.3 M at 97 %; scaled-down variants may exist).

Corrections to earlier notes in this session:
- airbench96 is **128/384/512 at 37 epochs** (the source file), not 128/512/512 at 40 (the paper's HTML summary as
  fetched); its parameter count is ≈ 9.9 M, not 12.7 M. The paper's 46.3 s run and the README's 34.7 s are
  different versions of the script.
- ConvMixer's small models reach 96.4-96.7 % at 1.0-1.3 M parameters (verified), stronger than the "about 96 % at
  0.6 M" recalled earlier. That makes "96 % under 2 M" documented at 200 epochs, and the 40-epoch axis in Target A
  necessary rather than optional.

## Neural families: further details

| Source | Claim | Status |
|---|---|---|
| Jordan 2024 (airbench) | Shared tricks: 2x2 frozen whitening first layer (24 channels, its bias trained for 3 epochs), dirac init, BN without affine scale and its bias at 64x the learning rate (`bias_scaler`), Lookahead, Nesterov SGD at lr 9.0 per 1,024 examples with momentum 0.85 and wd 0.012, batch 1,024, label smoothing 0.2, alternating flip, output × 1/9; multi-crop TTA weights 0.25 unshifted / 0.125 shifted | verified |
| Gastaldi 2017 | Recipe: 1,800 epochs, lr 0.2, cosine without restart, batch 128 on 2 GPUs, standard flip + translation, otherwise fb.resnet.torch; 26 2x32d = depth 26, two residual branches, first block width 32; branch = ReLU-Conv3x3-BN-ReLU-Conv3x3-BN-Mul(0.5) | verified |
| Local arithmetic on 2x32d | ≈ 2.9 M params (matches the paper's 2.9 M); ≈ 450 M MACs an image (151 M a stage), twice B0's | computed |
| Müller & Hutter 2021, appendix | Shake-Shake 2x96d trained 1,600 epochs, SGD Nesterov, batch 128, cosine; TA needs no search | verified |
| Hassani et al. 2021 (CCT) | CCT-7/3x1: 3.76 M params, 1.19 G MACs, 96.53 % at 300 epochs; 97.48 % at 1,500; 98.00 % at 5,000 | verified (abstract and repository table) |
| Gupta et al. 2023, *Efficient ResNets* | a ResNet under 5 M at 96.04 % | verified (abstract) |
| DMSResNet, preprints.org 202407.1953 | 96.56 % at about 4.3 M with SE blocks, mixup, KD | verified (summary); not peer reviewed |
| Zagoruyko & Komodakis 2016 (WRN) | WRN-16-4 ≈ 2.7 M at ≈ 5 % error, 200 epochs | recalled |

## Non-neural methods (from the earlier goal; kept for reference)

| Source | Claim | Status |
|---|---|---|
| Thiry, Arbel, Belilovsky & Oyallon, ICLR 2021, arXiv 2101.07528 | a single layer over a whitened patch dictionary plus a linear classifier reaches the 87-90 % range of earlier convolutional kernel methods | verified (abstract) |
| Coates, Lee & Ng 2011 | k-means patches + linear SVM: 77.9 % at 1,600 features, 79.6 % at 4,000 | recalled |
| Convolutional kernel methods (Shankar et al. 2020) | ~89-90 %; inference needs the training set | recalled; out of scope now |

## M0: closed 2026-10-29

1. **FastBoost** (arXiv 2511.01026, full text read): Table 4 gives FastBoost-Base 0.85 M at 95.57 % and Tiny 0.37 M at
   93.80 % on CIFAR-10. The headline runs' epochs are not stated; its ablations train 100 epochs with AdamW, batch
   256, lr 1e-3. Its comparison table lists CCT-6/3x1 at 77.3 %, against the 95.3 % reported by CCT's own authors,
   which lowers confidence in the paper's comparisons. Below 96 % either way: no effect on Target A. Not kept as a
   family (A10 dropped: no released code found, an architecture of attention blocks over MBConv stacks that would
   need reimplementing from a sparse description).
2. **NAS cells under 2 M:** not searched in depth. DARTS-style cells are 3.3 M at about 97 % with 600 epochs,
   cutout and drop-path; scaled-down variants are possible but would also be long schedules. Low risk to the
   ≤ 40-epoch claim; to be rechecked before any write-up (M5).
3. **airbench parameter counts confirmed** by building the ported networks (`tests/test_airbench.py`): airbench94
   1,972,792 (paper 1.97 M), airbench96 9,918,776 (our ≈ 9.9 M). The other families against their papers:
   ConvMixer-256/8 k9 p1 706,570 (707 K), 256/12 k8 1,005,578 (1,006 K), 256/16 k8 1,339,402 (1,339 K);
   DenseNet-BC-100-12 769,162 (the paper's rounded 0.8 M); PyramidNet-164 α48 bottleneck 1,715,442 (1.7 M),
   PyramidNet-110 α48 basic 1,772,706 (1.7 M stated, 4 % over: channel rounding differs from the reference code);
   Shake-Shake 26 2x32d 2,923,162 (2.9 M); WRN-16-4 2,748,890 (2.7 M); ResNet-9 width 32 1,646,122.
