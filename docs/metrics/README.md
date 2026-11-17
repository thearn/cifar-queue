# Primary metric: relative error-cost score

Adopted as the repo's main model-comparison and new-autoresearch objective on **2026-11-16**. Maximize this
score; report accuracy, latency, bytes and uncertainty alongside it. NetScore is a secondary literature comparison,
not the selection objective. The metric evaluates complete prediction pipelines, regardless of model family.

This policy supersedes the earlier accuracy-only objective, the proposal to cap every cost at d3's values, and
the recommendation to call d3 the champion regardless of its score. Historical measurements and campaign
records retain their original protocols and objectives; they are not retroactively converted into new evidence.

## Definition and use

Higher is better:

```text
S = 100 * ((1 - reference_accuracy) / (1 - mean_validation_accuracy))
    * (reference_latency / latency)**alpha
    * (reference_bytes / deployed_bytes)**beta
```

The primary weights are **`alpha=0.25`, `beta=0.10`**, frozen before a campaign. The d3 reference scores **100**;
its identity and historical normalization are [below](#current-reference-and-provisional-leader). Changing the
reference only rescales a compatible cohort; it does not change its order. Changing weights changes the
preference and must be reported as a sensitivity analysis or a separately versioned objective, not used to
choose whichever ranking makes a candidate win.

These are declared preferences, not fitted constants or a universal measure of model quality. Their practical
meaning at d3's historical mean accuracy of 94.9338% is:

| Cost increase, everything else unchanged | Relative error reduction to break even | Accuracy gain |
|---|---:|---:|
| Double deployed bytes | 6.70% | 0.34 percentage points |
| Double latency | 15.91% | 0.81 percentage points |
| Double both | 21.54% | 1.09 percentage points |

A larger model can win if its accuracy gain pays for the cost. A smaller or faster model can win without higher
accuracy. Neither outcome is gaming the metric. Changing the weights can change the winner; that is an expected
property of a trade-off, not evidence that a scalar objective is unusable.

The implementation is [`cifarq.efficiency`](../../src/cifarq/efficiency.py), with no training, GPU or queue imports:

```python
from cifarq.efficiency import Measurement, score

reference = Measurement(accuracy=0.95, latency=10, size=1_000_000)
candidate = Measurement(accuracy=0.96, latency=20, size=1_000_000)
print(100 * score(candidate, reference))  # approximately 105.1; reference is 100
```

For compatibility, the Python API and CLI still return the dimensionless ratio with reference **1**, metric ID
`relative-error-cost-v1`. The leading factor of 100 is a reporting convention, not a new formula or a percentage
accuracy. Existing JSON documents preserve their original references, so multiplying their scores by 100 alone
does not rebase them to d3.

Accuracy must be a fraction in `[0, 1)`. Zero observed error makes inverse error undefined and is rejected rather
than silently capped. Costs must be finite and positive. Missing costs are accepted only when their exponent is
zero. Weights must be finite and nonnegative. This metric uses classification error, not F1; substituting another
metric changes its meaning.

Primary comparisons require deployed bytes. Parameter count is permitted only as an explicitly labelled
historical proxy; it cannot represent quantization, serialization overhead or a tree/feature pipeline fairly.
Keep units consistent with the reference. Changing units consistently does not change the score.

### Complete-pipeline measurement protocol

The primary serving regime is **one-view CIFAR-10, batch 256, on one fixed T4-equipped host**. Batch 1 remains a
separate secondary cohort; do not select whichever batch makes a candidate win. One view means no test-time
mirrors or shifts. Other policies require a separately declared cohort with matched reference measurements.

| Input to the score | Required measurement |
|---|---|
| Accuracy | Top-1 validation accuracy on the fixed 5000-image split, using the deployed precision and inference implementation. For a recipe, use the arithmetic mean across its declared independent seed runs, not the best seed. |
| Latency | Warmed end-to-end milliseconds per batch, from decoded host-resident uint8 RGB images to host-available class predictions. Include preprocessing, feature extraction, device transfers and all prediction/aggregation work. Synchronize asynchronous execution before stopping the timer. |
| Bytes | Total serialized inference-artifact bytes: all model weights, trees, fitted scalers, PCA bases, vocabularies, configuration and any learned feature extractor. Exclude training-only checkpoints/optimizer state and shared runtime installations. Record the serialization format. |

Exclude disk/network input loading, model loading and compilation from the warmed latency metric for every
candidate; report startup cost separately if relevant. Fix hardware allocation, CPU thread limits, runtime versions,
batch size, warmup/repetition policy and the latency statistic before measurement. Start with median batch latency,
20 warmups and 100 timed requests per process, repeat across fresh processes and retain the timing distributions.
Benchmark without competing workloads. The deployment precision may differ by candidate (for example
quantization), but its accuracy, time and bytes must all describe that same deployed configuration.

For multi-seed recipes, sample each seed equally in timing measurements and report the median of those request
times. Use mean artifact bytes across seeds if sizes differ, retaining the individual sizes and range. Shared
graph timings are only a labelled approximation when seed-dependent execution could change the cost.
Stored bytes are not peak RAM/VRAM; report memory separately and enforce a limit only if deployment requires it.

This boundary supports different model families without parameter-count or MAC conversions:

| Pipeline | Work and artifacts that must not be hidden |
|---|---|
| d3 or fine-tuned ResNet-50 | Full backbone and classifier, resizing/normalization and device transfers; count all weights, not just fine-tuned layers. |
| XGBoost on engineered features | Feature extraction, fitted transforms, tree inference and all fitted artifacts. Fit transforms on training data only. |
| XGBoost on neural embeddings | Both the embedding network and the trees, including feature-extraction latency and weights. |
| Ensemble | Every member, any gating/router, and the final aggregation. |

Start every candidate from the same raw-image boundary; cached evaluation features are not free. Neural
parameters and MACs are useful diagnostics where meaningful, but are not required by the primary score.
Disclose external pretraining data and training/search compute. This score measures inference utility, not
training efficiency; a pretrained winner is not proof of better learning from CIFAR-10 alone.

### Implementation status and evidence boundary

`cifarq.efficiency` is model-agnostic: it only consumes accuracy, latency and size. The current measurement tool,
`pipeline/tools/rank_saved.py`, is **airbench-specific** and times device-resident forward passes. It records
float32 accuracy alongside float16 accuracy, but the historical scored rows use float32 accuracy with float16
latency. It does not yet implement the end-to-end cross-family protocol above.

Consequently, `recorded.json` and `measured.json` remain **historical/provisional cohorts**, not a complete-pipeline
leaderboard. Before cross-family promotion, add the needed measurement adapters and remeasure the reference and
all candidates under the same end-to-end protocol, with accuracy at deployed precision. Freeze those reference
measurements, software/hardware identities and dataset hash as a new cohort; do not overwrite or mix in old
forward-only timing numbers.

The legacy Optuna, training and champion scripts still use their original accuracy objectives. They can provide
screening candidates but do not by themselves establish a primary-metric champion. This adoption changes the
repo's evaluation policy, not those historical studies or the published model.

The new **ResNet-18 campaign** implements `cifarq.pipeline_score` for complete-pipeline evaluation and
`cifarq.resnet18_campaign` for score-maximizing Optuna. Its separate
[`resnet18-baselines.json`](resnet18-baselines.json) cohort remeasures d3's deployed-fp16 validation accuracy
over 16 seeds and host-to-host batch-256 latency in three fresh processes. Reference mean accuracy is
94.93375%, sample seed SD 0.18341 percentage points, latency 3.70476 ms and size 7,210,007 bytes.
Do not substitute the historical 3.562 ms forward-only value into this cohort.

The first seed-40 ResNet-18 baselines score **52.46** (ImageNet fine-tune, 93.30% accuracy) and **44.76**
(scratch, 92.16%). Both use 64x64 inputs and the ImageNet stem, require approximately 44.81 MB, and take
7.70-7.75 ms per host-to-host request. These practical recipes use different learning rates and epoch
counts, so the gap does not identify the causal effect of pretraining alone.

The [final ResNet cohort](resnet18-final.json), recorded 2026-11-17, confirms the selected ImageNet
baseline recipe at **93.72%**, 7.71945 ms, 44,805,696 bytes and score **55.9346** over fresh seeds 71-73.
Accuracy sample SD is 0.13115 percentage points. No completed sampled trial improved on the original
baseline's score. The study stopped after 18 COMPLETE and five per-trial timeouts (Optuna FAIL), which
triggered the frozen failure-streak guard before the 50-entry maximum. It is not an exhaustive
ResNet-18 comparison or a significance claim; see the [full result](../autoresearch/2026-11-16-resnet18/RESULT-E1.md).
The dependent feature-boosting study also finished: all 36 entries and three paired-seed checks completed
without failures. Its selected frozen ResNet embedding + train-only PCA64 + XGBoost recipe achieved
**93.3133%**, 8.45274 ms, mean 45,530,022.333 bytes and score **51.2721**. It did not improve on its
original encoder's classifier or d3. The [combined 42-row cohort](resnet18-boost-final.json) preserves
every boosting study measurement/configuration, confirmed aggregate and the ResNet/reference comparison;
the [full result](../autoresearch/2026-11-16-resnet18/RESULT-E2.md) describes the paired encoders and
171-file artifact audit. Both campaigns used a conservative 8.683 hours of the original 24-hour budget.

### Offline scoring

Rank compatible saved records:

```bash
python -m cifarq.efficiency docs/metrics/measured.json
python -m cifarq.efficiency docs/metrics/recorded.json
# Sensitivity only, not the primary objective:
python -m cifarq.efficiency docs/metrics/recorded.json --beta 0
```

The JSON input contains `cohorts`, each with `id`, `reference`, `protocol`, `sources` and `rows`; see
[`recorded.json`](recorded.json). Rows contain `id`, `accuracy`, `latency`, `size`, `accuracy_policy` and
`latency_policy`. A shared protocol declares dataset, validation split, hardware, precision, batch, timing scope
and size metric. The scorer rejects mismatched accuracy/latency policies, test-based selection, duplicate IDs and
unknown references. It preserves evidence metadata and ranks each cohort separately. Metadata is a caller
assertion: the scorer cannot detect falsely labelled hardware, policies or source numbers.

Freeze the benchmark, reference and weights before selection. Record raw metrics and uncertainty alongside
scores. The same formula can score historical cohorts, but those numbers are comparable only within their
declared protocols.

## Retrospective assessment, 2026-11-16

No retraining, GPU benchmarks, new test predictions or remote jobs were run.

### Preservation and provenance

Before restarting development, local work and all tracked/untracked research evidence were snapshotted at
`61f36e62961d3c2634c17f8b8bb09632b3483839`, branch `archive/autoresearch-2026-11-16`.
Local `master` was reset to tracked `origin/master`, `a5002474766a1f66093a2fb2ad3b1fc05b70fec3`.
This feature starts there, on `feat/error-cost-score`; it does not merge campaign implementation code.
A verified Git bundle and a separate archive of the existing ignored results were also saved outside the checkout.
The results backup contains only files present on this machine, not checkpoints referenced on other machines.

Every cohort in the input identifies source paths at the immutable archive commit. For example:

```bash
git show 61f36e62961d3c2634c17f8b8bb09632b3483839:docs/autoresearch/2026-11-12-readout-recipe-spectral/protocols/FINAL-TEST-thresholds.json
git show 61f36e62961d3c2634c17f8b8bb09632b3483839:docs/autoresearch/2026-11-12-readout-recipe-spectral/RESULT-V.md
```

Accuracy for the capacity comparison comes from the four validation checkpoints per arm, seeds 52-55, in the
threshold manifest. The means in the input were checked against that archived manifest. Timing and parameter
counts come from `RESULT-W.md` and `RESULT-F.md`. Timing semantics were checked in `experiments/g_gates.py`.
Policy-sweep means and relative times come from `RESULT-V.md`; timing semantics were checked in
`experiments/v_policy_sweep.py`. Source evidence is unchanged.

### Architecture comparison: one view, two serving regimes

The campaign's capacity table combines conditional-policy accuracy with latency from a one-view graph benchmark.
Those quantities are not matched deployment measurements. This assessment instead uses **one-view validation
accuracy** with those one-view timings. It does not claim a conditional-policy architecture ranking.

| Model | Validation accuracy | Stored parameters | Batch-256 ms | Batch-1 ms | Score, batch 256 | Score, batch 1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 94.500% | 1,113,336 | 2.630 | 0.330 | **1.0000** | **1.0000** |
| d3 | 95.075% | 1,798,488 | 3.409 | 0.462 | 0.9976 | 0.9786 |
| d3_e150 | 94.880% | 1,798,488 | 3.409 | 0.462 | 0.9596 | 0.9413 |
| w_200 | 95.305% | 4,424,376 | 6.215 | 0.470 | 0.8231 | 0.9341 |

These are the default weights, with **stored parameters substituted explicitly for size**, because deployed bytes
for these checkpoints are unavailable locally. They are not byte-based scores. Timings are warmed FP16 T4
device-resident forward passes, including normalization and softmax, not host-to-host serving latency. The bench
uses random weights of the correct graph shape, one process per architecture, and published medians rounded to
three decimal places. `d3_e150` inherits `d3`'s timing because changing training epochs leaves its inference graph
unchanged.

**Interpretation:** default weights price away the accuracy gains of increased capacity. Baseline beats `d3` by
only 0.24% in score at batch 256: treat this as unresolved, not evidence that baseline is better. The historical
seed floor was about 0.183 accuracy percentage points, and repeated-process timing uncertainty is unavailable.
The scoring formula provides no confidence interval by itself.

Removing the size penalty (`beta=0`, same `alpha=0.25`) changes the result:

| Model | Score, batch 256 | Score, batch 1 |
|---|---:|---:|
| baseline | 1.0000 | 1.0000 |
| d3 | **1.0466** | 1.0267 |
| d3_e150 | 1.0068 | 0.9876 |
| w_200 | 0.9448 | **1.0723** |

This recovers the useful serving-regime distinction: depth for batch throughput, width for single-image latency
when size is not penalized. No accuracy or cost observations changed; only the declared preference changed.

### Policy comparison: fixed weights

Ten settings survive in the published 21-setting sweep, evaluated on the same twelve M0 checkpoints. The reference
is 15% escalation to six views. Size is **relative model bytes = 1** for every row, because all policies reuse the
same weights. No absolute byte measurement is invented.

| Escalation | Views | Validation accuracy | Time / reference | Default score |
|---|---:|---:|---:|---:|
| 0% | 1 | 94.308% | 0.535 | 1.0076 |
| 5% | 6 | 94.852% | 0.689 | **1.0458** |
| 10% | 6 | 95.055% | 0.844 | 1.0349 |
| 15% | 6 | 95.095% | 1.000 | 1.0000 |
| 25% | 6 | 95.112% | 1.313 | 0.9374 |
| 40% | 6 | 95.115% | 1.772 | 0.8703 |
| 100% | 6 | 95.115% | 3.634 | 0.7272 |
| 15% | 2 | 94.828% | 0.692 | 1.0398 |
| 15% | 18 | 95.002% | 2.104 | 0.8149 |
| 40% | 18 | 95.010% | 4.729 | 0.6666 |

The metric prefers 5%/6 among **these ten recorded settings**, accepting 0.243 percentage points less accuracy
for 31.1% less reported time. This is not a claim about all 21 settings or an automatic policy promotion.

Timing has an important limitation: the source adds separately timed predictor passes and scales padded batches
by their real-image fraction. It excludes margin selection and merge overhead and is not end-to-end wall time.
Relative times above are reconstructed from rounded reported percentages. This is a useful retrospective
component-cost comparison, not a production latency result.

### Sensitivity and conclusion

| Latency weight alpha | Size weight beta | Best batch-256 architecture | Best batch-1 architecture | Best recorded policy |
|---:|---:|---|---|---|
| 0 | 0 | w_200 | w_200 | 40%/6 and 100%/6 tie at published precision |
| 0.10 | 0.10 | d3 | d3 | 10%/6 |
| 0.25 | 0 | d3 | w_200 | 5%/6 |
| 0.25 | 0.10 | baseline, near-tied with d3 | baseline | 5%/6 |
| 0.50 | 0.10 | baseline | baseline | one view, no escalation |

The score behaves as intended: it rewards lower error at unchanged cost, penalizes larger/slower models, and
rejects expensive TTA with negligible quality gain. It also exposes how consequential the weights are. The
proposed defaults are usable starting preferences, not validated deployment utility.

Excluded from this ranking: historical candidates and the MAC-based Pareto front without matched latency/bytes;
untrained latency-gate arms without measured trained accuracy; and capacity-policy pairs without matching policy
timings. Existing test results were not used as selection inputs.

Next useful measurement requires no retraining: retrieve the saved finalists and measure actual deployed bytes
plus end-to-end latency for the chosen policy on one fixed target and batch regime. Then freeze weights according
to the acceptable error-for-cost trade, and quantify score uncertainty with paired seeds and repeated timings.
Until then, keep defaults unchanged and treat the near ties and component-timing rankings as provisional.

## Measured assessment from saved checkpoints, 2026-11-16

This pass measured all saved checkpoints without retraining, replacing random-weight graph timings and
parameter-count size proxies with trained-weight timings and artifact bytes. It did **not** complete the proposed
end-to-end serving measurement: latency still starts with device-resident images. Keep this cohort separate from
future complete-pipeline comparisons.

`python pipeline/tools/rank_saved.py` performs the measurement and writes [`measured.json`](measured.json);
`--report` ranks that file again without a GPU, and `--rebuild` also reconstructs the cohorts from it.

### Protocol

| Quantity | How it was measured |
|---|---|
| Accuracy | The checkpoint's own trained weights over all 5000 validation images, one view, no escalation, float32. Float16 accuracy is recorded alongside it for precision comparison. |
| Size | `footprint.static()` bytes of `model.safetensors` plus `config.json`, not a parameter-count proxy. |
| Compute | MACs per image, traced from the inference jaxpr. |
| Latency | Device-resident forward pass including normalisation and softmax, float16, on one Tesla T4, median of 100 passes after 20 warmups, at batch 256 and batch 1. This matches the campaign's own `g_gates.py` timing loop. |
| Data | `dataset-56cff0f67835.npz`, whose sha256 is the protocol hash recorded above. The validation split only; the test split was not opened. |

189 checkpoints resolve to 19 distinct inference graphs, so latency is measured once per graph and shared by the
seeds that use it. Both a per-checkpoint and a per-arm cohort are written at each batch.

The measurement reproduces the campaign records it was run against: of the 189 checkpoints, 104 match their
recorded `val_acc` exactly and none deviates by more than 0.04 points, including the one diverged run
(`operator-structure/xca` seed 30) that is reproduced at its recorded 9.40 %. The four `w_200` final-stage seeds
52-55 average 0.9531 against the 0.95305 archived in `recorded.json`; this is not the 16-seed arm mean below.

### Ranking by arm, default weights (alpha 0.25, beta 0.10)

Read the per-arm cohort for recipe comparisons rather than selecting the best checkpoint. The median seed
standard deviation across these arms is about 0.18 accuracy points. This is variability, not a fixed significance
threshold: paired comparisons and repeated runs can resolve smaller effects.

The table preserves the original reference (`operator-structure/baseline`) at 1. The
[current reporting table](#current-reference-and-provisional-leader) rebases the same observations to d3 at 100.

Batch 256:

| Arm | Seeds | Val % | Seed sd | ms | Score |
|---|---:|---:|---:|---:|---:|
| spectral/`baseline` | 4 | 94.48 | 0.16 | 2.631 | 1.0281 |
| spectral/`scaling_factor_6` | 3 | 94.47 | 0.04 | 2.640 | 1.0247 |
| spectral/`kd_w030` | 3 | 94.46 | 0.09 | 2.631 | 1.0244 |
| spectral/`dynamics_scaling6` | 12 | 94.44 | 0.17 | 2.640 | 1.0198 |
| spectral/`dynamics` | 3 | 94.43 | 0.12 | 2.631 | 1.0182 |
| spectral/`kd_disagree` | 3 | 94.41 | 0.14 | 2.631 | 1.0158 |
| spectral/`kd_w030_ls010` | 3 | 94.34 | 0.07 | 2.631 | 1.0026 |
| spectral/`kd_ens030` | 3 | 94.33 | 0.21 | 2.631 | 1.0015 |
| structure/`baseline` (reference) | 4 | 94.33 | 0.17 | 2.631 | 1.0000 |
| spectral/`d3` | 16 | 94.93 | 0.18 | 3.562 | 0.9898 |

Batch 1 reorders the leaders slightly, `scaling_factor_6` taking the top on its marginally cheaper graph, and
drops `d3` further; the ordering is otherwise the same.

**The point-estimate leader is spectral `baseline`, not d3.** Nearby recipe arms are close, but this summary
alone does not establish either superiority or equivalence. The file lacks repeated-process latency uncertainty,
and seed standard deviation is not an uncertainty interval on a score difference. Capacity increases do raise
observed mean accuracy (`d3` +0.60, `w_200` +0.94, `teacher` +1.53 points over this reference), and the score
charges for their additional cost. These are descriptive differences, not significance claims.

### Sensitivity, measured arms

| alpha | beta | Best at batch 256 | Best at batch 1 |
|---:|---:|---|---|
| 0 | 0 | `teacher` | `teacher` |
| 0.10 | 0.10 | `d3` | `d3` |
| 0.25 | 0 | `d3` | `teacher` |
| 0.25 | 0.10 | spectral/`baseline` | `scaling_factor_6` |
| 0.50 | 0.10 | `iso_width` | `scaling_factor_6` |

Different preferences favor different points on the observed accuracy-cost frontier. This sensitivity is why the
primary weights are fixed in advance, not a reason to discard the metric. NetScore remains a secondary comparison;
neither retuning exponents after seeing a result nor switching metrics can establish a primary-score win.

## NetScore (secondary comparison)

[NetScore](https://arxiv.org/abs/1806.05512) (Wong, 2018) is a published collapsed accuracy/cost metric, kept
here for comparison, not as the campaign objective. `python -m cifarq.netscore docs/metrics/measured.json`
ranks the same evidence by

```
Omega = 20 * log10(a^alpha / (p^beta * m^gamma)),  alpha = 2, beta = gamma = 0.5
```

with accuracy `a` in percent, parameters `p` in millions and `m` in billions of MACs. The units are part of the
definition and the module converts to them internally, so call sites pass repo-native values (accuracy as a
fraction, raw parameter and MAC counts). Unlike the error-cost score it needs no reference model and returns a
value in decibels under the stated unit convention. The published defaults encode a particular preference;
the implementation also permits other exponents. Information density, `a / p` in percent per million parameters,
is reported beside it. Do not invent parameter or MAC equivalents for tree/feature pipelines merely to fill a
NetScore column.

| Arm | Val % | Params | MMACs | MiB | NetScore (dB) |
|---|---:|---:|---:|---:|---:|
| spectral/`iso_width` | 92.86 | 474,781 | 32 | 1.8 | 96.96 |
| t4/`compact-scratch` | 92.91 | 785,400 | 53 | 3.0 | 92.52 |
| t4/`width-early` | 93.19 | 795,896 | 53 | 3.0 | 92.51 |
| t4/`donor-assisted` | 92.49 | 785,400 | 53 | 3.0 | 92.44 |
| ... | | | | | |
| spectral/`w_150` | 94.95 | 2,494,104 | 148 | 9.5 | 83.44 |
| spectral/`w_200` | 95.27 | 4,424,376 | 256 | 16.9 | 78.63 |
| t4/`teacher` | 95.86 | 9,918,776 | 452 | 37.9 | 72.75 |

NetScore's default 0.5 exponents on parameters and MACs strongly favor compact models in this accuracy range:
`iso_width` leads while `teacher`, the most accurate saved arm here, trails it by about 24 dB. The primary
error-cost score instead favors the 1.11 M baseline family. This reflects different trade-offs; it does not
invalidate either calculation. Primary-score promotion does not require also winning NetScore.

### Recovered modules

Three modules some saved T4-campaign checkpoints import — `attention`, `reparam` and `subspace` — are not on
`master`. Their source is in git on the unmerged `integrate/t4-campaigns` branch (`4fd9f16`, with the airbench
grafts that call them). When `measured.json` and the ResNet-18/boosting jobs ran, the measurement host had them only
as `__pycache__` bytecode, copied to `src/cifarq/` as sourceless Python 3.13 modules; the job manifests record their
hashes. `master`'s airbench never imports them, so the d3 reference and the `pipeline_score` evaluator do not need
them. Remeasuring the whole back-catalogue with `rank_saved.py` does: run it from that branch.

## Current reference and provisional leader

**Reference:** the `readout-recipe-spectral/d3` recipe, baseline `depth: 2 -> 3`, widths `[48, 192, 192]`,
trained for 200 epochs at batch 1024. The saved arm has 16 independent seeds (40-55), mean one-view validation
accuracy **94.9338%**, seed sd about **0.18 percentage points**, and 1,798,488 parameters. Its historical graph
timings are 3.562 ms at batch 256 and 0.460 ms at batch 1; these are not end-to-end reference timings.

The representative checkpoint closest to that mean is
`results/autoresearch/readout-recipe-spectral/w-capacity/attempts/20261113T055245Z-d3-s43/model/`
(94.94% validation). It is an experimental starting point, not the source of the arm's mean. The reference
recipe and seed identities stay fixed when establishing a new measurement protocol; freeze its newly measured
accuracy, latency and bytes before searching. Do not substitute the best seed or update the normalizer after
each improvement.

**Provisional score leader:** spectral `baseline`, under the saved batch-256 protocol. The following table is
the existing `measured-arm-one-view-b256` cohort rebased to d3 at 100, not a new measurement:

| Arm | Mean validation accuracy | Score (d3 = 100) |
|---|---:|---:|
| spectral `baseline` | 94.48% | **103.86** |
| spectral `d3` | 94.93% | **100.00** |
| spectral `iso_width` | 92.86% | **98.84** |
| spectral `w_200` | 95.27% | **84.05** |
| t4 `teacher` | 95.86% | **78.03** |

The normalization is not the promotion threshold: exceeding 100 beats the reference, not necessarily the
incumbent. Compare a challenger against the best confirmed score under the same protocol. The current table is
provisional and provides no cross-family end-to-end champion.

Reproduce the d3-normalized ranking without loading a model or modifying the saved evidence:

```bash
python - <<'PY'
import json
from pathlib import Path
from cifarq.efficiency import assess

saved = json.loads(Path("docs/metrics/measured.json").read_text())
cohort = next(c for c in saved["cohorts"] if c["id"] == "measured-arm-one-view-b256")
reference = next(r for r in cohort["rows"] if r["campaign"] == "readout-recipe-spectral" and r["arm"] == "d3")
cohort["reference"] = reference["id"]
ranked = assess({"cohorts": [cohort]}, alpha=0.25, beta=0.10)
for row in ranked["cohorts"][0]["rows"]:
    print(f'{row["campaign"]}/{row["arm"]}: {100 * row["score"]:.2f}')
PY
```

### Campaign selection and promotion

Train models and fit preprocessing on the **training** split; evaluate and select on validation. Do not train on
validation or tune on the test set. Preserve prior test exposures: the historical test reports are not untouched
holdout evidence for designs they influenced. External pretraining requires disclosure and a declared data policy.

The search objective is the primary score, not accuracy alone, NetScore, a lower confidence bound on accuracy, or
a requirement to improve every raw cost. Cheap accuracy-only screens are allowed as explicitly labelled proxies,
but finalist ranking and promotion require measured primary scores. Leave higher-capacity regimes eligible when
their accuracy could repay their cost.

Before a campaign, freeze the protocol, reference, weights, data policy, compute allowance, confirmation seeds,
and uncertainty/promotion rule. Start with inexpensive screens, then confirm a short list with repeated runs.
Use paired seeds and evaluation examples where useful, fresh confirmation seeds after selection, and repeated
latency measurements. Report means, seed variability and an uncertainty interval for the **score difference**
against the incumbent; distinguish training/timing uncertainty from finite-validation-set generalization uncertainty.

There is no universal +0.15/+0.20 accuracy threshold, mandatory 5/16-seed count, or fixed variance cutoff.
Size confirmation to the observed paired variability and effect worth detecting. A point-estimate improvement is
promising until the declared confirmation rule supports it. A smaller model can win through efficiency; a larger
one can win through accuracy. Disagreement with NetScore does not veto a primary-score improvement.

Repeated search on one validation split can overfit it; subtracting `2 * SE` does not remove that selection
bias. Keep any genuinely unexposed evaluation data for a predeclared final assessment and do not feed that
result back into the same selection claim.

Hard gates are for correctness, data leakage, and real deployment limits such as available memory, not blanket
caps at d3's bytes, MACs or both serving latencies. Training spend is a separate campaign budget; equal epochs
do not imply equal compute. Changing weights, a serving protocol or a budgeted task creates a separately named
comparison rather than retroactively promoting an old result.

### Research context

[MnasNet](https://arxiv.org/abs/1807.11626), Section 3, uses a weighted accuracy-latency product with
application-specific weights. It supports declaring a trade-off, not transferring its numerical weights blindly
to CIFAR-10 or tuning ours to crown d3. This repo uses error reduction to make gains near 95% accuracy meaningful.
[Cawley and Talbot (2010)](https://jmlr.org/papers/v11/cawley10a.html) describe overfitting in model selection:
the scalar objective and the independent assessment of a selected winner are separate concerns.
