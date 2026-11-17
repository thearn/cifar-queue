# Score-driven ResNet-18 and feature boosting: final summary

Completed 2026-11-17. **Neither tested new family beats the fixed d3 reference.**
Both local queue jobs and their fresh-seed checks finished; no work remains
queued. Nothing was pushed, published or merged into a default branch.

| Complete-pipeline recipe | Validation accuracy | ms / 256 | MB | Primary score |
|---|---:|---:|---:|---:|
| d3, 16-seed reference | 94.934% | 3.705 | 7.21 | 100.00 |
| Selected ImageNet ResNet-18, three fresh seeds | 93.720% | 7.719 | 44.81 | 55.93 |
| Selected ResNet embedding/PCA/XGBoost, three paired seeds | 93.313% | 8.453 | 45.53 | 51.27 |

The original canonical-stem, 64px, 30-epoch ImageNet ResNet recipe remained the
best completed E1 configuration. None of its 16 completed sampled alternatives
improved on the initial baseline score. E2 selected a 64-component train-only
PCA followed by XGBoost with 50 rounds/depth 9, but all three paired encoder
seeds lost accuracy relative to the original neural classifier.

## Outcomes and unique-job cost

| Experiment | Queue job | Scoped outcome | Completed study entries | Other outcomes | Conservative wall charge |
|---|---:|---|---:|---|---:|
| E0, reference/smoke | Within 47 | Validated evaluator/reference capability, not a model gain | Not a search | Three reference processes; saved-model GPU smoke | Included in E1 |
| E1, ResNet-18 | 47 | Rejected as a gain for completed recipes; wider search remains incomplete | 18, including two baselines | Five per-trial timeouts; three successful fresh-seed confirmations | 30,510.021 s |
| E2, feature boosting | 48 | Rejected as a gain for tested representations/recipes | 36, including six baselines | Zero failures; three successful paired-seed confirmations | 749.943 s |
| Total | Two distinct job IDs | No new promoted model | 54 | Six confirmations; five incomplete candidates | **31,259.965 s / 8.683 h** |

The original total limit was 24 hours, with E2 additionally capped at four hours
of unused allowance. CPU stages and recovery spending are included. Runq's raw
ledger contains repeated progress entries for the same job; adding their
per-entry cost fields would double-count that job. The table above uses each
driver's final budget record once, not the raw per-entry rollup.

## Evidence and uncertainty

- [E1 result](RESULT-E1.md): all 18 completed study entries, exact stop reason,
  failed/budget-stopped candidates, selected recipe and seed outcomes. The
  five-timeout safeguard stopped the search before the 50-entry maximum;
  this was **not a completed 50-trial study**.
- [E2 result](RESULT-E2.md): all feature/learner families, selected configuration,
  paired seed outcomes, precision/encoder/cache provenance and artifact audit.
  All 36 entries completed.
- [Machine-readable final comparison](../../metrics/resnet18-boost-final.json):
  42 scored rows with roles, raw evidence, configurations and hashes.
  [Figure](../../metrics/resnet18-boost-final.svg).

Accuracy sample SD is 0.13115 percentage points for the selected ResNet recipe
and 0.15275 for the selected tree pipeline. Recipe scores use mean seed error,
median of equally sampled request times and mean deployed bytes, rather than
mean nonlinear scores. Three fresh seeds are not a general significance claim.

These are complete one-view prediction pipelines on the same local T4, host
uint8 RGB to host class IDs, batch 256. All saved inference components count.
ImageNet-1K pretraining is disclosed for both selected new pipelines.
No new test read was made, and historical exposures remain disclosed.

The validated claims are the evaluator/reference capability (E0), the recorded
measurements (E1/E2), and successful infrastructure checks, not a general claim
that ResNet or boosting can never win. Historical forward-only compact-model
rankings are not imported into this cohort; d3 is its best measured entry,
not a proved champion over the entire back catalogue.

## Reproduce and preserve

```bash
PYTHONPATH=src .venv/bin/python -m cifarq.efficiency docs/metrics/resnet18-boost-final.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-boost/bin/python -m pytest tests/test_image_boost.py tests/test_boost_campaign.py \
  tests/test_pipeline_score.py tests/test_efficiency.py tests/test_netscore.py -q
```

Multiply the scorer's ratios by 100. Final focused tests: **53 passed**. All
36 boosting objective values and both confirmed recipe scores were checked
against raw measurements; 171 boosting deployment-file sizes/hashes matched.
Model artifacts, request/config files, SQLite studies, partial attempts and
budget records remain under `results/autoresearch/resnet18-score/` and
`results/autoresearch/image-boost/`, with stable paths/checksums in the records.

Frozen source identities remain `e5b6bb1` (E1) and `94fb7c8` (E2); the verified
50-entry control amendment is `8402d83`. The intentional coordinator restart
did not discard its active training candidate. No rejected attempts or branches
were deleted as cleanup.

## Next consequential decision

The [FoldGrid proposal](ARCHITECTURE-NEXT.md) supplies a cited JAX structural
track alongside compact-CNN distillation, but has no trained result. Before
running it, approve/freeze a separate bounded campaign, remeasure the historical
compact frontier under this protocol, and give incumbents the same legal
deployment optimizations. Unused time in the completed campaign is not treated
as authorization for a different architecture campaign.
