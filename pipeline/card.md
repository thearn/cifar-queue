---
license: $license
library_name: pytorch
pipeline_tag: $pipeline_tag
tags: [$tags]
datasets: [$dataset]
metrics: [accuracy]
model-index:
- name: $name
  results:
  - task: {type: $pipeline_tag, name: Image Classification}
    dataset: {type: $dataset, name: CIFAR-10, split: test, revision: $revision}
    metrics:
    - {type: accuracy, value: $test_acc, name: Test accuracy}
---
{{README_INTRO}}
## The published model

The weights here are the champion of the search ($source) retrained over $n_seeds seeds,
keeping the seed with the median validation accuracy (seed $seed: $val_pct validation,
$test_pct test). Over the seeds the test accuracy was $test_mean +-
$test_stdev.

| Seed | Validation | Test | GPU |
|---|---|---|---|
$runs

`hyperparameters.json` is the full training configuration.$demo The dataset is `$dataset_file` (sha256
`$dataset_sha`), built at commit `$commit` of the code in this repository.

{{README_REST}}
