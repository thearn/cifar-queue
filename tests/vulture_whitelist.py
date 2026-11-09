"""Names vulture reports as unused that are used, each with the reason (tests/test_style.py reads this file beside the
source). Add a name here only with its reason; a name nothing uses is deleted instead."""

# ruff: noqa: B018
from cifarq import airbench, data, families, hub, methods, nets, pbt, train, tune  # noqa: F401

# queue jobs name these functions by string ('module:function'): pipeline/57_front.py, 55_tune_on_vm.py, tools/pbt_g4.py
families.train_front
tune.on_vm
tune.on_vm_ray
pbt.run_pbt

# the airbench presets pipeline/tools/seeds.py reads by name
airbench.AIRBENCH94
airbench.AIRBENCH96

# the hold-out's one evaluation, run by hand on a final model (its log, data/holdout/looks.jsonl, is the record)
data.holdout.look


# Ray Tune calls a callback's hooks with these parameters
tune.on_trial_complete
tune.on_trial_error
tune.iteration
tune.info

# pipeline/common.py's Direct executor: runq's search reads its slots (hpo.concurrency_of)
Direct = None
Direct.slots

# huggingface_hub's ModelHubMixin calls these hooks (save_pretrained, from_pretrained)
hub.ResNet9._save_pretrained
hub.ResNet9._from_pretrained

# a queue job names it as a string (tune.TRAIN_SEEDS: pipeline/60_champion.py)
train.train_seeds

# PyTorch reads it (the Torch profiling harness: cudnn picks its fastest convolution algorithms)
torch = None
torch.backends.cudnn.benchmark
