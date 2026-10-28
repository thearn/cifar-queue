"""The search space against the training function (no GPU): every sampled point is a valid set of hyperparameters,
every optional feature can be sampled off, and the queue options grow with the trial's size."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import optuna  # noqa: E402

from cifarq import train, tune  # noqa: E402


def test_the_space_samples_valid_hyperparameters_with_every_feature_off_somewhere():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=0))
    seen_off = set()
    for _ in range(400):
        t = study.ask()
        p = tune.space(t)
        assert not set(p) - set(train.DEFAULTS)
        seen_off |= {k for k in ('aug_translate', 'aug_rotate', 'aug_scale', 'aug_brightness', 'aug_contrast',
                                 'aug_saturation', 'cutout_size', 'grad_clip', 'ema_decay', 'dropout', 'label_smoothing')
                     if not p[k]}
        seen_off |= {'mix'} if p['mix'] == 'none' else set()
        study.tell(t, 0.0)
    assert len(seen_off) == 12


def test_queue_options_grow_with_the_trial():
    small, big = tune.job_options({'width': 16, 'epochs': 6}), tune.job_options({'width': 64, 'epochs': 20})
    assert small['minutes'] < big['minutes'] and small['gpu_mb'] < big['gpu_mb']


def test_pages_recipe_is_a_point_of_the_space():
    study = optuna.create_study(direction='maximize')
    study.enqueue_trial(tune.PAGE)
    p = tune.space(study.ask())
    assert p['epochs'] == 24 and p['width'] == 64 and p['lr'] == 0.4 and p['logit_scale'] == 0.125
    assert p['cutout_size'] == 8 and p['decay'] == 'linear' and not set(p) - set(train.DEFAULTS)
