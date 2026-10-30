"""The search space against the training function (no GPU): every sampled point is a valid set of hyperparameters,
every optional feature can be sampled off, the queue options grow with the trial's size, a trial of a study made before
schedule-free training still maps to train's arguments, and a short search runs end to end in this process."""
import optuna

from cifarq import train, tune


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


def test_a_trial_from_before_schedule_free_maps_to_train_arguments():
    old = {k: v for k, v in tune.PAGE.items() if k != 'schedule_free'}
    assert tune.train_params(old) == tune.train_params(tune.PAGE)


def test_short_search(built):
    from conftest import SMALL
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=tune.DIRECTION, sampler=tune.sampler(startup=2))
    study.enqueue_trial(tune.REFERENCE)
    study.optimize(lambda t: train.train(**{**tune.space(t), 'epochs': 1, **SMALL})[tune.OBJECTIVE], n_trials=3)
    assert len(study.trials) == 3 and study.best_value > 0.1
