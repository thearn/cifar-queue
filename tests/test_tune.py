"""The search space against the training function (no GPU): every sampled point is a valid set of hyperparameters,
every optional feature can be sampled off, the queue options grow with the trial's size, a trial of a study made before
schedule-free training still maps to train's arguments, and a short search runs end to end in this process."""

import optuna
import pytest

from cifarq import train, tune


def test_the_space_samples_valid_hyperparameters_with_every_feature_off_somewhere():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=0))
    seen_off = set()
    for _ in range(400):
        t = study.ask()
        p = tune.space(t)
        assert not set(p) - set(train.DEFAULTS)
        seen_off |= {
            k
            for k in (
                'aug_translate',
                'aug_rotate',
                'aug_scale',
                'aug_brightness',
                'aug_contrast',
                'aug_saturation',
                'cutout_size',
                'grad_clip',
                'ema_decay',
                'dropout',
                'label_smoothing',
            )
            if not p[k]
        }
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


def test_a_start_point_is_checked_against_the_space():
    assert tune.misfit(dict(tune.PAGE)) == []
    bad = {**tune.PAGE, 'width': 7, 'not_a_knob': 1}
    bad.pop('epochs')
    problems = tune.misfit(bad)
    assert any(p.startswith('epochs: missing') for p in problems)
    assert any(p.startswith('width: 7 outside') for p in problems)
    assert 'not_a_knob: not in the space' in problems


def test_the_idle_search_starts_from_the_best_known_eqairbench_points():
    import functools

    from cifarq import families

    space = functools.partial(families.eqaug_space, kd='teacher')
    assert [tune.misfit(p, space) for p in families.WARM['eqaug']] == [[], []]


def test_the_standing_ranks_complete_trials_and_counts_every_state():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    for v in (0.90, 0.95, 0.93):
        t = study.ask({'x': optuna.distributions.FloatDistribution(0, 1)})
        study.tell(t, v)
    study.tell(study.ask(), state=optuna.trial.TrialState.PRUNED)
    study.tell(study.ask(), state=optuna.trial.TrialState.FAIL)
    study.ask()  # running
    rows, counts = tune.standing(study.trials, n=2)
    assert [r['value'] for r in rows] == [0.95, 0.93] and set(rows[0]['params']) == {'x'}
    assert counts == {'complete': 3, 'pruned': 1, 'failed': 1, 'running': 1}


def test_the_augmented_front_space_holds_every_eqaug_point():
    import functools

    from cifarq import families

    space = functools.partial(families.frontaug_space, kd='teacher')
    assert [tune.misfit(p, space) for p in families.WARM['eqaug']] == [[], []]
    assert tune.misfit({**families.WARM['eqaug'][0], 'augment': 'randaugment', 'augment_m': 30, 'w1': 48}, space) == []


def test_the_second_space_starts_from_the_first_ones_best_and_takes_its_fitting_trials():
    import functools
    import sys
    from pathlib import Path

    from cifarq import families

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
    from common import known_point

    space = functools.partial(families.eqaug2_space, kd='teacher')
    assert [tune.misfit(p, space) for p in families.WARM['eqaug2']] == [[], [], []]
    best = {**families.WARM['eqaug2'][0], 'srf': True}
    assert known_point(best, space) == families.WARM['eqaug2'][0]  # (srf: fixed at the same value, dropped)
    assert known_point({**best, 'srf': False}, space) is None  # fixed at another value
    assert known_point({**best, 'augment': 'none'}, space) is None  # no longer a choice


def test_the_third_space_settles_augmentation_and_reads_an_exported_study(tmp_path):
    import functools
    import json
    import sys
    from pathlib import Path

    from cifarq import families

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline' / 'tools'))
    from common import completed_trials, known_point
    from export_trials import rows

    space = functools.partial(families.eqaug3_space, kd='teacher')
    assert [tune.misfit(p, space) for p in families.WARM['eqaug3']] == [[], [], []]
    old = {**families.WARM['eqaug2'][0]}  # (RandAugment, cutout 8: fixed at the same values, dropped)
    assert known_point(old, space) == {k: v for k, v in old.items() if k not in ('augment', 'cutout')}
    assert known_point({**old, 'cutout': 12}, space) is None
    assert space(optuna.trial.FixedTrial({**families.WARM['eqaug3'][0], 'batch_size': 128}))['batch_size'] == 128
    fixed = optuna.trial.FixedTrial(families.WARM['eqaug2'][0])
    families.eqaug2_space(fixed, kd='teacher')
    t = optuna.trial.create_trial(params=families.WARM['eqaug2'][0], distributions=fixed.distributions, value=0.95, user_attrs={'device': 'x'})
    t.number = 7
    out = tmp_path / 'trials.jsonl'
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows([t])))
    back = completed_trials(out)
    assert [(b.number, b.value, b.params, b.user_attrs) for b in back] == [(7, 0.95, t.params, {'device': 'x'})]
    stopped = optuna.trial.create_trial(
        state=optuna.trial.TrialState.PRUNED, params=t.params, distributions=fixed.distributions, intermediate_values={4: 0.8}
    )
    stopped.number = 8
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows([t, stopped])))
    assert [b.number for b in completed_trials(out)] == [7]
    back = completed_trials(out, pruned=True)
    assert [(b.number, b.state.name, b.intermediate_values) for b in back] == [(7, 'COMPLETE', {}), (8, 'PRUNED', {4: 0.8})]


def test_over_the_cap_points_still_fit_and_seed_as_pruned(tmp_path):
    import functools
    import sys
    from pathlib import Path

    from cifarq import families

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
    from common import known_point

    space = functools.partial(families.eqaug3_space, kd='teacher')
    wide = {**families.WARM['eqaug3'][0], 'w1': 200, 'w2': 192, 'w3': 224}
    with pytest.raises(optuna.TrialPruned):
        space(optuna.trial.FixedTrial(wide))
    assert tune.misfit(wide, space) == []
    assert known_point({**wide, 'augment': 'randaugment'}, space) == wide
