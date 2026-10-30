"""The session's search spaces: every warm point is a point of its space under the cap, sampled configurations over
the cap are pruned at step 0 without training, and the spaces reach both sides of the cap."""
import optuna
import pytest

from cifarq import airbench, families


@pytest.mark.parametrize('family', sorted(families.SPACES))
def test_warm_points_are_in_the_space_and_under_the_cap(family):
    for point in families.WARM[family]:
        p = families.SPACES[family](optuna.trial.FixedTrial(point))
        h = {**airbench.DEFAULTS, **{k: v for k, v in p.items() if k != 'param_cap'}}
        assert airbench.param_counts(airbench.build(h)[0])['stored'] <= 2_000_000


@pytest.mark.parametrize('family', sorted(families.SPACES))
def test_the_space_reaches_both_sides_of_the_cap(family):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    over = 0
    for _ in range(60):
        t = study.ask()
        p = families.SPACES[family](t)
        h = {**airbench.DEFAULTS, **{k: v for k, v in p.items() if k != 'param_cap'}}
        over += airbench.param_counts(airbench.build(h)[0])['stored'] > p['param_cap']
        study.tell(t, 0.0)
    assert over < 60 and (over > 0 or family == 'resnet9')    # (resnet9's widths stop under the cap)


def test_over_the_cap_is_pruned_without_training():
    r = airbench.train(widths=(128, 384, 512), depth=3, param_cap=2_000_000, data='/nonexistent')
    assert r['pruned'] and r['over_cap'] and r['progress'] == 0 and r['params'] > 2_000_000
