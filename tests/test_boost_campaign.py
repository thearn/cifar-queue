import pytest

from cifarq.boost_campaign import remaining_allowance, sample
from cifarq.image_boost import settings


def test_followup_never_adds_to_the_existing_approval():
    assert remaining_allowance({'spent_seconds': 2 * 3600}) == 4 * 3600
    assert remaining_allowance({'spent_seconds': 23 * 3600}) == 3600
    assert remaining_allowance({'spent_seconds': 24 * 3600}) == 0
    assert remaining_allowance({'spent_seconds': 25 * 3600}) == 0
    with pytest.raises(RuntimeError, match='still active'):
        remaining_allowance({'spent_seconds': 3600, 'active': True})
    with pytest.raises(ValueError):
        remaining_allowance({})


def test_unclean_restart_charges_downtime_too():
    own = {'spent_seconds': 100, 'spent_at_start': 50, 'active': True, 'wall_started': 1000}
    assert remaining_allowance({'spent_seconds': 23 * 3600}, own, now=1200) == 3600 - 250
    assert remaining_allowance({'spent_seconds': 23 * 3600}, own, now=5000) == 0


def test_conditional_search_space_is_valid():
    import optuna

    for learned in (False, True):
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=51))
        for _ in range(20):
            trial = study.ask()
            hp = settings(sample(trial, learned))
            assert learned or hp['features'] != 'learned'
            study.tell(trial, 1)
