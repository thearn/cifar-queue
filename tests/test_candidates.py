"""Candidates keep each stage's result in git, link a champion to the trial it retrained, and say when one is stale."""

from cifarq import candidates as C


def test_best_per_group_verification_and_staleness(tmp_path):
    f = tmp_path / 'candidates.jsonl'
    code = {'tree': 't1', 'dirty': False}
    search = C.from_record(
        'tune',
        {'provenance': {'dataset': 'd1', 'commit': 'abc'}, 'study': 's', 'best': {'trial': 3, 'value': 0.94, 'params': {'lr': 0.1}}},
        'results/tune/a.json',
    )
    assert search['id'] == 'tune-a' and search['metric'] == 'val_acc'
    assert C.record({**search, 'code': code}, f)
    old = C.from_record('on-vm', {'best_value': 0.93, 'best_params': {'lr': 0.2}, 'best_test_acc': 0.92}, 'results/on-vm/b.json')
    assert not C.record({**old, 'code': code}, f)  # worse in the same search group
    assert C.best('search', 'val_acc', path=f)['id'] == 'tune-a'
    C.verify('tune-a', 'champion', True, 'retrained', path=f)
    assert C.best('search', 'val_acc', path=f)['verified']['champion']['passed']
    e = C.best('search', 'val_acc', path=f)
    assert not C.stale(e, 'd1', code) and C.stale(e, 'd2', code) and C.stale(e, 'd1', {'tree': 't2', 'dirty': False})


def test_a_champion_names_the_candidate_it_retrained():
    rec = {
        'provenance': {'dataset': 'd'},
        'candidate': 'tune-a',
        'runs': [{'val_acc': 0.95}, {'val_acc': 0.94}],
        'chosen': {'val_acc': 0.945},
        'params': {'lr': 1},
    }
    e = C.from_record('champion', rec, 'results/champion/x.json')
    assert e['from'] == 'tune-a' and e['value'] == 0.945 and e['seeds'] == 2


def test_a_search_starts_from_the_best_or_a_named_candidate(tmp_path):
    import pytest

    f = tmp_path / 'candidates.jsonl'
    for name, v in (('a', 0.93), ('b', 0.95)):
        C.record({**C.from_record('on-vm', {'best_value': v, 'best_params': {'lr': v}}, f'results/on-vm/{name}.json'), 'code': {'tree': 't'}}, f)
    assert C.start_point('best', f)['params'] == {'lr': 0.95}
    assert C.start_point('on-vm-a', f)['value'] == 0.93
    with pytest.raises(KeyError):
        C.start_point('nope', f)
