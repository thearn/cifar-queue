"""The data stages on the synthetic source: decoding keeps every image, the stages are no-ops when nothing changed,
the split is the fixed permutation train() always drew, and the checks catch bad data."""
import numpy as np
import pytest

from cifarq import paths
from cifarq.data import clean, dataset, ingest, validate


def test_decoding_keeps_every_image(built):
    m = paths.read_manifest('clean')
    assert m['kept'] == m['rows'] == {'train': 600, 'test': 200} and not any(m['dropped'].values())
    assert built['x_train'].dtype == np.uint8 and built['x_train'].shape[1:] == (32, 32, 3)


def test_stages_are_no_ops_when_nothing_changed(built):
    before = {n: paths.read_manifest(n) for n in ('raw', 'clean', 'dataset')}
    stamp = dataset.path().stat().st_mtime_ns
    assert ingest.fetch() == before['raw'] and clean.run() == before['clean'] and dataset.build() == before['dataset']
    assert dataset.path().stat().st_mtime_ns == stamp


def test_split_is_the_fixed_permutation(built):
    """Validation: the first `validation` of default_rng(seed).permutation over the training file, training the rest,
    both in that order (as train() drew them before the dataset stage: every study so far)."""
    split = paths.config('data')['split']
    raw = np.load(clean.CLEAN / 'train.npz')
    order = np.random.default_rng(split['seed']).permutation(len(raw['y']))
    v, t = order[:split['validation']], order[split['validation']:]
    assert np.array_equal(built['x_val'], raw['x'][v]) and np.array_equal(built['y_train'], raw['y'][t])
    assert np.array_equal(built['x_train'], raw['x'][t]) and built['ids_val'][0] == f'train:{v[0]}'
    assert len(built['y_test']) == 200 and built['ids_test'][0] == 'test:0'


def test_checks_catch_bad_data(built):
    with pytest.raises(validate.DataError, match='more than one split'):
        validate.dataset({**built, 'ids_test': built['ids_train'][:len(built['ids_test'])]})
    with pytest.raises(validate.DataError, match='class'):
        validate.dataset({**built, 'y_val': np.zeros_like(built['y_val'])})
    with pytest.raises(validate.DataError, match='dropped'):
        validate.cleaned([{'split': 'train', 'rows': 100, 'kept': 99, 'dropped': {'wrong_size': 1}},
                          {'split': 'test', 'rows': 10, 'kept': 10, 'dropped': {'wrong_size': 0}}],
                         {'min_rows': 10, 'max_drop_fraction': 0.0})
    with pytest.raises(validate.DataError, match='splits'):
        validate.cleaned([{'split': 'train', 'rows': 1, 'kept': 1, 'dropped': {}}], {'min_rows': 0, 'max_drop_fraction': 0})
