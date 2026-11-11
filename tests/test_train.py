"""``train`` keeps the job contract on whatever device it runs on: its result's keys, early stopping through
``report``, unknown hyperparameters refused, and a saved network that loads back and predicts what training
measured."""

import numpy as np
import pytest
from conftest import SMALL

from cifarq import evaluate, jx, train
from cifarq.hub import ResNet9


def test_result_contract(built):
    r = train.train(epochs=2, **SMALL)
    for k in ('val_acc', 'test_acc', 'train_seconds', 'images_per_s', 'params', 'device', 'history', evaluate.KEY):
        assert k in r
    assert r['device'] == jx.device_name() and len(r['history']) == 2 and 'device' not in r['hyperparameters']


def test_report_stops_a_trial(built):
    steps = []
    r = train.train(epochs=4, report=lambda step, value: steps.append(step) or step >= 50, **SMALL)
    assert r['pruned'] and r['progress'] == 50 and steps == [25, 50] and 'test_acc' not in r


def test_unknown_hyperparameters_are_refused(built):
    with pytest.raises(TypeError, match='unknown hyperparameters'):
        train.train(epochs=1, learning_rate=0.1)


@pytest.mark.parametrize('extra', [{}, {'ema_decay': 0.9, 'tta_flip': True, 'whiten': True}])
def test_saved_network_predicts_what_training_measured(built, tmp_path, monkeypatch, extra):
    monkeypatch.chdir(tmp_path)
    r = train.train(epochs=2, save='model', **SMALL, **extra)
    net = ResNet9.from_pretrained(tmp_path / r['saved'])
    probs = net.predict(built['x_test'])
    assert probs.shape == (200, 10) and np.allclose(probs.sum(1), 1, atol=1e-5)
    acc = (probs.argmax(1) == built['y_test']).mean()
    assert abs(acc - r['test_acc']) <= 3 / len(probs)  # (training measured under bf16 autocast, predict in fp32)


def test_seeds_in_one_program_are_separate_trainings(built):
    """``train_seeds`` vmaps one configuration over seeds (the schedule-free average, whitening, cutmix and cutout
    included): one seed is exactly ``train``'s run, a repeated seed gives the same run twice, and different seeds
    differ. (Two seeds against one are equal only to rounding: the batched convolutions sum in another order. On a
    GPU that is true of one seed against ``train`` as well -- vmapping changes which kernels XLA picks -- so there
    the two agree to rounding rather than bitwise.)"""
    h = {
        **SMALL,
        'epochs': 1,
        'optimizer': 'adamw_sf',
        'lr': 0.006,
        'whiten': True,
        'tta_flip': True,
        'mix': 'cutmix',
        'mix_prob': 1.0,
        'cutout_size': 8,
    }
    loss = lambda r: r['history'][0]['loss']
    vmapped, plain = loss(train.train_seeds([1], **h)[0]), loss(train.train(seed=1, **h))
    assert vmapped == plain if jx.device_name() == 'cpu' else vmapped == pytest.approx(plain, rel=5e-3)
    a, b, c = train.train_seeds([1, 1, 0], together=True, **h)
    assert loss(a) == loss(b) and a['test_acc'] == b['test_acc'] and loss(a) != loss(c)
    assert [r['hyperparameters']['seed'] for r in (a, b, c)] == [1, 1, 0]


@pytest.mark.parametrize('optimizer', ['sgd', 'sgd_sf'])
def test_the_training_set_is_an_argument_not_a_constant(built, optimizer):
    """Closed over, the training set was baked into the compiled epoch and weights programs (138 MB each on CIFAR)."""
    run = train.Run(train._check({**SMALL, 'epochs': 1, 'optimizer': optimizer}), [0], device='cpu')
    data = run.x_train.nbytes
    epoch = run._epoch.lower(run.st, run.hp, run.x_train, run.y_train).as_text()
    weights = run._weights.lower(run.st, run.x_train).as_text()
    assert len(epoch) < data and len(weights) < data


def test_maximal_augmentation_has_enough_rng_keys():
    import jax
    import jax.numpy as jnp

    h = train._check(
        {
            'aug_brightness': 0.1,
            'aug_contrast': 0.1,
            'aug_saturation': 0.1,
            'aug_translate': 2.0,
            'aug_rotate': 5.0,
            'aug_scale': 0.1,
            'aug_flip': 0.5,
            'cutout_size': 4,
            'cutout_prob': 0.5,
        }
    )
    x = jax.random.uniform(jax.random.key(0), (2, 8, 8, 3))
    out = train.augment(jax.random.key(1), x, h, jx.dynamic(h, train.DYNAMIC))
    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))
