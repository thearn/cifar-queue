"""``train`` keeps the job contract on a CPU: its result's keys, early stopping through ``report``, unknown
hyperparameters refused, and a saved network that loads back and predicts what training measured."""
import numpy as np
import pytest

from cifarq import evaluate, train
from cifarq.hub import ResNet9
from conftest import SMALL


def test_result_contract(built):
    r = train.train(epochs=2, **SMALL)
    for k in ('val_acc', 'test_acc', 'train_seconds', 'images_per_s', 'params', 'device', 'history', evaluate.KEY):
        assert k in r
    assert r['device'] == 'cpu' and len(r['history']) == 2 and 'device' not in r['hyperparameters']


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
    assert abs(acc - r['test_acc']) <= 3 / len(probs)          # (training measured under bf16 autocast, predict in fp32)
