"""Complete-pipeline scoring boundaries, with no accelerator or model download."""

import json

import numpy as np
import pytest

from cifarq.pipeline_score import classes, evaluate, load_development, objective
from cifarq.resnet18 import settings
from cifarq.resnet18_campaign import sample


def test_pipeline_accuracy_and_primary_tradeoff():
    images = np.zeros((8, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(8) % 2
    predictions = iter([labels[:4], labels[4:]])
    result = evaluate(lambda x: next(predictions, np.zeros(len(x), dtype=int)), images, labels, batch=4, warmup=0, repeats=2)
    assert result['accuracy'] == 1
    assert len(result['latency_ms_samples']) == 2
    reference = {'accuracy': 0.95, 'latency': 10, 'size': 1000}
    assert objective(reference, reference) == pytest.approx(100)
    assert objective({**reference, 'size': 2000}, reference) < 100
    assert objective({**reference, 'accuracy': 0.97, 'size': 2000}, reference) > 100


@pytest.mark.parametrize('bad', [np.array([np.nan]), np.array([10]), np.array([[0, 1]])])
def test_rejects_broken_predictors(bad):
    with pytest.raises(ValueError):
        classes(bad, 1)


def test_data_hash_and_sealed_test(tmp_path):
    import hashlib

    path = tmp_path / 'development.npz'
    np.savez(path, **{k: np.zeros(2) for k in ('x_train', 'y_train', 'x_val', 'y_val')}, x_test=np.array([object()]))
    with pytest.raises(ValueError, match='hash'):
        load_development(str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    arrays = load_development(str(path), expected_hash=digest)
    assert set(arrays) == {'x_train', 'y_train', 'x_val', 'y_val'}


def test_resnet_settings_reject_silent_options():
    with pytest.raises(ValueError, match='unknown'):
        settings({'tta': 6})
    with pytest.raises(ValueError):
        settings({'freeze_epochs': 30})
    with pytest.raises(ValueError):
        settings({'lr': float('nan')})


def test_search_space_produces_valid_configs():
    import optuna

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=8))
    for _ in range(20):
        trial = study.ask()
        config = settings(sample(trial))
        assert config['initialization'] in ('imagenet', 'scratch')
        assert config['resolution'] in (32, 48, 64, 96)
        assert json.loads(json.dumps(config)) == config
        study.tell(trial, 1.0)


def test_campaign_order_and_optuna_objective(tmp_path, monkeypatch):
    from cifarq import resnet18_campaign as campaign

    reference = {'accuracy': 0.95, 'latency': 10, 'size': 1000}
    out = tmp_path / 'results/autoresearch/resnet18-score'
    out.mkdir(parents=True)
    (out / 'reference.json').write_text(json.dumps(reference))
    calls = []

    def fake_child(python, payload, output, seconds):
        config = settings(payload['config'])
        calls.append(output.stem)
        measured = {'accuracy': 0.96, 'latency': 8.0, 'size': 2000, 'latency_ms_samples': [8.0]}
        result = {**measured, 'score': objective(measured, reference), 'config': config, 'saved': str(output)}
        output.write_text(json.dumps(result))
        assert seconds > payload['seconds']
        return result

    monkeypatch.setattr(campaign, 'child', fake_child)
    result = campaign.run(str(tmp_path), 'unused.npz', 'unused-python', hours=24, trials=3)
    assert calls[:3] == ['correctness-smoke', 'imagenet-baseline', 'scratch-baseline']
    assert calls[3].startswith('trial-')
    assert calls[4:] == ['confirmation-71', 'confirmation-72', 'confirmation-73']
    assert result['completed_trials'] == 3
    expected = objective({'accuracy': 0.96, 'latency': 8, 'size': 2000}, reference)
    assert result['best_search']['score'] == pytest.approx(expected)
    assert result['confirmed_recipe_score'] == pytest.approx(expected)
    import optuna

    study = optuna.load_study(study_name='resnet18-primary-score-v1', storage=f'sqlite:///{out / "optuna.db"}')
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE
    assert all(t.value == pytest.approx(expected) for t in study.trials)
    assert json.loads((out / 'budget.json').read_text())['active'] is False
