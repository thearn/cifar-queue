import json

import numpy as np
import pytest

from cifarq.image_boost import artifact_files, engineered, settings


def test_engineered_features_preserve_color_and_structure():
    images = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    images[1, :, :, 1] = 255
    images[2, :, 16:, :] = 255
    features = engineered(images, 'hog_color')
    assert features.shape[0] == 3 and features.dtype == np.float32
    assert np.isfinite(features).all()
    assert not np.array_equal(features[0], features[1])
    assert not np.array_equal(features[1], features[2])
    assert np.all(engineered(images[:1], 'pooled_rgb') == 0)
    with pytest.raises(ValueError):
        engineered(images.astype(float), 'hog_color')


def test_settings_and_complete_artifact_accounting(tmp_path):
    hp = settings({'features': 'learned'})
    assert {p.name for p in artifact_files(tmp_path, hp)} == {'config.json', 'projection.npz', 'trees.ubj', 'encoder.pt', 'encoder.json'}
    with pytest.raises(ValueError):
        settings({'lr': float('nan')})
    with pytest.raises(ValueError):
        settings({'pca_components': 1024})


def test_pca_is_fit_only_on_training_rows():
    pytest.importorskip('sklearn')
    from cifarq.image_boost import fit_projection, project

    train = np.array([[0, 1, 2], [1, 3, 5], [2, 5, 8], [3, 7, 11]], dtype=np.float32)
    transform = fit_projection(train, 1, 40)
    np.testing.assert_allclose(transform['mean'], train.mean(axis=0))
    before = transform['mean'].copy()
    projected = project(np.full((2, 3), 1000, dtype=np.float32), transform)
    assert projected.shape == (2, 1)
    np.testing.assert_array_equal(transform['mean'], before)


def test_exported_encoder_preserves_backbone_without_classifier(tmp_path):
    torch = pytest.importorskip('torch')
    from cifarq.image_boost import export_encoder
    from cifarq.resnet18 import build
    from cifarq.resnet18 import settings as resnet_settings

    torch.set_num_threads(2)
    source, destination = tmp_path / 'source', tmp_path / 'encoder'
    source.mkdir()
    destination.mkdir()
    hp = resnet_settings({'initialization': 'scratch'})
    model = build(hp, pretrained=False).eval()
    torch.save(model.state_dict(), source / 'weights.pt')
    (source / 'config.json').write_text(json.dumps(hp))
    export_encoder(source, destination)
    weights = torch.load(destination / 'encoder.pt', map_location='cpu', weights_only=True)
    assert not any(key.startswith('fc.') for key in weights)
    restored = build(hp, pretrained=False).eval()
    model.fc = restored.fc = torch.nn.Identity()
    restored.load_state_dict(weights)
    with torch.inference_mode():
        x = torch.rand(2, 3, 32, 32)
        torch.testing.assert_close(model(x), restored(x))
    assert (destination / 'encoder.pt').stat().st_size > 40_000_000


@pytest.mark.parametrize('backend', ['xgboost', 'lightgbm'])
def test_saved_pipeline_predicts_from_images_without_feature_cache(tmp_path, backend, monkeypatch):
    pytest.importorskip(backend)
    from cifarq import image_boost

    rng = np.random.default_rng(42)
    images = rng.integers(0, 256, (40, 32, 32, 3), dtype=np.uint8)
    labels = np.tile(np.arange(10), 4)
    arrays = {'x_train': images, 'y_train': labels, 'x_val': np.tile(images[:10], (26, 1, 1, 1)), 'y_val': np.tile(labels[:10], 26)}
    monkeypatch.setattr(image_boost, 'load_development', lambda _: arrays)
    result = image_boost.train(
        'synthetic-only',
        str(tmp_path),
        {'accuracy': 0.95, 'latency': 10, 'size': 1000000},
        {'backend': backend, 'features': 'pooled_rgb', 'rounds': 2, 'depth': 2, 'threads': 1, 'train_device': 'cpu'},
    )
    assert result['size'] == sum(p.stat().st_size for p in artifact_files(tmp_path, result['config']))
    predictor = image_boost.Predictor(str(tmp_path))
    calls = []
    original = predictor.extract

    def extract(x):
        calls.append(len(x))
        return original(x)

    predictor.extract = extract
    for _ in range(2):
        assert predictor.predict(images[:3]).shape == (3,)
    assert calls == [3, 3]
    predictor.extract = lambda x: np.full((len(x), 192), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match='nonfinite'):
        predictor.predict(images[:3])
    assert set(json.loads((tmp_path / 'assessment.json').read_text())['artifacts']) == {p.name for p in artifact_files(tmp_path, result['config'])}
