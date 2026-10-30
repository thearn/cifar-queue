"""The footprint evaluator: its counts match the networks', a fresh-process measurement returns the table's keys,
and the classes reject a larger model and a model that stores its training images (the negative controls)."""
import json

import numpy as np

from cifarq import airbench, footprint

SMALL = {'widths': (16, 32, 32), 'batch_size': 64, 'epochs': 1}


def test_static_counts(built, tmp_path):
    r = airbench.train(**SMALL, tta_level=2, save=str(tmp_path / 'm'))
    s = footprint.static(r['saved'])
    assert s['params_stored'] == r['params'] and s['params_trainable'] == r['params_trainable']
    assert s['views'] == 6 and s['macs_per_image'] == 6 * s['macs_per_view'] > 0
    assert s['bytes'] >= 4 * s['params_stored']


def test_measure_in_a_fresh_process(built, tmp_path):
    r = airbench.train(**SMALL, save=str(tmp_path / 'm'))
    m = footprint.measure(r['saved'], 'cpu', threads=1, batch=4, warmup=1, timed=3)
    for k in ('p50_ms', 'p95_ms', 'images_per_s', 'peak_rss_mb', 'rss_mb'):
        assert k in m and m[k] >= 0


def test_classes_reject_larger_and_leaky_models():
    ref = {'params_stored': 1000, 'bytes': 4100, 'runs': {'cpu1_b1': {'rss_mb': 100.0}}}
    row = lambda p, b, rss: {'params_stored': p, 'bytes': b, 'runs': {'cpu1_b1': {'rss_mb': rss}}}
    assert footprint.classify(row(900, 3700, 100.0), ref) == 'S'
    assert footprint.classify(row(900, 3700, 150.0), ref) == 'P'           # more memory: parameter-matched only
    assert footprint.classify(row(2000, 8100, 90.0), ref) == 'out'         # more parameters
    # a kernel method keeping its 45000 training images: few parameters, 138 MB stored
    assert footprint.classify(row(500, 45000 * 3072 + 4100, 90.0), ref) == 'out'
    assert footprint.classify(row(1500, 6100, 90.0), ref, param_cap=2000) == 'S'


def test_predictions_are_probabilities_for_unseen_images(built, tmp_path):
    r = airbench.train(**SMALL, save=str(tmp_path / 'm'))
    images = np.random.default_rng(1).integers(0, 256, size=(8, 32, 32, 3), dtype=np.uint8)
    probs = footprint.predictor(r['saved']).predict(images)
    assert probs.shape == (8, 10) and np.allclose(probs.sum(1), 1, atol=1e-5)
    assert json.loads((tmp_path / 'm' / 'config.json').read_text())['family'] == 'airbench'
