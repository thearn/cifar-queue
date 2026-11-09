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
    assert footprint.classify(row(900, 3700, 150.0), ref) == 'P'  # more memory: parameter-matched only
    assert footprint.classify(row(2000, 8100, 90.0), ref) == 'out'  # more parameters
    # a kernel method keeping its 45000 training images: few parameters, 138 MB stored
    assert footprint.classify(row(500, 45000 * 3072 + 4100, 90.0), ref) == 'out'
    assert footprint.classify(row(1500, 6100, 90.0), ref, param_cap=2000) == 'S'


def test_predictions_are_probabilities_for_unseen_images(built, tmp_path):
    r = airbench.train(**SMALL, save=str(tmp_path / 'm'))
    images = np.random.default_rng(1).integers(0, 256, size=(8, 32, 32, 3), dtype=np.uint8)
    probs = footprint.predictor(r['saved']).predict(images)
    assert probs.shape == (8, 10) and np.allclose(probs.sum(1), 1, atol=1e-5)
    assert json.loads((tmp_path / 'm' / 'config.json').read_text())['family'] == 'airbench'


def test_macs_count_equivariant_layers_as_the_convolution_they_run():
    import jax.numpy as jnp
    from flax import nnx

    from cifarq import methods, nets

    for lift in (True, False):
        layer = methods.FlipConv(8, 8, lift=lift)
        kh, kw, cin, cout = layer.full().shape
        plain = nets.Conv(cin, cout, 3, padding=1, bias=False, dtype=jnp.float32, rngs=nnx.Rngs(0))
        shape = (1, 16, 16, cin)
        assert footprint.count_macs(layer, shape) == footprint.count_macs(plain, shape) == 16 * 16 * cout * cin * 9


def test_macs_match_the_torch_counts():
    """The Torch counter's numbers for the session's networks (results/footprint/20261030-151849.json)."""
    ab96, _ = airbench.build(airbench.AIRBENCH96, abstract=True)
    fin, _ = airbench.build({**airbench.DEFAULTS, 'depth': 3, 'widths': (128, 112, 144), 'flip_eq': 2}, abstract=True)
    assert footprint.count_macs(ab96) == 451967264
    assert footprint.count_macs(fin) == 237405888
    assert airbench.param_counts(fin) == {'stored': 1944024, 'trainable': 1942584}
