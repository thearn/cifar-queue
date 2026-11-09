"""A trained network saved as a ``hub.ResNet9`` loads back with the same outputs (no GPU): every head, with and without
the whitening layer, and ``predict``'s normalisation and flip averaging match the training code's."""

import numpy as np
import pytest

from cifarq import evaluate, jx, train
from cifarq.hub import ResNet9, save


@pytest.mark.parametrize('head,whiten', [('linear', False), ('cosine', True), ('bn', False)])
def test_saved_network_loads_with_the_same_outputs(tmp_path, head, whiten):
    import jax.numpy as jnp

    h = {**train.DEFAULTS, 'width': 16, 'head': head, 'logit_scale': 0.125, 'whiten': whiten, 'tta_flip': True, 'activation': 'celu'}
    images = np.random.default_rng(0).integers(0, 256, (8, 32, 32, 3), dtype=np.uint8)
    xn = (images.astype(np.float32) / 255 - np.asarray(train.MEAN, np.float32)) / np.asarray(train.STD, np.float32)
    whitening = train.whitening_filters(xn, h['whiten_eps']) if whiten else None
    net = train.model(
        h['width'], h['dropout'], h['activation'], celu_alpha=h['celu_alpha'], head=head, logit_scale=h['logit_scale'], whitening=whitening
    )
    S = jx.Split(net)
    rest = S.rest
    for _ in range(3):  # batch-norm statistics away from their initial values
        _, rest = S.apply(S.params, rest, jnp.asarray(xn), train=True)

    loaded = ResNet9.from_pretrained(save(S, S.params, rest, h, tmp_path / 'model'))
    assert loaded.tta_flip and loaded.classes[0] == 'airplane'
    z = np.asarray(evaluate.logits(S, S.params, rest, images, tta_flip=True, batch=8))
    expected = np.exp(z - z.max(1, keepdims=True))
    expected /= expected.sum(1, keepdims=True)
    assert np.allclose(loaded.predict(images), expected, atol=1e-5)
    assert (tmp_path / 'model' / 'hyperparameters.json').exists()
