"""A trained network saved as a ``hub.ResNet9`` loads back with the same outputs (no GPU): every head, with and without
the whitening layer, and ``predict``'s normalisation and flip averaging match the training code's."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from cifarq import train  # noqa: E402
from cifarq.hub import ResNet9, save  # noqa: E402


@pytest.mark.parametrize('head,whiten', [('linear', False), ('cosine', True), ('bn', False)])
def test_saved_network_loads_with_the_same_outputs(tmp_path, head, whiten):
    torch.manual_seed(0)
    h = {**train.DEFAULTS, 'width': 16, 'head': head, 'logit_scale': 0.125, 'whiten': whiten, 'tta_flip': True,
         'activation': 'celu'}
    images = np.random.default_rng(0).integers(0, 256, (8, 32, 32, 3), dtype=np.uint8)
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255
    whitening = train.whitening_filters(x, h['whiten_eps']) if whiten else None
    net = train.model(h['width'], h['dropout'], h['activation'], celu_alpha=h['celu_alpha'], head=head,
                      logit_scale=h['logit_scale'], whitening=whitening).to(memory_format=torch.channels_last)
    net.train()
    for _ in range(3):                                       # batch-norm statistics away from their initial values
        net(x)
    net.eval()

    loaded = ResNet9.from_pretrained(save(net, h, tmp_path / 'model'))
    assert loaded.tta_flip and loaded.classes[0] == 'airplane'
    xn = (x - torch.tensor(train.MEAN).view(1, 3, 1, 1)) / torch.tensor(train.STD).view(1, 3, 1, 1)
    with torch.no_grad():
        expected = (net(xn) + net(xn.flip(3))).softmax(1).numpy()
    assert np.allclose(loaded.predict(images), expected, atol=1e-5)
    assert (tmp_path / 'model' / 'hyperparameters.json').exists()
