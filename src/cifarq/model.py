"""The network: a ResNet-9-style CNN (the fast CIFAR-10 architecture of the DAWNBench entries) with a width multiplier,
dropout before the classifier, a choice of activation, of classifier head and of a scale on its logits, and optionally
a fixed input whitening layer (``whitening_filters``). ``MEAN`` and ``STD`` normalise the images (per channel, 0-1)."""
from __future__ import annotations

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def model(width: int = 32, dropout: float = 0.0, activation: str = 'relu', classes: int = 10, celu_alpha: float = 0.075,
          head: str = 'linear', logit_scale: float = 1.0, whitening=None):
    """The network. ``head``: the classifier and how its output is normalised: ``linear`` (a linear layer, its logits
    times ``logit_scale``: Page's network is 0.125), ``cosine`` (the cosine of features and class weights, both L2
    normalised, times ``logit_scale``: NormFace, cosine classifiers) or ``bn`` (a linear layer, then batch norm on the
    logits without an affine part, times ``logit_scale``). ``whitening``: a fixed (27, 3, 3, 3) filter bank
    (``whitening_filters``) put first, the prep convolution then taking its 27 channels."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    act = {'relu': lambda: nn.ReLU(inplace=True), 'gelu': nn.GELU, 'silu': lambda: nn.SiLU(inplace=True),
           'celu': lambda: nn.CELU(alpha=celu_alpha)}[activation]

    def conv(cin, cout, pool=False):
        layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), act()]
        return nn.Sequential(*layers, *([nn.MaxPool2d(2)] if pool else []))

    class Residual(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.f = nn.Sequential(conv(c, c), conv(c, c))

        def forward(self, x):
            return x + self.f(x)

    class Scale(nn.Module):
        def __init__(self, s):
            super().__init__()
            self.s = s

        def forward(self, x):
            return x * self.s

    class Cosine(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(cout, cin) / cin ** 0.5)

        def forward(self, x):
            return F.normalize(x, dim=1) @ F.normalize(self.weight, dim=1).T

    w = width
    first = []
    if whitening is not None:
        white = nn.Conv2d(3, whitening.shape[0], 3, padding=1, bias=False)
        white.weight.data.copy_(whitening)
        white.weight.requires_grad_(False)
        first = [white]
    cin = whitening.shape[0] if whitening is not None else 3
    classifier = {'linear': lambda: [nn.Linear(8 * w, classes)], 'cosine': lambda: [Cosine(8 * w, classes)],
                  'bn': lambda: [nn.Linear(8 * w, classes, bias=False), nn.BatchNorm1d(classes, affine=False)]}[head]()
    return nn.Sequential(*first, conv(cin, w), conv(w, 2 * w, pool=True), Residual(2 * w), conv(2 * w, 4 * w, pool=True),
                         conv(4 * w, 8 * w, pool=True), Residual(8 * w), nn.AdaptiveMaxPool2d(1), nn.Flatten(),
                         nn.Dropout(dropout), *classifier, *([Scale(logit_scale)] if logit_scale != 1.0 else []))


def whitening_filters(x, eps: float):
    """The whitening filter bank of normalised images ``x`` (N, 3, 32, 32): the eigenvectors of their 3x3 patches'
    covariance, each divided by the square root of its eigenvalue (plus ``eps``), as (27, 3, 3, 3) filters."""
    import torch
    import torch.nn.functional as F
    patches = F.unfold(x.float(), 3).transpose(1, 2).reshape(-1, 27)          # every 3x3x3 patch, unpadded
    patches = patches - patches.mean(0)
    cov = patches.T @ patches / len(patches)
    vals, vecs = torch.linalg.eigh(cov.double())
    return (vecs.T / torch.sqrt(vals + eps).unsqueeze(1)).float().reshape(27, 3, 3, 3)
