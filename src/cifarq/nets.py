"""The other network families of the autoresearch session's Goal A screen, built to train under ``airbench.train``
(``arch`` and ``arch_kw``): each builder returns an ``nn.Sequential``-like module taking normalised (N, 3, 32, 32)
images to logits, with plain ``nn.BatchNorm2d`` layers (``airbench.train`` freezes their scale when its
``bn_scale`` is off, as airbench does) and an attribute ``whiten``: a frozen first convolution to set from the
training patches, or None.

- ``convmixer``: Trockman & Kolter, "Patches are all you need?" (2022): a patch embedding of ``dim`` channels, then
  ``depth`` blocks of a residual depthwise convolution (``kernel``) and a pointwise one, GELU and batch norm after
  each; global average pool, linear.
- ``resnet9``: this project's ResNet-9 (``cifarq.model.model``) at ``width``, linear head, logits times ``logit_scale``.
- ``pyramidnet``: Han, Kim & Kim (2017), additive widening by ``alpha`` over ``depth`` (basic blocks: depth = 6n + 2;
  bottleneck: 9n + 2), zero-padded shortcuts.
- ``densenet``: Huang et al. (2017), DenseNet-BC: three dense blocks of bottleneck layers at ``growth``, compression 0.5.
- ``shakeshake``: Gastaldi (2017), 26 2x``width``d: three stages of 4 blocks, two residual branches mixed by a random
  convex weight per image in training (forward) and an independent one in the backward pass, 0.5 each at test.
- ``wrn``: Zagoruyko & Komodakis (2016), WRN-``depth``-``k``, pre-activation basic blocks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Net(nn.Module):
    def __init__(self, body: nn.Module, whiten: nn.Conv2d | None = None):
        super().__init__()
        self.body, self.whiten = body, whiten

    def forward(self, x):
        return self.body(x)


def convmixer(dim: int = 256, depth: int = 8, kernel: int = 9, patch: int = 1, classes: int = 10, **_):
    class Residual(nn.Module):
        def __init__(self, f):
            super().__init__()
            self.f = f

        def forward(self, x):
            return x + self.f(x)

    body = nn.Sequential(
        nn.Conv2d(3, dim, patch, stride=patch), nn.GELU(), nn.BatchNorm2d(dim),
        *[nn.Sequential(Residual(nn.Sequential(nn.Conv2d(dim, dim, kernel, groups=dim, padding='same'), nn.GELU(),
                                               nn.BatchNorm2d(dim))),
                        nn.Conv2d(dim, dim, 1), nn.GELU(), nn.BatchNorm2d(dim)) for _ in range(depth)],
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dim, classes))
    return _Net(body)


def resnet9(width: int = 32, logit_scale: float = 0.125, activation: str = 'relu', classes: int = 10, **_):
    from .model import model
    return _Net(model(width, 0.0, activation, classes=classes, logit_scale=logit_scale))


def pyramidnet(depth: int = 110, alpha: float = 48, bottleneck: bool = False, classes: int = 10, **_):
    n = (depth - 2) // (9 if bottleneck else 6)
    expansion = 4 if bottleneck else 1
    add = alpha / (3 * n)

    class Block(nn.Module):
        def __init__(self, cin, c, stride):
            super().__init__()
            cout = c * expansion
            if bottleneck:
                self.f = nn.Sequential(nn.BatchNorm2d(cin), nn.Conv2d(cin, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                                       nn.Conv2d(c, c, 3, stride, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                                       nn.Conv2d(c, cout, 1, bias=False), nn.BatchNorm2d(cout))
            else:
                self.f = nn.Sequential(nn.BatchNorm2d(cin), nn.Conv2d(cin, c, 3, stride, 1, bias=False), nn.BatchNorm2d(c),
                                       nn.ReLU(inplace=True), nn.Conv2d(c, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
            self.stride, self.pad = stride, cout - cin

        def forward(self, x):
            s = F.avg_pool2d(x, 2) if self.stride == 2 else x
            if self.pad:
                s = F.pad(s, (0, 0, 0, 0, 0, self.pad))
            return self.f(x) + s

    layers, c, cin = [nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16)], 16.0, 16
    for stage in range(3):
        for i in range(n):
            c += add
            layers.append(Block(cin, int(round(c)), 2 if stage > 0 and i == 0 else 1))
            cin = int(round(c)) * expansion
    layers += [nn.BatchNorm2d(cin), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, classes)]
    return _Net(nn.Sequential(*layers))


def densenet(depth: int = 100, growth: int = 12, classes: int = 10, **_):
    n = (depth - 4) // 6

    class Layer(nn.Module):
        def __init__(self, cin):
            super().__init__()
            self.f = nn.Sequential(nn.BatchNorm2d(cin), nn.ReLU(inplace=True), nn.Conv2d(cin, 4 * growth, 1, bias=False),
                                   nn.BatchNorm2d(4 * growth), nn.ReLU(inplace=True), nn.Conv2d(4 * growth, growth, 3, 1, 1, bias=False))

        def forward(self, x):
            return torch.cat([x, self.f(x)], 1)

    layers, c = [nn.Conv2d(3, 2 * growth, 3, 1, 1, bias=False)], 2 * growth
    for stage in range(3):
        for _ in range(n):
            layers.append(Layer(c))
            c += growth
        if stage < 2:
            layers += [nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.Conv2d(c, c // 2, 1, bias=False), nn.AvgPool2d(2)]
            c //= 2
    layers += [nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, classes)]
    return _Net(nn.Sequential(*layers))


class _Shake(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, training):
        if not training:
            return 0.5 * (a + b)
        w = torch.rand(a.size(0), 1, 1, 1, device=a.device, dtype=a.dtype)
        return w * a + (1 - w) * b

    @staticmethod
    def backward(ctx, g):
        w = torch.rand(g.size(0), 1, 1, 1, device=g.device, dtype=g.dtype)
        return w * g, (1 - w) * g, None


def shakeshake(depth: int = 26, width: int = 32, classes: int = 10, **_):
    n = (depth - 2) // 6

    def branch(cin, cout, stride):
        return nn.Sequential(nn.ReLU(), nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                             nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))

    class Shortcut(nn.Module):                              # (the paper's: two 1x1 paths, the second shifted a pixel)
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.a, self.b = nn.Conv2d(cin, cout // 2, 1, bias=False), nn.Conv2d(cin, cout // 2, 1, bias=False)
            self.bn, self.stride = nn.BatchNorm2d(cout), stride

        def forward(self, x):
            x = F.relu(x)
            shifted = F.pad(x[:, :, 1:, 1:], (0, 1, 0, 1))
            return self.bn(torch.cat([self.a(F.avg_pool2d(x, 1, self.stride)), self.b(F.avg_pool2d(shifted, 1, self.stride))], 1))

    class Block(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.b1, self.b2 = branch(cin, cout, stride), branch(cin, cout, stride)
            self.short = Shortcut(cin, cout, stride) if stride != 1 or cin != cout else nn.Identity()

        def forward(self, x):
            return self.short(x) + _Shake.apply(self.b1(x), self.b2(x), self.training)

    layers, cin = [nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16)], 16
    for stage, c in enumerate((width, 2 * width, 4 * width)):
        for i in range(n):
            layers.append(Block(cin, c, 2 if stage > 0 and i == 0 else 1))
            cin = c
    layers += [nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, classes)]
    return _Net(nn.Sequential(*layers))


def wrn(depth: int = 16, k: int = 4, classes: int = 10, **_):
    n = (depth - 4) // 6

    class Block(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.bn1, self.conv1 = nn.BatchNorm2d(cin), nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
            self.bn2, self.conv2 = nn.BatchNorm2d(cout), nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.short = nn.Conv2d(cin, cout, 1, stride, bias=False) if stride != 1 or cin != cout else None

        def forward(self, x):
            o = F.relu(self.bn1(x))
            y = self.conv2(F.relu(self.bn2(self.conv1(o))))
            return y + (self.short(o) if self.short is not None else x)

    layers, cin = [nn.Conv2d(3, 16, 3, 1, 1, bias=False)], 16
    for stage, c in enumerate((16 * k, 32 * k, 64 * k)):
        for i in range(n):
            layers.append(Block(cin, c, 2 if stage > 0 and i == 0 else 1))
            cin = c
    layers += [nn.BatchNorm2d(cin), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, classes)]
    return _Net(nn.Sequential(*layers))


BUILDERS = {'convmixer': convmixer, 'resnet9': resnet9, 'pyramidnet': pyramidnet, 'densenet': densenet,
            'shakeshake': shakeshake, 'wrn': wrn}


def build(arch: str, **kw):
    return BUILDERS[arch](**kw)
