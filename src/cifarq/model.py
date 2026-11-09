"""The network: a ResNet-9-style CNN (the fast CIFAR-10 architecture of the DAWNBench entries) with a width multiplier,
dropout before the classifier, a choice of activation, of classifier head and of a scale on its logits, and optionally
a fixed input whitening layer (``whitening_filters``), as a Flax NNX module on NHWC images. ``MEAN`` and ``STD``
normalise the images (per channel, 0-1)."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from . import jx

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


class ConvBN(nnx.Module):
    def __init__(self, cin: int, cout: int, act: Callable[[jax.Array], jax.Array], pool: bool = False, *, rngs: nnx.Rngs) -> None:
        self.conv, self.bn, self.act, self.pool = jx.conv(cin, cout, 3, padding=1, rngs=rngs), jx.batchnorm(cout, rngs=rngs), act, pool

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.act(self.bn(self.conv(x)))
        return jx.max_pool(x) if self.pool else x


class Residual(nnx.Module):
    def __init__(self, c: int, act: Callable[[jax.Array], jax.Array], *, rngs: nnx.Rngs) -> None:
        self.a, self.b = ConvBN(c, c, act, rngs=rngs), ConvBN(c, c, act, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + self.b(self.a(x))


class Cosine(nnx.Module):
    """The cosine of L2-normalised features and class weights (NormFace, cosine classifiers)."""

    def __init__(self, cin: int, cout: int, *, rngs: nnx.Rngs) -> None:
        self.weight = nnx.Param(nnx.initializers.normal(1 / cin**0.5)(rngs.params(), (cin, cout)))

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.astype(jnp.float32)
        w = self.weight[...]
        return (x / jnp.maximum(jnp.linalg.norm(x, axis=1, keepdims=True), 1e-12)) @ (
            w / jnp.maximum(jnp.linalg.norm(w, axis=0, keepdims=True), 1e-12)
        )


class BNHead(nnx.Module):
    """A linear layer without bias, then batch norm on the logits without an affine part."""

    def __init__(self, cin: int, cout: int, *, rngs: nnx.Rngs) -> None:
        self.fc, self.bn = jx.linear(cin, cout, bias=False, rngs=rngs), jx.batchnorm(cout, affine=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.bn(self.fc(x))


class ResNet9(nnx.Module):
    def __init__(
        self,
        width: int = 32,
        dropout: float = 0.0,
        activation: str = 'relu',
        classes: int = 10,
        celu_alpha: float = 0.075,
        head: str = 'linear',
        logit_scale: float = 1.0,
        whitening: np.ndarray | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        act, w = jx.activation(activation, celu_alpha), int(width)
        self.white = jx.FixedConv(whitening, padding=1) if whitening is not None else None
        cin = whitening.shape[-1] if whitening is not None else 3
        self.prep = ConvBN(cin, w, act, rngs=rngs)
        self.layer1, self.res1 = ConvBN(w, 2 * w, act, pool=True, rngs=rngs), Residual(2 * w, act, rngs=rngs)
        self.layer2 = ConvBN(2 * w, 4 * w, act, pool=True, rngs=rngs)
        self.layer3, self.res3 = ConvBN(4 * w, 8 * w, act, pool=True, rngs=rngs), Residual(8 * w, act, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs) if dropout else None
        self.head = {
            'linear': lambda: jx.linear(8 * w, classes, rngs=rngs),
            'cosine': lambda: Cosine(8 * w, classes, rngs=rngs),
            'bn': lambda: BNHead(8 * w, classes, rngs=rngs),
        }[head]()
        self.logit_scale = float(logit_scale)

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.white is not None:
            x = self.white(x)
        x = self.res1(self.layer1(self.prep(x)))
        x = self.res3(self.layer3(self.layer2(x)))
        x = jx.global_max(x)
        if self.dropout is not None:
            x = self.dropout(x)
        z = self.head(x).astype(jnp.float32)
        return z * self.logit_scale if self.logit_scale != 1.0 else z


def model(
    width: int = 32,
    dropout: float = 0.0,
    activation: str = 'relu',
    classes: int = 10,
    celu_alpha: float = 0.075,
    head: str = 'linear',
    logit_scale: float = 1.0,
    whitening: np.ndarray | None = None,
    seed: int = 0,
) -> ResNet9:
    """The network. ``head``: the classifier and how its output is normalised: ``linear`` (a linear layer, its logits
    times ``logit_scale``: Page's network is 0.125), ``cosine`` (the cosine of features and class weights, both L2
    normalised, times ``logit_scale``: NormFace, cosine classifiers) or ``bn`` (a linear layer, then batch norm on the
    logits without an affine part, times ``logit_scale``). ``whitening``: a fixed (3, 3, 3, 27) HWIO filter bank
    (``whitening_filters``) put first, the prep convolution then taking its 27 channels. ``seed``: the
    initialisation's and dropout's random numbers."""
    return ResNet9(
        width, dropout, activation, classes, celu_alpha, head, logit_scale, whitening, rngs=nnx.Rngs(params=seed, dropout=seed + 1_000_003)
    )


def whitening_filters(x: np.ndarray, eps: float) -> np.ndarray:
    """The whitening filter bank of normalised images ``x`` (N, 32, 32, 3): the eigenvectors of their 3x3 patches'
    covariance, each divided by the square root of its eigenvalue (plus ``eps``), as a (3, 3, 3, 27) HWIO kernel."""
    return jx.whitening_kernel(x, 3, eps)
