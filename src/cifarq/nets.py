"""The other network families of the autoresearch session's Goal A screen, built to train under ``airbench.train``
(``arch`` and ``arch_kw``): each builder returns a Flax NNX module taking normalised (N, 32, 32, 3) images to logits,
with plain batch norm layers (``airbench.train`` freezes their scale when its ``bn_scale`` is off, as airbench does,
and sets their momentum to its ``bn_momentum``) and an attribute ``whiten``: a frozen first convolution to set from the
training patches, or None. Layers are initialised as PyTorch's defaults; ``dtype`` is the arithmetic's.

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

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx

from . import jx

if TYPE_CHECKING:
    from jax.typing import DTypeLike

relu = jax.nn.relu
gelu = lambda x: jax.nn.gelu(x, approximate=False)


class Conv(nnx.Module):
    """``torch.nn.Conv2d``'s layer and initialisation; ``padding`` an int (symmetric), 'SAME' or 'VALID'."""

    def __init__(
        self,
        cin: int,
        cout: int,
        k: int,
        stride: int = 1,
        padding: int | str = 0,
        bias: bool = True,
        groups: int = 1,
        *,
        dtype: DTypeLike,
        rngs: nnx.Rngs,
    ) -> None:
        self.weight = nnx.Param(jx.torch_kernel_init()(rngs.params(), (k, k, cin // groups, cout)))
        self.bias = nnx.Param(jx.torch_bias_init(cin // groups * k * k)(rngs.params(), (cout,))) if bias else None
        self.padding = [(padding, padding)] * 2 if isinstance(padding, int) else padding
        self.stride, self.groups, self.dtype = stride, groups, dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        y = jax.lax.conv_general_dilated(
            x.astype(self.dtype),
            self.weight[...].astype(self.dtype),
            (self.stride, self.stride),
            self.padding,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=self.groups,
        )
        return y + self.bias[...].astype(y.dtype) if self.bias is not None else y


def _linear(cin: int, cout: int, dtype: DTypeLike, rngs: nnx.Rngs) -> nnx.Linear:
    return nnx.Linear(cin, cout, kernel_init=jx.torch_kernel_init(), bias_init=jx.torch_bias_init(cin), dtype=dtype, rngs=rngs)


def _bn(c: int, rngs: nnx.Rngs) -> nnx.BatchNorm:
    return jx.batchnorm(c, rngs=rngs)


def _avg_pool(x: jax.Array, k: int) -> jax.Array:
    return nnx.avg_pool(x, (k, k), strides=(k, k))


class _Net(nnx.Module):
    def __init__(self, body: nnx.Module) -> None:
        self.body, self.whiten = body, None

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.body(x)


class Seq(nnx.Module):
    """Layers (modules or functions) applied in order."""

    def __init__(self, *layers: Callable) -> None:
        self.layers = nnx.List(list(layers))

    def __call__(self, x: jax.Array) -> jax.Array:
        for f in self.layers:
            x = f(x)
        return x


class Residual(nnx.Module):
    def __init__(self, f: Callable) -> None:
        self.f = f

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + self.f(x)


def _head(cin: int, classes: int, dtype: DTypeLike, rngs: nnx.Rngs) -> Seq:
    return Seq(lambda x: x.mean((1, 2)), _linear(cin, classes, dtype, rngs), lambda z: z.astype(jnp.float32))


def convmixer(
    dim: int = 256, depth: int = 8, kernel: int = 9, patch: int = 1, classes: int = 10, dtype: DTypeLike = jnp.float32, seed: int = 0, **_
) -> _Net:
    r = nnx.Rngs(params=seed)
    blocks = [
        Seq(
            Residual(Seq(Conv(dim, dim, kernel, padding='SAME', groups=dim, dtype=dtype, rngs=r), gelu, _bn(dim, r))),
            Conv(dim, dim, 1, dtype=dtype, rngs=r),
            gelu,
            _bn(dim, r),
        )
        for _ in range(depth)
    ]
    body = Seq(Conv(3, dim, patch, stride=patch, dtype=dtype, rngs=r), gelu, _bn(dim, r), *blocks, _head(dim, classes, dtype, r))
    return _Net(body)


def resnet9(width: int = 32, logit_scale: float = 0.125, activation: str = 'relu', classes: int = 10, seed: int = 0, **_) -> _Net:
    from .model import model

    return _Net(model(width, 0.0, activation, classes=classes, logit_scale=logit_scale, seed=seed))


class _PyramidBlock(nnx.Module):
    def __init__(self, cin: int, c: int, stride: int, bottleneck: bool, expansion: int, dtype: DTypeLike, r: nnx.Rngs) -> None:
        cout = c * expansion
        if bottleneck:
            self.f = Seq(
                _bn(cin, r),
                Conv(cin, c, 1, bias=False, dtype=dtype, rngs=r),
                _bn(c, r),
                relu,
                Conv(c, c, 3, stride, 1, bias=False, dtype=dtype, rngs=r),
                _bn(c, r),
                relu,
                Conv(c, cout, 1, bias=False, dtype=dtype, rngs=r),
                _bn(cout, r),
            )
        else:
            self.f = Seq(
                _bn(cin, r),
                Conv(cin, c, 3, stride, 1, bias=False, dtype=dtype, rngs=r),
                _bn(c, r),
                relu,
                Conv(c, cout, 3, 1, 1, bias=False, dtype=dtype, rngs=r),
                _bn(cout, r),
            )
        self.stride, self.pad = stride, cout - cin

    def __call__(self, x: jax.Array) -> jax.Array:
        s = _avg_pool(x, 2) if self.stride == 2 else x
        y = self.f(x)
        if self.pad:
            s = jnp.pad(s, ((0, 0), (0, 0), (0, 0), (0, self.pad)))
        return y + s.astype(y.dtype)


def pyramidnet(
    depth: int = 110, alpha: float = 48, bottleneck: bool = False, classes: int = 10, dtype: DTypeLike = jnp.float32, seed: int = 0, **_
) -> _Net:
    r = nnx.Rngs(params=seed)
    n = (depth - 2) // (9 if bottleneck else 6)
    expansion = 4 if bottleneck else 1
    add = alpha / (3 * n)
    layers, c, cin = [Conv(3, 16, 3, 1, 1, bias=False, dtype=dtype, rngs=r), _bn(16, r)], 16.0, 16
    for stage in range(3):
        for i in range(n):
            c += add
            layers.append(_PyramidBlock(cin, int(round(c)), 2 if stage > 0 and i == 0 else 1, bottleneck, expansion, dtype, r))
            cin = int(round(c)) * expansion
    layers += [_bn(cin, r), relu, _head(cin, classes, dtype, r)]
    return _Net(Seq(*layers))


class _DenseLayer(nnx.Module):
    def __init__(self, cin: int, growth: int, dtype: DTypeLike, r: nnx.Rngs) -> None:
        self.f = Seq(
            _bn(cin, r),
            relu,
            Conv(cin, 4 * growth, 1, bias=False, dtype=dtype, rngs=r),
            _bn(4 * growth, r),
            relu,
            Conv(4 * growth, growth, 3, 1, 1, bias=False, dtype=dtype, rngs=r),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.f(x)
        return jnp.concatenate([x.astype(y.dtype), y], -1)


def densenet(depth: int = 100, growth: int = 12, classes: int = 10, dtype: DTypeLike = jnp.float32, seed: int = 0, **_) -> _Net:
    r = nnx.Rngs(params=seed)
    n = (depth - 4) // 6
    layers, c = [Conv(3, 2 * growth, 3, 1, 1, bias=False, dtype=dtype, rngs=r)], 2 * growth
    for stage in range(3):
        for _ in range(n):
            layers.append(_DenseLayer(c, growth, dtype, r))
            c += growth
        if stage < 2:
            layers += [_bn(c, r), relu, Conv(c, c // 2, 1, bias=False, dtype=dtype, rngs=r), lambda x: _avg_pool(x, 2)]
            c //= 2
    layers += [_bn(c, r), relu, _head(c, classes, dtype, r)]
    return _Net(Seq(*layers))


@jax.custom_vjp
def _shake(a: jax.Array, b: jax.Array, wf: jax.Array, wb: jax.Array) -> jax.Array:
    return wf * a + (1 - wf) * b


def _shake_fwd(a: jax.Array, b: jax.Array, wf: jax.Array, wb: jax.Array) -> tuple[jax.Array, jax.Array]:
    return _shake(a, b, wf, wb), wb


def _shake_bwd(wb: jax.Array, g: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    return wb * g, (1 - wb) * g, jnp.zeros_like(wb), jnp.zeros_like(wb)


_shake.defvjp(_shake_fwd, _shake_bwd)


class Shake(nnx.Module):
    """Shake-shake mixing: a random convex weight per image forward and an independent one backward in training,
    0.5 each at test (``deterministic``, set by ``eval()``)."""

    def __init__(self, r: nnx.Rngs) -> None:
        self.rngs, self.deterministic = r, False

    def __call__(self, a: jax.Array, b: jax.Array) -> jax.Array:
        if self.deterministic:
            return 0.5 * (a + b)
        shape = (a.shape[0], 1, 1, 1)
        wf = jax.random.uniform(self.rngs.shake(), shape, a.dtype)
        wb = jax.random.uniform(self.rngs.shake(), shape, a.dtype)
        return _shake(a, b, wf, wb)


class _Shortcut(nnx.Module):  # (the paper's: two 1x1 paths, the second shifted a pixel)
    def __init__(self, cin: int, cout: int, stride: int, dtype: DTypeLike, r: nnx.Rngs) -> None:
        self.a, self.b = Conv(cin, cout // 2, 1, bias=False, dtype=dtype, rngs=r), Conv(cin, cout // 2, 1, bias=False, dtype=dtype, rngs=r)
        self.bn, self.stride = _bn(cout, r), stride

    def __call__(self, x: jax.Array) -> jax.Array:
        x = relu(x)
        shifted = jnp.pad(x[:, 1:, 1:], ((0, 0), (0, 1), (0, 1), (0, 0)))
        s = self.stride
        return self.bn(jnp.concatenate([self.a(x[:, ::s, ::s]), self.b(shifted[:, ::s, ::s])], -1))


class _ShakeBlock(nnx.Module):
    def __init__(self, cin: int, cout: int, stride: int, dtype: DTypeLike, r: nnx.Rngs) -> None:
        branch = lambda: Seq(
            relu,
            Conv(cin, cout, 3, stride, 1, bias=False, dtype=dtype, rngs=r),
            _bn(cout, r),
            relu,
            Conv(cout, cout, 3, 1, 1, bias=False, dtype=dtype, rngs=r),
            _bn(cout, r),
        )
        self.b1, self.b2 = branch(), branch()
        self.short = _Shortcut(cin, cout, stride, dtype, r) if stride != 1 or cin != cout else None
        self.shake = Shake(nnx.Rngs(shake=r.params()))

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.shake(self.b1(x), self.b2(x))
        return (self.short(x) if self.short is not None else x).astype(y.dtype) + y


def shakeshake(depth: int = 26, width: int = 32, classes: int = 10, dtype: DTypeLike = jnp.float32, seed: int = 0, **_) -> _Net:
    r = nnx.Rngs(params=seed)
    n = (depth - 2) // 6
    layers, cin = [Conv(3, 16, 3, 1, 1, bias=False, dtype=dtype, rngs=r), _bn(16, r)], 16
    for stage, c in enumerate((width, 2 * width, 4 * width)):
        for i in range(n):
            layers.append(_ShakeBlock(cin, c, 2 if stage > 0 and i == 0 else 1, dtype, r))
            cin = c
    layers += [relu, _head(cin, classes, dtype, r)]
    return _Net(Seq(*layers))


class _WRNBlock(nnx.Module):
    def __init__(self, cin: int, cout: int, stride: int, dtype: DTypeLike, r: nnx.Rngs) -> None:
        self.bn1, self.conv1 = _bn(cin, r), Conv(cin, cout, 3, stride, 1, bias=False, dtype=dtype, rngs=r)
        self.bn2, self.conv2 = _bn(cout, r), Conv(cout, cout, 3, 1, 1, bias=False, dtype=dtype, rngs=r)
        self.short = Conv(cin, cout, 1, stride, bias=False, dtype=dtype, rngs=r) if stride != 1 or cin != cout else None

    def __call__(self, x: jax.Array) -> jax.Array:
        o = relu(self.bn1(x))
        y = self.conv2(relu(self.bn2(self.conv1(o))))
        return y + (self.short(o) if self.short is not None else x).astype(y.dtype)


def wrn(depth: int = 16, k: int = 4, classes: int = 10, dtype: DTypeLike = jnp.float32, seed: int = 0, **_) -> _Net:
    r = nnx.Rngs(params=seed)
    n = (depth - 4) // 6
    layers, cin = [Conv(3, 16, 3, 1, 1, bias=False, dtype=dtype, rngs=r)], 16
    for stage, c in enumerate((16 * k, 32 * k, 64 * k)):
        for i in range(n):
            layers.append(_WRNBlock(cin, c, 2 if stage > 0 and i == 0 else 1, dtype, r))
            cin = c
    layers += [_bn(cin, r), relu, _head(cin, classes, dtype, r)]
    return _Net(Seq(*layers))


BUILDERS = {'convmixer': convmixer, 'resnet9': resnet9, 'pyramidnet': pyramidnet, 'densenet': densenet, 'shakeshake': shakeshake, 'wrn': wrn}


def build(arch: str, **kw) -> nnx.Module:
    """``BUILDERS[arch](**kw)``; ``dtype`` (the arithmetic's) and ``seed`` (the initialisation's) pass through, and
    ``None`` means float32."""
    if kw.get('dtype') is None:
        kw['dtype'] = jnp.float32
    return BUILDERS[arch](**kw)
