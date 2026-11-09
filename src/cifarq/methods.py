"""The method track of the autoresearch session (docs/autoresearch/2026-10-29-footprint-arch/PLAN.md, section 5,
"Methods"): pieces ``airbench`` switches on by hyperparameter, each off by default. JAX (Flax NNX) on NHWC images;
kernels are HWIO, so a kernel's spatial axes are (0, 1) and an image's (1, 2).

- MT1 ``FlipConv`` and ``GroupMaxPool``: horizontal-flip equivariance by weight sharing. Features come in pairs (a, b),
  b the response to the mirrored image; a layer's weight is [[w1, w2], [T w2, T w1]] (T mirrors a kernel), so
  mirroring the input mirrors and swaps the output pair, and its free weights are half a plain convolution's of the
  same width. The lifting layer takes plain channels to pairs ([w; T w]). ``GroupMaxPool(2)`` pools globally, then over
  each pair: invariant to mirroring up to the boundary effects of odd-sized maps.
- MT3 ``pca_init``: each 3x3 convolution after the whitening layer, in forward order, starts as the principal
  directions of its own input patches (as airbench's first layer does for pixels), instead of a partial identity.
- MT4 ``kd_weights``: per-example weights from the student's divergence from its teacher (``disagree``: in proportion,
  clipped to [0.25, 4] times the batch mean; ``hard``: the half that disagrees most), so the epoch budget goes where
  the student is still wrong, at no extra forward passes.
- MT6 ``gated_logits``: one view first; the full test-time augmentation only for images whose top-two probability
  margin is under ``gate``; the share that needed it, for the average views counted in the footprint.
(MT2, trunk-and-tails averaging, and MT5, weight-tied depth, live in ``airbench.train`` and ``airbench.model``.)

Session 2026-10-30-equivariant-small adds ``P4MConv`` and ``GroupMaxPool`` (Q1: the flips and quarter turns, D4) and
``MaxBlurPool`` (Q2: anti-aliased downsampling), ``FieldSE`` (O2), structured receptive fields (``use_srf``, Q4) and
the Muon optimizer for the filters (``MuonSGD``, O4: an Optax transformation here).

Differences from the Torch version: ``gated_logits`` computes the full views for every image and keeps them where the
margin is small (the same result, without data-dependent shapes); ``gaussian_basis_3x3`` is orthonormalised by NumPy's
QR, an orthonormal basis of the same span (the coefficients it stores differ by a fixed rotation; the kernels it can
represent are the same); ``pca_init``'s eigenvectors are taken in float64.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from . import jx

if TYPE_CHECKING:
    from jax.typing import ArrayLike, DTypeLike


def _torch_uniform(rngs: nnx.Rngs, shape: tuple[int, ...]) -> jax.Array:
    """PyTorch's default kernel initialisation (kaiming-uniform, a = sqrt(5)) for an HWIO ``shape``."""
    return jx.torch_kernel_init()(rngs.params(), shape)


def _conv(
    x: jax.Array, kernel: jax.Array, dtype: DTypeLike, padding: str | tuple[tuple[int, int], ...] = ((1, 1), (1, 1)), stride: int = 1, groups: int = 1
) -> jax.Array:
    return jax.lax.conv_general_dilated(
        x.astype(dtype),
        kernel.astype(dtype),
        (stride, stride),
        padding,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=groups,
    )


def dirac(kernel: jax.Array, n: int | None = None) -> jax.Array:
    """``torch.nn.init.dirac_`` on the first ``n`` output channels of an HWIO kernel (all when None): those channels
    zeroed, then 1 at the centre for output i from input i (i < min(n, inputs))."""
    kh, kw, cin, cout = kernel.shape
    n = cout if n is None else min(n, cout)
    kernel = kernel.at[..., :n].set(0.0)
    m = min(n, cin)
    return kernel.at[kh // 2, kw // 2, jnp.arange(m), jnp.arange(m)].set(1.0)


# ---- structured receptive fields (Q4) --------------------------------------------------------------------------


def gaussian_basis_3x3(sigma: float = 1.0) -> np.ndarray:
    """(6, 3, 3): the Gaussian and its derivatives to second order (1, x, y, xx, xy, yy) sampled on the 3x3 grid,
    orthonormalised (structured receptive fields, Jacobsen et al. 2016). Closed under flips and quarter turns up to
    sign and mixing, so it composes with the equivariant layers."""
    r = np.array([-1.0, 0.0, 1.0])
    y, x = np.meshgrid(r, r, indexing='ij')
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    fs = [g, x * g, y * g, (x**2 / sigma**2 - 1) * g, x * y * g, (y**2 / sigma**2 - 1) * g]
    b = np.stack([f.flatten() for f in fs])
    q, _ = np.linalg.qr(b.T)  # (9, 6): orthonormal columns spanning the basis
    return q.T.reshape(6, 3, 3).astype(np.float32)


BASIS = gaussian_basis_3x3()


def srf_kernel(coeff: jax.Array) -> jax.Array:
    """Coefficients (I, O, 6) as an HWIO 3x3 kernel."""
    return jnp.einsum('iob,bhw->hwio', coeff, jnp.asarray(BASIS, coeff.dtype))


def srf_project(kernel: jax.Array) -> jax.Array:
    """An HWIO 3x3 kernel projected onto the basis: coefficients (I, O, 6)."""
    return jnp.einsum('hwio,bhw->iob', kernel, jnp.asarray(BASIS, kernel.dtype))


class Kernel3(nnx.Module):
    """A trainable HWIO kernel, stored as it is or (after ``to_srf``) as structured-receptive-field coefficients
    (2/3 of the stored weights for a 3x3 kernel)."""

    srf_able = True

    def _kernel(self) -> jax.Array:
        return srf_kernel(self.weight[...]) if self.srf else self.weight[...]

    def to_srf(self) -> None:
        if not self.srf and tuple(self.weight.shape[:2]) == (3, 3):
            self.weight = nnx.Param(srf_project(self.weight[...]))
            self.srf = True


def use_srf(net: nnx.Module) -> int:
    """Every trainable 3x3 kernel of ``net`` (plain, flip or p4m convolutions; not the frozen whitening layer) stored
    as structured-receptive-field coefficients. The number of kernels changed."""
    n = 0
    for _, node in list(nnx.iter_graph(net)):
        if isinstance(node, Kernel3) and node.srf_able and not node.srf and node.weight.ndim == 4 and tuple(node.weight.shape[:2]) == (3, 3):
            node.to_srf()
            n += 1
    return n


# ---- equivariant convolutions (MT1, Q1) --------------------------------------------------------------------------


class FlipConv(Kernel3):
    """A 3x3 convolution equivariant to horizontal flips: ``lift`` takes ``cin`` plain channels to ``cout`` pairs
    (2 x ``cout`` channels), otherwise ``cin`` pairs to ``cout`` pairs. Starts as a partial identity on the a-half."""

    order = 2

    def __init__(self, cin: int, cout: int, lift: bool = False, *, dtype: DTypeLike = jnp.float32, rngs: nnx.Rngs | None = None) -> None:
        rngs = rngs or nnx.Rngs(0)
        self.lift, self.cout, self.dtype, self.srf = lift, cout, dtype, False
        # (w1 and w2 stacked along the output dimension)
        w = _torch_uniform(rngs, (3, 3, cin, cout if lift else 2 * cout))
        w = dirac(w, min(cout, cin))
        if not lift:
            w = w.at[..., cout:].multiply(0.1)  # (the cross term small at the start)
        self.weight = nnx.Param(w)

    def full(self) -> jax.Array:
        w = self._kernel()
        t = lambda k: k[:, ::-1]
        if self.lift:
            return jnp.concatenate([w, t(w)], -1)
        w1, w2 = w[..., : self.cout], w[..., self.cout :]
        return jnp.concatenate([jnp.concatenate([w1, w2], 2), jnp.concatenate([t(w2), t(w1)], 2)], -1)

    def __call__(self, x: jax.Array) -> jax.Array:
        return _conv(x, self.full(), self.dtype)


def d4_transform(x: ArrayLike, g: int, axes: tuple[int, int] = (-3, -2)) -> jax.Array:
    """The D4 element ``g`` (0-7: ``g % 4`` quarter turns after a horizontal flip when ``g >= 4``) applied to the two
    spatial ``axes`` (default an NHWC array's H and W; an HWIO kernel's are (0, 1))."""
    if g >= 4:
        x = jnp.flip(x, axes[1])
    return jnp.rot90(x, g % 4, axes=axes)


def _d4_tables() -> tuple[list[list[int]], list[int]]:
    """(composition table mul[a][b] = a∘b, inverses), read from the action on a generic 3x3 array, so the convention is
    whatever ``d4_transform`` does."""
    x = np.arange(9.0).reshape(3, 3)
    images = [np.asarray(d4_transform(x, g, (0, 1))) for g in range(8)]
    find = lambda y: next(c for c in range(8) if np.array_equal(images[c], y))
    mul = [[find(np.asarray(d4_transform(d4_transform(x, b, (0, 1)), a, (0, 1)))) for b in range(8)] for a in range(8)]
    inv = [next(b for b in range(8) if mul[a][b] == 0) for a in range(8)]
    return mul, inv


class P4MConv(Kernel3):
    """A 3x3 convolution equivariant to the 8 flips and quarter turns (p4m, the group D4 acting on the grid): features
    come in fields of 8 (group-major: channel block g is the response to the g-transformed pattern). ``lift`` takes
    ``cin`` plain channels to ``cout`` fields (out block g = conv with g(W)); otherwise ``cin`` fields to ``cout``
    fields, out block g, in block k with kernel g(W[g^-1 k]). Stored weights are 1/8 of the expanded convolution's
    (8 cin x 8 cout); the expanded kernel is what runs (and what ``footprint.count_macs`` counts). The session's Q1."""

    order = 8

    def __init__(self, cin: int, cout: int, lift: bool = False, *, dtype: DTypeLike = jnp.float32, rngs: nnx.Rngs | None = None) -> None:
        rngs = rngs or nnx.Rngs(0)
        self.lift, self.cin, self.cout, self.dtype, self.srf = lift, cin, cout, dtype, False
        mul, inv = _d4_tables()
        self.perm = tuple(tuple(mul[inv[g]][k] for k in range(8)) for g in range(8))
        w = _torch_uniform(rngs, (3, 3, cin * (1 if lift else 8), cout))
        if lift:
            w = dirac(w, min(cout, cin))
        else:
            w5 = w.reshape(3, 3, 8, cin, cout)
            w5 = w5.at[:, :, 1:].multiply(0.1)  # (the cross-element terms small at the start)
            w5 = w5.at[:, :, 0].set(dirac(w5[:, :, 0], min(cout, cin)))  # (identity element: a partial identity per block)
            w = w5.reshape(3, 3, 8 * cin, cout)
        self.weight = nnx.Param(w)

    def full(self) -> jax.Array:
        w = self._kernel()
        sp = (0, 1)
        if self.lift:
            return jnp.concatenate([d4_transform(w, g, sp) for g in range(8)], -1)
        w5 = w.reshape(3, 3, 8, self.cin, self.cout)
        return jnp.concatenate([d4_transform(w5[:, :, list(self.perm[g])].reshape(3, 3, 8 * self.cin, self.cout), g, sp) for g in range(8)], -1)

    def __call__(self, x: jax.Array) -> jax.Array:
        return _conv(x, self.full(), self.dtype)


class GroupMaxPool(nnx.Module):
    """(N, H, W, kC) -> (N, C): the global maximum of each channel, then the maximum over each field of ``k``
    (``k=2``: the pairs of ``FlipConv``)."""

    def __init__(self, k: int = 2) -> None:
        self.k = k

    def __call__(self, x: jax.Array) -> jax.Array:
        n, kc = x.shape[0], x.shape[-1]
        return x.max((1, 2)).reshape(n, self.k, kc // self.k).max(1)


class MaxBlurPool(nnx.Module):
    """Anti-aliased downsampling by 2 (Zhang 2019, MaxBlurPool): a dense max (3x3, stride 1, centred) then a fixed
    [1,3,3,1] x [1,3,3,1] / 64 blur at stride 2 (reflect-padded). Both are symmetric about the grid's centre on
    even-sized maps, so flips and quarter turns commute with it exactly, as with a 2x2 max pool. The session's Q2."""

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.max_pool(x, (3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        c = x.shape[-1]
        f = np.array([1.0, 3.0, 3.0, 1.0], np.float32)
        filt = jnp.asarray(np.broadcast_to((f[:, None] * f[None, :] / 64)[:, :, None, None], (4, 4, 1, c)))
        padded = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='reflect')
        return _conv(padded, filt, x.dtype, padding='VALID', stride=2, groups=c)


class FieldSE(nnx.Module):
    """Squeeze-and-excitation (Hu et al.) that keeps equivariance: each channel's mean over space and over the ``k``
    members of its field, a bottleneck MLP (``reduction``), and one sigmoid scale per field channel applied to all its
    members (``k=1``: the plain block). The session's O2."""

    def __init__(self, c: int, k: int = 1, reduction: int = 8, *, dtype: DTypeLike = jnp.float32, rngs: nnx.Rngs | None = None) -> None:
        rngs = rngs or nnx.Rngs(0)
        self.k = k
        hidden = max(4, c // reduction)
        self.fc1 = nnx.Linear(c, hidden, kernel_init=jx.torch_kernel_init(), bias_init=jx.torch_bias_init(c), dtype=dtype, rngs=rngs)
        self.fc2 = nnx.Linear(
            hidden, c, kernel_init=nnx.initializers.zeros, bias_init=nnx.initializers.constant(3.0), dtype=dtype, rngs=rngs
        )  # (starts near the identity)

    def __call__(self, x: jax.Array) -> jax.Array:
        n, kc = x.shape[0], x.shape[-1]
        z = x.astype(jnp.float32).mean((1, 2)).reshape(n, self.k, kc // self.k).mean(1)
        s = jax.nn.sigmoid(self.fc2(jax.nn.gelu(self.fc1(z), approximate=False)).astype(jnp.float32)).astype(x.dtype)
        return x * jnp.tile(s, (1, self.k))[:, None, None, :]


# ---- Muon (O4) ------------------------------------------------------------------------------------------------


def newton_schulz(g: jax.Array, steps: int = 5) -> jax.Array:
    """An approximate orthogonalisation of a matrix (the quintic Newton-Schulz iteration of Muon, Jordan 2024)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(jnp.bfloat16)
    x = x / (jnp.linalg.norm(x.astype(jnp.float32)).astype(jnp.bfloat16) + 1e-7)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for _ in range(steps):
        m = x @ x.T
        x = a * x + (b * m + c * m @ m) @ x
    return (x.T if tall else x).astype(g.dtype)


def orthogonalise() -> optax.GradientTransformation:
    """Each update (an HWIO kernel) viewed as (out, in x k x k), orthogonalised (``newton_schulz``) and scaled by
    sqrt(max(1, out / (in x k x k))). The view is a column permutation of Torch's (out, in, k, k) flattening, which the
    iteration commutes with, so the result is the same."""

    def one(d: jax.Array) -> jax.Array:
        out = d.shape[-1]
        flat = d.reshape(-1, out).T
        o = newton_schulz(flat) * max(1.0, flat.shape[0] / flat.shape[1]) ** 0.5
        return o.T.reshape(d.shape)

    return optax.GradientTransformation(lambda params: optax.EmptyState(), lambda u, s, params=None: (jax.tree.map(one, u), s))


def MuonSGD(lr: ArrayLike, muon_momentum: float) -> optax.GradientTransformation:  # noqa: N802  (the Torch optimizer's name)
    """Muon for the convolution filters: the update the orthogonalised Nesterov momentum (``muon_momentum``) of the
    gradient, times -``lr``. ``airbench.train`` uses it on the 4-d filters and its SGD on the rest (airbench94_muon's
    split: Muon for the filters, SGD for the norms and the head), one schedule driving both. The session's O4."""
    return optax.chain(optax.trace(decay=muon_momentum, nesterov=True), orthogonalise(), optax.scale(-lr))


# ---- PCA initialisation (MT3), distillation weights (MT4), gated test-time views (MT6) -----------------------------


def _convs_in_order(net: nnx.Module) -> list[Kernel3]:
    """The trainable 3x3 convolutions of an airbench network after its whitening layer, in forward order."""
    return net.convs_in_order()


def pca_init(net: nnx.Module, x: ArrayLike, patches: int = 50_000, seed: int = 0) -> None:
    """MT3: each 3x3 convolution's filters set to the top principal directions (unit norm) of 3x3 patches of its input
    on normalised images ``x`` (NHWC), one layer at a time in forward order (each sees the previous layers
    re-initialised); filters beyond the patch dimension keep their initialisation. Batch norm in training mode for the
    passes, its running statistics restored after."""
    saved = nnx.state(net, nnx.BatchStat)
    net.train()
    rng = np.random.default_rng(seed)
    x = jnp.asarray(x)
    for conv in _convs_in_order(net):
        record = {}
        net(x, record=record)
        inp = np.asarray(record[id(conv)], np.float32)
        cin = inp.shape[-1]
        cols = jx.patches(np.pad(inp, ((0, 0), (1, 1), (1, 1), (0, 0))), 3)  # (n, 9 cin), (dy, dx, c) order
        cols = cols[rng.permutation(len(cols))[:patches]].astype(np.float64)
        cols = cols - cols.mean(0)
        vals, vecs = np.linalg.eigh(cols.T @ cols / len(cols))
        top = vecs[:, ::-1].T.astype(np.float32)  # (9 cin, 9 cin), largest first
        cout = conv.weight.shape[-1] if not conv.srf else conv.weight.shape[1]
        k = min(cout, top.shape[0])
        kernel = conv._kernel()
        kernel = kernel.at[..., :k].set(jnp.asarray(top[:k].reshape(k, 3, 3, cin).transpose(1, 2, 3, 0)))
        conv.weight = nnx.Param(srf_project(kernel) if conv.srf else kernel)
    nnx.update(net, saved)


def kd_weights(per_example_kl: jax.Array, kind: str) -> jax.Array:
    """MT4: weights (no gradient) for each example's loss from its divergence from the teacher."""
    kl = jax.lax.stop_gradient(per_example_kl).astype(jnp.float32)
    if kind == 'disagree':
        return jnp.clip(kl / jnp.maximum(kl.mean(), 1e-8), 0.25, 4.0)
    if kind == 'hard':
        return (kl >= jnp.median(kl)).astype(jnp.float32) * 2.0
    raise ValueError(f"kd_weighting: 'none', 'disagree' or 'hard', not {kind!r}")


def gated_logits(
    apply: Callable[[jax.Array], jax.Array], x: jax.Array, level: int, gate: float, tta: Callable[[Callable, jax.Array, int], jax.Array]
) -> tuple[jax.Array, jax.Array]:
    """MT6: (logits, share of images that got the full ``level`` views) for normalised images ``x`` through ``apply``
    (images -> logits): one view, then ``tta(apply, x, level)`` for those whose top-two softmax margin is under
    ``gate``."""
    base = apply(x)
    if gate <= 0 or level == 0:
        return base, jnp.zeros(())
    top2 = jax.lax.top_k(jax.nn.softmax(base.astype(jnp.float32), 1), 2)[0]
    need = (top2[:, 0] - top2[:, 1]) < gate
    out = jnp.where(need[:, None], tta(apply, x, level).astype(base.dtype), base)
    return out, need.astype(jnp.float32).mean()
