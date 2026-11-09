"""Policy augmentation on the device, per image: TrivialAugment (Müller & Hutter 2021; ``ta_wide``, its wide ranges, one
operation a image at a uniformly drawn strength) and RandAugment (Cubuk et al. 2020; ``randaugment``, ``n = 2``
operations a image at strength ``m`` of 30). Each image draws its own operation and strength. In JAX every operation is
computed for the whole chunk and each image keeps the one it drew (no data-dependent shapes): the same result as the
Torch version's grouping by operation, at more arithmetic.

The operations are torchvision's/PIL's set: identity, auto-contrast, equalize (per channel, by the rank of each
pixel in its image's channel: the histogram-equalisation map for 1024 pixels), rotate, solarize, colour (saturation),
posterize, contrast, brightness, sharpness, shear and translate in x and y; geometric ones fill with grey (0.5).
``apply`` takes and returns normalised NHWC images (``(x - mean) / std``), working in [0, 1] in between.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax
    from jax.typing import ArrayLike

OPS = (
    'identity',
    'autocontrast',
    'equalize',
    'rotate',
    'solarize',
    'color',
    'posterize',
    'contrast',
    'brightness',
    'sharpness',
    'shear_x',
    'shear_y',
    'translate_x',
    'translate_y',
)
# (maximum strength: TrivialAugment wide, RandAugment) at strength 1; signed ops draw a sign per image
RANGES = {
    'rotate': (135.0, 30.0),
    'solarize': (1.0, 1.0),
    'color': (0.99, 0.9),
    'posterize': (6, 4),
    'contrast': (0.99, 0.9),
    'brightness': (0.99, 0.9),
    'sharpness': (0.99, 0.9),
    'shear_x': (0.99, 0.3),
    'shear_y': (0.99, 0.3),
    'translate_x': (32.0, 14.4),
    'translate_y': (32.0, 14.4),
}
SIGNED = {'rotate', 'color', 'contrast', 'brightness', 'sharpness', 'shear_x', 'shear_y', 'translate_x', 'translate_y'}


def _gray(x: jax.Array) -> jax.Array:
    return 0.299 * x[..., 0:1] + 0.587 * x[..., 1:2] + 0.114 * x[..., 2:3]


def _blend(a: jax.Array, b: jax.Array, f: jax.Array) -> jax.Array:
    """a + f (b - a)... PIL's enhance: ``degenerate`` a, image b, factor f (1 = unchanged)."""
    import jax.numpy as jnp

    return jnp.clip(a + f[:, None, None, None] * (b - a), 0, 1)


def _affine(x: jax.Array, theta: jax.Array) -> jax.Array:
    """``grid_sample`` through ``affine_grid(theta)`` with zero padding, bilinear, the pixels whose nearest source is
    outside the image filled with grey."""
    import jax
    import jax.numpy as jnp

    n, hgt, wid, c = x.shape
    xs = (2 * jnp.arange(wid) + 1) / wid - 1
    ys = (2 * jnp.arange(hgt) + 1) / hgt - 1
    gx, gy = jnp.meshgrid(xs, ys)
    base = jnp.stack([gx, gy, jnp.ones_like(gx)], -1)
    grid = jnp.einsum('hwk,nik->nhwi', base, theta)
    col = ((grid[..., 0] + 1) * wid - 1) / 2
    row = ((grid[..., 1] + 1) * hgt - 1) / 2

    def one(img: jax.Array, r: jax.Array, cl: jax.Array) -> jax.Array:
        sample = lambda ch: jax.scipy.ndimage.map_coordinates(ch, [r, cl], order=1, mode='constant', cval=0.0)
        return jax.vmap(sample, in_axes=2, out_axes=2)(img)

    out = jax.vmap(one)(x, row, col)
    rr, cc = jnp.round(row), jnp.round(col)
    inside = ((rr >= 0) & (rr <= hgt - 1) & (cc >= 0) & (cc <= wid - 1)).astype(x.dtype)[..., None]
    return out + (1 - inside) * 0.5


def _op(name: str, x: jax.Array, v: jax.Array) -> jax.Array:
    """Operation ``name`` on images ``x`` (n, H, W, 3) in [0, 1] at signed strengths ``v`` (n,) in its own units."""
    import jax
    import jax.numpy as jnp

    n = x.shape[0]
    if name == 'identity':
        return x
    if name == 'autocontrast':
        lo, hi = x.min((1, 2), keepdims=True), x.max((1, 2), keepdims=True)
        return jnp.where(hi > lo, (x - lo) / jnp.maximum(hi - lo, 1e-6), x)
    if name == 'equalize':
        flat = jnp.moveaxis(x, -1, 1).reshape(n * 3, -1)
        ranks = jax.vmap(lambda row: jnp.searchsorted(jnp.sort(row), row, side='right'))(flat)
        return jnp.moveaxis((ranks.astype(x.dtype) / flat.shape[1]).reshape(n, 3, *x.shape[1:3]), 1, -1)
    col = lambda a: a[:, None, None, None]
    if name == 'solarize':  # v: the share of the range left unsolarized, 1 = none
        return jnp.where(x >= col(1 - v), 1 - x, x)
    if name == 'posterize':  # v: bits removed (0-6)
        keep = jnp.clip(8 - jnp.round(v), 1, 8)
        q = col(2 ** (8 - keep))
        return jnp.floor(x * 255 / q) * q / 255
    if name == 'color':
        return _blend(jnp.broadcast_to(_gray(x), x.shape), x, 1 + v)
    if name == 'contrast':
        return _blend(jnp.broadcast_to(_gray(x).mean((1, 2, 3), keepdims=True), x.shape), x, 1 + v)
    if name == 'brightness':
        return _blend(jnp.zeros_like(x), x, 1 + v)
    if name == 'sharpness':
        k = jnp.asarray([[1, 1, 1], [1, 5, 1], [1, 1, 1]], x.dtype) / 13
        padded = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='edge')
        blurred = jax.lax.conv_general_dilated(
            padded,
            jnp.broadcast_to(k[:, :, None, None], (3, 3, 1, 3)),
            (1, 1),
            'VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=3,
        )
        return _blend(blurred, x, 1 + v)
    zero, one = jnp.zeros(n, x.dtype), jnp.ones(n, x.dtype)
    if name == 'rotate':
        a = v * math.pi / 180
        theta = jnp.stack([jnp.stack([jnp.cos(a), -jnp.sin(a), zero], 1), jnp.stack([jnp.sin(a), jnp.cos(a), zero], 1)], 1)
    elif name == 'shear_x':
        theta = jnp.stack([jnp.stack([one, v, zero], 1), jnp.stack([zero, one, zero], 1)], 1)
    elif name == 'shear_y':
        theta = jnp.stack([jnp.stack([one, zero, zero], 1), jnp.stack([v, one, zero], 1)], 1)
    elif name == 'translate_x':
        theta = jnp.stack([jnp.stack([one, zero, 2 * v / x.shape[2]], 1), jnp.stack([zero, one, zero], 1)], 1)
    elif name == 'translate_y':
        theta = jnp.stack([jnp.stack([one, zero, zero], 1), jnp.stack([zero, one, 2 * v / x.shape[1]], 1)], 1)
    else:
        raise ValueError(name)
    return _affine(x, theta)


def _strength(name: str, s: jax.Array, wide: bool, key: jax.Array | None = None) -> jax.Array:
    """Strengths ``s`` in [0, 1] (n,) as signed values in the operation's units (``key``: the signs' draw)."""
    import jax
    import jax.numpy as jnp

    if name not in RANGES:
        return s
    v = s * RANGES[name][0 if wide else 1]
    if name in SIGNED:
        key = key if key is not None else jax.random.key(0)
        v = jnp.where(jax.random.uniform(key, v.shape) < 0.5, -v, v)
    return v


def policy(key: jax.Array, x: jax.Array, kind: str, m: int = 9) -> jax.Array:
    """``kind`` (``ta_wide`` or ``randaugment``) on images ``x`` in [0, 1] (NHWC, float32)."""
    import jax
    import jax.numpy as jnp

    n = x.shape[0]
    rounds = 1 if kind == 'ta_wide' else 2
    for _ in range(rounds):
        key, kw, ks, kg = jax.random.split(key, 4)
        which = jax.random.randint(kw, (n,), 0, len(OPS))
        if kind == 'ta_wide':
            s = jax.random.randint(ks, (n,), 0, 31).astype(x.dtype) / 30
        elif kind == 'randaugment':
            s = jnp.full((n,), m / 30, x.dtype)
        else:
            raise ValueError(f"augment: 'none', 'ta_wide' or 'randaugment', not {kind!r}")
        out = x
        signs = jax.random.split(kg, len(OPS))
        for k, name in enumerate(OPS):
            y = _op(name, x, _strength(name, s, kind == 'ta_wide', signs[k])).astype(x.dtype)
            out = jnp.where((which == k)[:, None, None, None], y, out)
        x = out
    return x


def apply(key: jax.Array, x: jax.Array, kind: str, m: int, mean: ArrayLike, std: ArrayLike, chunk: int = 5000) -> jax.Array:
    """``policy`` on normalised images ``x`` (any float type), in chunks; normalised images of ``x``'s type back."""
    import jax
    import jax.numpy as jnp

    fn = jax.jit(lambda k, xb: ((policy(k, jnp.clip(xb.astype(jnp.float32) * std + mean, 0, 1), kind, m) - mean) / std).astype(x.dtype))
    outs = []
    for i in range(0, len(x), chunk):
        key, k = jax.random.split(key)
        outs.append(fn(k, x[i : i + chunk]))
    return jnp.concatenate(outs)
