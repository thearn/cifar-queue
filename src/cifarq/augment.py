"""Policy augmentation on the GPU, per image: TrivialAugment (Müller & Hutter 2021; ``ta_wide``, its wide ranges, one
operation a image at a uniformly drawn strength) and RandAugment (Cubuk et al. 2020; ``randaugment``, ``n = 2``
operations a image at strength ``m`` of 30). Each image draws its own operation and strength; images that drew the
same operation are transformed together with per-image parameters, so a batch costs one kernel group per operation.

The operations are torchvision's/PIL's set: identity, auto-contrast, equalize (per channel, by the rank of each
pixel in its image's channel: the histogram-equalisation map for 1024 pixels), rotate, solarize, colour (saturation),
posterize, contrast, brightness, sharpness, shear and translate in x and y; geometric ones fill with grey (0.5).
``apply`` takes and returns normalised images (``(x - mean) / std``), working in [0, 1] in between.
"""
from __future__ import annotations

import math

OPS = ('identity', 'autocontrast', 'equalize', 'rotate', 'solarize', 'color', 'posterize', 'contrast', 'brightness',
       'sharpness', 'shear_x', 'shear_y', 'translate_x', 'translate_y')
# (maximum strength: TrivialAugment wide, RandAugment) at strength 1; signed ops draw a sign per image
RANGES = {'rotate': (135.0, 30.0), 'solarize': (1.0, 1.0), 'color': (0.99, 0.9), 'posterize': (6, 4), 'contrast': (0.99, 0.9),
          'brightness': (0.99, 0.9), 'sharpness': (0.99, 0.9), 'shear_x': (0.99, 0.3), 'shear_y': (0.99, 0.3),
          'translate_x': (32.0, 14.4), 'translate_y': (32.0, 14.4)}
SIGNED = {'rotate', 'color', 'contrast', 'brightness', 'sharpness', 'shear_x', 'shear_y', 'translate_x', 'translate_y'}


def _gray(x):
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def _blend(a, b, f):
    """a + f (b - a)... PIL's enhance: ``degenerate`` a, image b, factor f (1 = unchanged)."""
    return (a + f.view(-1, 1, 1, 1) * (b - a)).clamp(0, 1)


def _affine(x, theta):
    import torch.nn.functional as F
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    ones = x.new_ones(x.shape[0], 1, *x.shape[2:])
    out = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    mask = F.grid_sample(ones, grid, mode='nearest', padding_mode='zeros', align_corners=False)
    return out + (1 - mask) * 0.5


def _op(name: str, x, v):
    """Operation ``name`` on images ``x`` (n, 3, H, W) in [0, 1] at signed strengths ``v`` (n,) in its own units."""
    import torch
    import torch.nn.functional as F
    n, dev = x.shape[0], x.device
    if name == 'identity':
        return x
    if name == 'autocontrast':
        lo, hi = x.amin((2, 3), keepdim=True), x.amax((2, 3), keepdim=True)
        return torch.where(hi > lo, (x - lo) / (hi - lo).clamp_min(1e-6), x)
    if name == 'equalize':
        flat = x.reshape(n * 3, -1)
        ranks = torch.searchsorted(flat.sort(1).values.contiguous(), flat.contiguous(), right=True)
        return (ranks.float() / flat.shape[1]).reshape(x.shape)
    if name == 'solarize':                                   # v: the share of the range left unsolarized, 1 = none
        t = (1 - v).view(-1, 1, 1, 1)
        return torch.where(x >= t, 1 - x, x)
    if name == 'posterize':                                  # v: bits removed (0-6)
        keep = (8 - v.round()).clamp(1, 8).view(-1, 1, 1, 1)
        q = 2 ** (8 - keep)
        return torch.floor(x * 255 / q) * q / 255
    if name == 'color':
        return _blend(_gray(x).expand_as(x), x, 1 + v)
    if name == 'contrast':
        return _blend(_gray(x).mean((1, 2, 3), keepdim=True).expand_as(x), x, 1 + v)
    if name == 'brightness':
        return _blend(torch.zeros_like(x), x, 1 + v)
    if name == 'sharpness':
        k = torch.tensor([[1, 1, 1], [1, 5, 1], [1, 1, 1]], device=dev, dtype=x.dtype) / 13
        blurred = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), k.expand(3, 1, 3, 3), groups=3)
        return _blend(blurred, x, 1 + v)
    zero, one = torch.zeros(n, device=dev, dtype=x.dtype), torch.ones(n, device=dev, dtype=x.dtype)
    if name == 'rotate':
        a = v * math.pi / 180
        theta = torch.stack([torch.stack([a.cos(), -a.sin(), zero], 1), torch.stack([a.sin(), a.cos(), zero], 1)], 1)
    elif name == 'shear_x':
        theta = torch.stack([torch.stack([one, v, zero], 1), torch.stack([zero, one, zero], 1)], 1)
    elif name == 'shear_y':
        theta = torch.stack([torch.stack([one, zero, zero], 1), torch.stack([v, one, zero], 1)], 1)
    elif name == 'translate_x':
        theta = torch.stack([torch.stack([one, zero, 2 * v / x.shape[-1]], 1), torch.stack([zero, one, zero], 1)], 1)
    elif name == 'translate_y':
        theta = torch.stack([torch.stack([one, zero, zero], 1), torch.stack([zero, one, 2 * v / x.shape[-2]], 1)], 1)
    else:
        raise ValueError(name)
    return _affine(x, theta)


def _strength(name: str, s, wide: bool):
    """Strengths ``s`` in [0, 1] (n,) as signed values in the operation's units."""
    import torch
    if name not in RANGES:
        return s
    v = s * RANGES[name][0 if wide else 1]
    if name in SIGNED:
        v = torch.where(torch.rand_like(v) < 0.5, -v, v)
    return v


def policy(x, kind: str, m: int = 9):
    """``kind`` (``ta_wide`` or ``randaugment``) on images ``x`` in [0, 1]."""
    import torch
    n, dev = x.shape[0], x.device
    rounds = 1 if kind == 'ta_wide' else 2
    for _ in range(rounds):
        which = torch.randint(0, len(OPS), (n,), device=dev)
        if kind == 'ta_wide':
            s = torch.randint(0, 31, (n,), device=dev).to(x.dtype) / 30
        elif kind == 'randaugment':
            s = torch.full((n,), m / 30, device=dev, dtype=x.dtype)
        else:
            raise ValueError(f"augment: 'none', 'ta_wide' or 'randaugment', not {kind!r}")
        out = x.clone()
        for k, name in enumerate(OPS):
            idx = (which == k).nonzero().squeeze(1)
            if len(idx):
                out[idx] = _op(name, x[idx], _strength(name, s[idx], kind == 'ta_wide')).to(x.dtype)
        x = out
    return x


def apply(x, kind: str, m: int, mean, std, chunk: int = 5000):
    """``policy`` on normalised images ``x`` (any float type), in chunks; normalised images of ``x``'s type back."""
    import torch
    out = torch.empty_like(x)
    for i in range(0, len(x), chunk):
        xb = (x[i:i + chunk].float() * std + mean).clamp(0, 1)
        out[i:i + chunk] = ((policy(xb, kind, m) - mean) / std).to(x.dtype)
    return out
