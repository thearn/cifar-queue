"""The method track of the autoresearch session (docs/autoresearch/2026-10-29-footprint-arch/PLAN.md, section 5,
"Methods"): pieces ``airbench`` switches on by hyperparameter, each off by default.

- MT1 ``FlipConv`` and ``PairMaxPool``: horizontal-flip equivariance by weight sharing. Features come in pairs (a, b),
  b the response to the mirrored image; a layer's weight is [[w1, w2], [T w2, T w1]] (T mirrors a kernel), so
  mirroring the input mirrors and swaps the output pair, and its free weights are half a plain convolution's of the
  same width. The lifting layer takes plain channels to pairs ([w; T w]). ``PairMaxPool`` pools globally, then over
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
the Muon optimizer for the filters (``MuonSGD``, O4).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlipConv(nn.Module):
    """A 3x3 convolution equivariant to horizontal flips: ``lift`` takes ``cin`` plain channels to ``cout`` pairs
    (2 x ``cout`` channels), otherwise ``cin`` pairs to ``cout`` pairs. Starts as a partial identity on the a-half."""

    def __init__(self, cin: int, cout: int, lift: bool = False):
        super().__init__()
        self.lift, self.cout = lift, cout
        # (w1 and w2 stacked along the output dimension: a 4-d tensor, as the memory formats expect)
        self.weight = nn.Parameter(torch.zeros((cout if lift else 2 * cout), cin, 3, 3))
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
            nn.init.dirac_(self.weight[:min(cout, cin)])
            if not lift:
                self.weight[cout:].mul_(0.1)                 # (the cross term small at the start)

    def full(self):
        t = lambda w: w.flip(-1)
        if self.lift:
            return torch.cat([self.weight, t(self.weight)], 0)
        w1, w2 = self.weight[:self.cout], self.weight[self.cout:]
        return torch.cat([torch.cat([w1, w2], 1), torch.cat([t(w2), t(w1)], 1)], 0)

    def forward(self, x):
        return F.conv2d(x, self.full().to(x.dtype), padding=1)


class PairMaxPool(nn.Module):
    """(N, 2C, H, W) -> (N, C): the global maximum of each channel, then the maximum over each pair."""

    def forward(self, x):
        n, c2 = x.shape[:2]
        return x.amax((2, 3)).view(n, 2, c2 // 2).amax(1)


FlipConv.order = 2


def d4_transform(x, g: int):
    """The D4 element ``g`` (0-7: ``g % 4`` quarter turns after a horizontal flip when ``g >= 4``) applied to the last
    two (spatial) dimensions."""
    return torch.rot90(x.flip(-1) if g >= 4 else x, g % 4, dims=(-2, -1))


def _d4_tables():
    """(composition table mul[a][b] = a∘b, inverses), read from the action on a generic 3x3 array, so the convention is
    whatever ``d4_transform`` does."""
    x = torch.arange(9.).view(3, 3)
    images = [d4_transform(x, g) for g in range(8)]
    find = lambda y: next(c for c in range(8) if torch.equal(images[c], y))
    mul = [[find(d4_transform(d4_transform(x, b), a)) for b in range(8)] for a in range(8)]
    inv = [next(b for b in range(8) if mul[a][b] == 0) for a in range(8)]
    return mul, inv


class P4MConv(nn.Module):
    """A 3x3 convolution equivariant to the 8 flips and quarter turns (p4m, the group D4 acting on the grid): features
    come in fields of 8 (group-major: channel block g is the response to the g-transformed pattern). ``lift`` takes
    ``cin`` plain channels to ``cout`` fields (out block g = conv with g(W)); otherwise ``cin`` fields to ``cout``
    fields, out block g, in block k with kernel g(W[g^-1 k]). Stored weights are 1/8 of the expanded convolution's
    (8 cin x 8 cout); the expanded kernel is what runs (and what ``footprint.count_macs`` counts). The session's Q1."""

    order = 8

    def __init__(self, cin: int, cout: int, lift: bool = False):
        super().__init__()
        self.lift, self.cin, self.cout = lift, cin, cout
        mul, inv = _d4_tables()
        self.register_buffer('perm', torch.tensor([[mul[inv[g]][k] for k in range(8)] for g in range(8)]), persistent=False)
        self.weight = nn.Parameter(torch.zeros(cout, cin * (1 if lift else 8), 3, 3))
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
            if lift:
                nn.init.dirac_(self.weight[:min(cout, cin)])
            else:
                w5 = self.weight.view(cout, 8, cin, 3, 3)
                w5[:, 1:].mul_(0.1)                          # (the cross-element terms small at the start)
                nn.init.dirac_(w5[:min(cout, cin), 0])       # (identity element: a partial identity per block)

    def full(self):
        if self.lift:
            return torch.cat([d4_transform(self.weight, g) for g in range(8)], 0)
        w5 = self.weight.view(self.cout, 8, self.cin, 3, 3)
        return torch.cat([d4_transform(w5[:, self.perm[g]].reshape(self.cout, 8 * self.cin, 3, 3), g)
                          for g in range(8)], 0)

    def forward(self, x):
        return F.conv2d(x, self.full().to(x.dtype), padding=1)


class GroupMaxPool(nn.Module):
    """(N, kC, H, W) -> (N, C): the global maximum of each channel, then the maximum over each field of ``k``
    (``k=2``: ``PairMaxPool``)."""

    def __init__(self, k: int = 2):
        super().__init__()
        self.k = k

    def forward(self, x):
        n, kc = x.shape[:2]
        return x.amax((2, 3)).view(n, self.k, kc // self.k).amax(1)


class MaxBlurPool(nn.Module):
    """Anti-aliased downsampling by 2 (Zhang 2019, MaxBlurPool): a dense max (3x3, stride 1, centred) then a fixed
    [1,3,3,1] x [1,3,3,1] / 64 blur at stride 2 (reflect-padded). Both are symmetric about the grid's centre on
    even-sized maps, so flips and quarter turns commute with it exactly, as with ``MaxPool2d(2)``. The session's Q2."""

    def __init__(self):
        super().__init__()
        f = torch.tensor([1., 3., 3., 1.])
        self.register_buffer('filt', (f[:, None] * f[None, :] / 64).view(1, 1, 4, 4), persistent=False)

    def forward(self, x):
        x = F.max_pool2d(x, 3, 1, 1)
        c = x.shape[1]
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), self.filt.to(x.dtype).expand(c, 1, 4, 4), stride=2,
                        groups=c)


class FieldSE(nn.Module):
    """Squeeze-and-excitation (Hu et al.) that keeps equivariance: each channel's mean over space and over the ``k``
    members of its field, a bottleneck MLP (``reduction``), and one sigmoid scale per field channel applied to all its
    members (``k=1``: the plain block). The session's O2."""

    def __init__(self, c: int, k: int = 1, reduction: int = 8):
        super().__init__()
        self.k = k
        hidden = max(4, c // reduction)
        self.fc1, self.fc2 = nn.Linear(c, hidden), nn.Linear(hidden, c)
        nn.init.zeros_(self.fc2.weight); nn.init.constant_(self.fc2.bias, 3.0)    # (starts near the identity)

    def forward(self, x):
        n, kc = x.shape[:2]
        # (pooled in fp32, then in the layers' own precision: under airbench's fp16 the Linear weights are fp16)
        z = x.float().mean((2, 3)).view(n, self.k, kc // self.k).mean(1).to(self.fc1.weight.dtype)
        s = torch.sigmoid(self.fc2(F.gelu(self.fc1(z)))).to(x.dtype)
        return x * s.repeat(1, self.k).view(n, kc, 1, 1)


def gaussian_basis_3x3(sigma: float = 1.0) -> torch.Tensor:
    """(6, 3, 3): the Gaussian and its derivatives to second order (1, x, y, xx, xy, yy) sampled on the 3x3 grid,
    orthonormalised (structured receptive fields, Jacobsen et al. 2016). Closed under flips and quarter turns up to
    sign and mixing, so it composes with the equivariant layers."""
    r = torch.tensor([-1., 0., 1.])
    y, x = torch.meshgrid(r, r, indexing='ij')
    g = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    fs = [g, x * g, y * g, (x ** 2 / sigma ** 2 - 1) * g, x * y * g, (y ** 2 / sigma ** 2 - 1) * g]
    b = torch.stack([f.flatten() for f in fs])
    q, _ = torch.linalg.qr(b.T)                              # (9, 6): orthonormal columns spanning the basis
    return q.T.reshape(6, 3, 3)


class SRFKernel(nn.Module):
    """A parametrization: a 3x3 weight (..., 3, 3) as coefficients (..., 6) on ``gaussian_basis_3x3`` (2/3 of the
    stored weights); ``right_inverse`` projects an initialisation onto the basis. The session's Q4."""

    def __init__(self):
        super().__init__()
        self.register_buffer('basis', gaussian_basis_3x3(), persistent=False)

    def forward(self, coeff):
        return torch.einsum('...b,bij->...ij', coeff, self.basis.to(coeff.dtype))

    def right_inverse(self, w):
        return torch.einsum('...ij,bij->...b', w, self.basis.to(w.dtype))


def use_srf(net) -> int:
    """Every trainable 3x3 kernel of ``net`` (plain, flip or p4m convolutions; not the frozen whitening layer) stored
    as structured-receptive-field coefficients. The number of kernels changed."""
    from torch.nn.utils import parametrize
    n = 0
    for m in net.modules():
        w = getattr(m, 'weight', None)
        if isinstance(w, nn.Parameter) and w.requires_grad and w.dim() == 4 and tuple(w.shape[-2:]) == (3, 3):
            parametrize.register_parametrization(m, 'weight', SRFKernel())
            n += 1
    return n


def newton_schulz(g, steps: int = 5):
    """An approximate orthogonalisation of a matrix (the quintic Newton-Schulz iteration of Muon, Jordan 2024)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    x = x / (x.norm() + 1e-7)
    tall = x.size(0) > x.size(1)
    if tall:
        x = x.T
    for _ in range(steps):
        m = x @ x.T
        x = a * x + (b * m + c * m @ m) @ x
    return (x.T if tall else x).to(g.dtype)


class MuonSGD(torch.optim.Optimizer):
    """Muon for the groups marked ``muon`` (each update the orthogonalised Nesterov momentum of a weight viewed as
    (out, in x k x k), scaled by sqrt(max(1, out/in))), Nesterov SGD with coupled weight decay for the rest
    (``torch.optim.SGD``'s rule), so one scheduler drives both (airbench94_muon's split: Muon for the convolution
    filters, SGD for the norms and the head). The session's O4."""

    def __init__(self, groups, momentum: float = 0.85, muon_momentum: float = 0.6):
        super().__init__(groups, dict(lr=1.0, weight_decay=0.0, muon=False))
        self.momentum, self.muon_momentum = momentum, muon_momentum

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            mu = self.muon_momentum if g['muon'] else self.momentum
            for p in g['params']:
                if p.grad is None:
                    continue
                d = p.grad.float()
                if not g['muon'] and g['weight_decay']:
                    d = d + g['weight_decay'] * p.float()
                buf = self.state[p].setdefault('buf', torch.zeros_like(d))
                buf.mul_(mu).add_(d)
                d = d + mu * buf
                if g['muon']:
                    flat = d.reshape(d.size(0), -1)              # (filters are channels_last: reshape, not view)
                    d = newton_schulz(flat).reshape(d.shape) * max(1.0, flat.size(0) / flat.size(1)) ** 0.5
                p.add_(d.to(p.dtype), alpha=-g['lr'])


def _convs_in_order(net):
    """The trainable 3x3 convolutions of an airbench network after its whitening layer, in forward order."""
    return [m for m in net.modules() if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3) and m.weight.requires_grad]


@torch.no_grad()
def pca_init(net, x, patches: int = 50_000):
    """MT3: each 3x3 convolution's filters set to the top principal directions (unit norm) of 3x3 patches of its input
    on normalised images ``x``, one layer at a time in forward order (each sees the previous layers re-initialised);
    filters beyond the patch dimension keep their initialisation. Batch norm in training mode for the passes, its
    running statistics restored after."""
    saved = {k: v.clone() for k, v in net.state_dict().items() if 'running' in k or 'num_batches' in k}
    was = net.training
    net.train()
    for conv in _convs_in_order(net):
        grabbed = {}
        handle = conv.register_forward_hook(lambda m, inp, out: grabbed.__setitem__('x', inp[0]))
        net(x)
        handle.remove()
        inp = grabbed['x'].float()
        cin = inp.shape[1]
        cols = F.unfold(inp, 3, padding=1).transpose(1, 2).reshape(-1, cin * 9)
        cols = cols[torch.randperm(len(cols), device=cols.device)[:patches]]
        cols = cols - cols.mean(0)
        vals, vecs = torch.linalg.eigh((cols.T @ cols / len(cols)).double())
        top = vecs.flip(1).T.float()                         # (cin*9, cin*9), largest first
        k = min(conv.out_channels, top.shape[0])
        conv.weight[:k] = top[:k].reshape(k, cin, 3, 3).to(conv.weight.dtype)
    net.load_state_dict({**net.state_dict(), **saved})
    net.train(was)


def kd_weights(per_example_kl, kind: str):
    """MT4: weights (detached) for each example's loss from its divergence from the teacher."""
    kl = per_example_kl.detach().float()
    if kind == 'disagree':
        return (kl / kl.mean().clamp_min(1e-8)).clamp(0.25, 4.0)
    if kind == 'hard':
        return (kl >= kl.median()).float() * 2.0
    raise ValueError(f"kd_weighting: 'none', 'disagree' or 'hard', not {kind!r}")


def gated_logits(net, x, level: int, gate: float, tta):
    """MT6: (logits, share of images that got the full ``level`` views) for normalised images ``x``: one view, then
    ``tta(net, x, level)`` for those whose top-two softmax margin is under ``gate``."""
    base = net(x)
    if gate <= 0 or level == 0:
        return base, 0.0
    top2 = F.softmax(base.float(), 1).topk(2, 1).values
    need = (top2[:, 0] - top2[:, 1]) < gate
    out = base.clone()
    if need.any():
        out[need] = tta(net, x[need], level).to(out.dtype)
    return out, need.float().mean().item()
