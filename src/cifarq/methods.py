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
