"""The airbench family (K. Jordan, "94 % on CIFAR-10 in 3.29 seconds on a single GPU", 2024; github.com/KellerJordan/
cifar10-airbench at 4c1b6d1, ``legacy/airbench94.py`` and ``legacy/airbench96.py``) ported to this project's dataset
and job contract, every choice a hyperparameter (``DEFAULTS``; ``AIRBENCH94`` and ``AIRBENCH96`` are the two published
recipes as points of it).

The network: a frozen 2x2 patch-whitening convolution (its 24 filters the training patches' eigenvectors over the
square roots of their eigenvalues, and their negatives; its bias trained for the first ``whiten_bias_epochs``), GELU,
then three groups of ``depth`` 3x3 convolutions at ``widths``, each group pooling after its first convolution, batch
norm without a learned scale and its bias at ``bias_scaler`` times the learning rate, depth 3 adding a residual over
the last two; a 3x3 max pool, a linear layer without bias, logits times ``scaling_factor``. Convolutions start as
partial identities (dirac).

The training: Nesterov SGD at a learning rate, momentum and weight decay expressed per 1024 examples (independent of
each other), the loss summed over the batch with label smoothing; a linear warmup over ``warmup_frac`` of the steps
from 0.2 of the peak, then linear to ``final_lr_frac``; Lookahead every 5 steps (its pull growing as the cube of the
progress); alternating flip (each image flipped at random once, then all flipped together every other epoch), random
translation from reflect padding, cutout. Weights are fp16 and batch norm fp32, as in the original (``precision``).
Evaluation averages ``tta_level`` views: 0 the image, 1 and its mirror, 2 and both shifted one pixel up-left and
down-right (6 views, the shifted ones at half weight).

Additions, off by default: ``train_on='train+val'`` (all 50000 images, the published setting, for a record-format
run: the validation accuracy is then on trained images and only reported), and distillation from a teacher saved by
``train(save=...)`` (``kd_teacher``: its directory, or several separated by commas for an ensemble; ``kd_weight`` of the loss is the KL divergence to its softened
predictions at ``kd_temp``, the teacher run on each augmented batch).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

DEFAULTS = dict(
    # network
    widths=(64, 256, 256), depth=2, scaling_factor=1 / 9, bn_momentum=0.4, whiten_kernel=2, whiten_eps=5e-4,
    # optimisation
    epochs=9.9, batch_size=1024, lr=11.5, momentum=0.85, weight_decay=0.0153, bias_scaler=64.0, label_smoothing=0.2,
    whiten_bias_epochs=3, warmup_frac=0.23, final_lr_frac=0.07, lookahead=True,
    # augmentation
    flip=True, translate=2, cutout=0,
    # evaluation
    tta_level=2,
    # additions
    arch='airbench', arch_kw={}, bn_scale=False, param_cap=0, tie=0, flip_eq=0, pca_init=False, tails=0,
    tail_epochs=3, kd_weighting='none', tta_gate=0.0, optimizer='sgd', beta2=0.99, augment='none', augment_m=9,
    train_on='train', kd_teacher='', kd_weight=0.0, kd_temp=4.0, precision='fp16', eq_group='d1', blur_pool=False,
    se=0, srf=False, filter_keep=1.0, filter_proxy_widths=(32, 64, 64), filter_proxy_epochs=0.0, muon_lr=0.24,
    muon_momentum=0.6,
    test='open',
    seed=0)

AIRBENCH94 = dict(DEFAULTS)
AIRBENCH96 = {**DEFAULTS, 'widths': (128, 384, 512), 'depth': 3, 'bn_momentum': 0.1, 'epochs': 37.0, 'lr': 9.0,
              'weight_decay': 0.012, 'warmup_frac': 0.1, 'final_lr_frac': 0.0, 'translate': 4, 'cutout': 12}

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def model(widths=(64, 256, 256), depth: int = 2, scaling_factor: float = 1 / 9, bn_momentum: float = 0.4,
          whiten_kernel: int = 2, classes: int = 10, tie: int = 0, flip_eq: int = 0, eq_group: str = 'd1',
          blur_pool: bool = False, se: int = 0, srf: bool = False):
    """The network (fp32; ``train`` casts it). ``bn_momentum`` is PyTorch's (the weight of the new batch).
    ``tie`` (method MT5): each group's last convolution applied ``tie`` more times, its weights shared, a batch norm of
    its own each time (depth at no stored convolution weights). ``flip_eq`` (method MT1): the last ``flip_eq`` groups
    made equivariant to horizontal flips (``methods.FlipConv``: channel pairs, half the free weights of a plain
    convolution of twice the width), the features pooled over each pair at the end (invariance by construction, not by
    test-time mirroring). ``eq_group`` 'd4': those groups equivariant to the flips and quarter turns instead
    (``methods.P4MConv``, fields of 8, 1/8 of the expanded convolution's weights). ``blur_pool``: each group's
    downsampling anti-aliased (``methods.MaxBlurPool``) instead of ``MaxPool2d(2)``. ``se``: a squeeze-and-excitation
    block at the end of each group with this reduction (``methods.FieldSE``, shared over a field; 0: none). ``srf``:
    every trainable 3x3 kernel stored as structured-receptive-field coefficients (``methods.use_srf``)."""
    import torch
    from torch import nn
    from . import methods

    class BatchNorm(nn.BatchNorm2d):
        def __init__(self, c):
            super().__init__(c, eps=1e-12, momentum=bn_momentum)
            self.weight.requires_grad = False

    class Conv(nn.Conv2d):
        def __init__(self, cin, cout, k=3, padding='same', bias=False):
            super().__init__(cin, cout, kernel_size=k, padding=padding, bias=bias)

        def reset_parameters(self):
            super().reset_parameters()
            if self.bias is not None:
                self.bias.data.zero_()
            w = self.weight.data
            torch.nn.init.dirac_(w[:w.size(1)])

    eq_conv = {'d1': methods.FlipConv, 'd4': methods.P4MConv}[eq_group]
    k = eq_conv.order

    class PairNorm(BatchNorm):                               # (one set of statistics for all members of a field)
        def forward(self, x):
            n, kc, hh, ww_ = x.shape
            return super().forward(x.reshape(n * k, kc // k, hh, ww_)).reshape(n, kc, hh, ww_)

    class Group(nn.Module):
        def __init__(self, cin, cout, eq: str = ''):
            super().__init__()
            if eq:                                           # 'lift' (plain in, pairs out) or 'pair' (pairs in and out)
                self.conv1, self.norm1 = eq_conv(cin, cout, lift=eq == 'lift'), PairNorm(cout)
                self.convs = nn.ModuleList(eq_conv(cout, cout) for _ in range(depth - 1))
                norm = lambda: PairNorm(cout)
            else:
                self.conv1, self.norm1 = Conv(cin, cout), BatchNorm(cout)
                self.convs = nn.ModuleList(Conv(cout, cout) for _ in range(depth - 1))
                norm = lambda: BatchNorm(cout)
            self.pool = methods.MaxBlurPool() if blur_pool else nn.MaxPool2d(2)
            self.se = methods.FieldSE(cout, k if eq else 1, se) if se else None
            self.norms = nn.ModuleList(norm() for _ in range(depth - 1 + (tie if depth > 1 else 0)))
            self.activ = nn.GELU()

        def forward(self, x):
            x = self.activ(self.norm1(self.pool(self.conv1(x))))
            x0 = x
            convs = list(self.convs) + ([self.convs[-1]] * tie if len(self.convs) else [])
            for i, (conv, norm) in enumerate(zip(convs, self.norms)):
                x = norm(conv(x))
                if depth >= 3 and i == len(convs) - 1:
                    x = x + x0                                   # (depth 3: the residual over the last two)
                x = self.activ(x)
            return self.se(x) if self.se is not None else x

    class Mul(nn.Module):
        def __init__(self, s):
            super().__init__()
            self.s = s

        def forward(self, x):
            return x * self.s

    ww = 2 * 3 * whiten_kernel ** 2
    w1, w2, w3 = widths
    eqs = [''] * (3 - flip_eq) + (['lift'] + ['pair'] * (flip_eq - 1) if flip_eq else [])
    pool = [methods.GroupMaxPool(k)] if flip_eq else [nn.MaxPool2d(3), nn.Flatten()]
    net = nn.Sequential(Conv(3, ww, whiten_kernel, padding=0, bias=True), nn.GELU(), Group(ww, w1, eqs[0]),
                        Group(w1, w2, eqs[1]), Group(w2, w3, eqs[2]), *pool, nn.Linear(w3, classes, bias=False),
                        Mul(scaling_factor))
    net[0].weight.requires_grad = False
    if srf:
        methods.use_srf(net)
    return net


def build(h: dict):
    """The network ``h`` describes (fp32), and its whitening convolution (or None): ``arch='airbench'`` is ``model``,
    any other a builder of ``cifarq.nets`` with ``arch_kw``."""
    if h.get('arch', 'airbench') == 'airbench':
        net = model(tuple(h['widths']), h['depth'], h['scaling_factor'], h['bn_momentum'], h['whiten_kernel'],
                    tie=int(h.get('tie', 0)), flip_eq=int(h.get('flip_eq', 0)), eq_group=h.get('eq_group', 'd1'),
                    blur_pool=bool(h.get('blur_pool', False)), se=int(h.get('se', 0)), srf=bool(h.get('srf', False)))
        return net, net[0]
    from . import nets
    net = nets.build(h['arch'], **h.get('arch_kw', {}))
    for m in net.modules():
        if isinstance(m, nn_batchnorm()):
            m.momentum = h['bn_momentum']
    return net, net.whiten


def nn_batchnorm():
    import torch
    return (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)


def whiten_init(layer, x, eps: float):
    """Set the whitening convolution from normalised images ``x`` (N, 3, 32, 32)."""
    import torch
    c, (h, w) = x.shape[1], layer.weight.shape[2:]
    patches = x.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()
    flat = patches.view(len(patches), -1)
    vals, vecs = torch.linalg.eigh(flat.T @ flat / len(flat), UPLO='U')
    vals, vecs = vals.flip(0).view(-1, 1, 1, 1), vecs.T.reshape(c * h * w, c, h, w).flip(0)
    scaled = vecs / torch.sqrt(vals + eps)
    layer.weight.data[:] = torch.cat((scaled, -scaled)).to(layer.weight.dtype)


def tta_logits(net, x, level: int):
    """Logits of normalised images ``x`` averaged over ``level``'s views (0: 1 view, 1: 2, 2: 6; 3: 18, every shift of
    -2, 0 or 2 pixels each way, mirrored: the session's P2, inference compute being uncapped)."""
    import torch.nn.functional as F
    mirror = lambda v: 0.5 * net(v) + 0.5 * net(v.flip(-1))
    if level == 0:
        return net(x)
    if level == 1:
        return mirror(x)
    if level >= 3:
        padded = F.pad(x, (2,) * 4, 'reflect')
        return sum(mirror(padded[:, :, dy:dy + 32, dx:dx + 32]) for dy in (0, 2, 4) for dx in (0, 2, 4)) / 9
    padded = F.pad(x, (1,) * 4, 'reflect')
    shifted = 0.5 * (mirror(padded[:, :, 0:32, 0:32]) + mirror(padded[:, :, 2:34, 2:34]))
    return 0.5 * mirror(x) + 0.5 * shifted


def _crop(images, size: int):
    """Random translations: crops of ``size`` from reflect-padded images, one shift per image (airbench's)."""
    import torch
    r = (images.size(-1) - size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    out = torch.empty((len(images), 3, size, size), device=images.device, dtype=images.dtype)
    if r <= 2:
        for sy in range(-r, r + 1):
            for sx in range(-r, r + 1):
                m = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                out[m] = images[m, :, r + sy:r + sy + size, r + sx:r + sx + size]
    else:
        tmp = torch.empty((len(images), 3, size, size + 2 * r), device=images.device, dtype=images.dtype)
        for s in range(-r, r + 1):
            m = shifts[:, 0] == s
            tmp[m] = images[m, :, r + s:r + s + size, :]
        for s in range(-r, r + 1):
            m = shifts[:, 1] == s
            out[m] = tmp[m, :, :, r + s:r + s + size]
    return out


def _cutout(images, size: int):
    import torch
    n, _, h, w = images.shape
    cy = torch.randint(0, h - size + 1, size=(n,), device=images.device)
    cx = torch.randint(0, w - size + 1, size=(n,), device=images.device)
    dy = torch.arange(h, device=images.device).view(1, 1, h, 1) - cy.view(-1, 1, 1, 1)
    dx = torch.arange(w, device=images.device).view(1, 1, 1, w) - cx.view(-1, 1, 1, 1)
    return images.masked_fill((dy >= 0) & (dy < size) & (dx >= 0) & (dx < size), 0)


def param_counts(net) -> dict:
    """{'stored': every parameter tensor's size, frozen ones included; 'trainable': those with gradients} (batch-norm
    running statistics are buffers, not counted; batch norm's frozen scale is a parameter and is counted)."""
    return {'stored': sum(p.numel() for p in net.parameters()),
            'trainable': sum(p.numel() for p in net.parameters() if p.requires_grad)}


def _groups(net, whiten, bn_scale: bool):
    """(batch-norm biases, every other parameter training updates): batch norm's scale is frozen unless ``bn_scale``
    (then it trains with the others), the whitening weight is frozen and its bias trains (until switched off)."""
    biases, others, seen = [], [], set()
    for m in net.modules():
        if isinstance(m, nn_batchnorm()):
            m.weight.requires_grad = bn_scale
            biases.append(m.bias)
            if bn_scale:
                others.append(m.weight)
            seen |= {id(m.weight), id(m.bias)}
    for p in net.parameters():
        if id(p) not in seen and (p.requires_grad or (whiten is not None and p is whiten.bias)):
            others.append(p)
    return biases, others


def train(data: str | None = None, save: str | None = None, report=None, device: str = 'auto', **hyperparameters) -> dict:
    """Train one airbench network with ``DEFAULTS`` updated by ``hyperparameters`` (an unknown name is an error); the
    same contract as ``cifarq.train.train``: validation and test accuracy (``test_acc`` at ``tta_level``, and without
    TTA beside it), time, throughput, per-epoch history, the parameter counts; ``report`` early stopping after each
    epoch; ``save`` a directory for ``load``."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .data import dataset
    unknown = set(hyperparameters) - set(DEFAULTS)
    if unknown:
        raise TypeError(f'unknown hyperparameters: {sorted(unknown)}')
    h = {**DEFAULTS, **hyperparameters}
    h['widths'] = tuple(int(w) for w in h['widths'])
    h['filter_proxy_widths'] = tuple(int(w) for w in h['filter_proxy_widths'])
    h['arch_kw'] = dict(h['arch_kw'])
    for k in ('depth', 'batch_size', 'translate', 'cutout', 'tta_level', 'whiten_kernel', 'whiten_bias_epochs', 'seed'):
        h[k] = int(h[k])
    h['epochs'] = float(h['epochs'])
    if h['param_cap']:                                       # over the cap: pruned before any work (the search's bound)
        stored = param_counts(build(h)[0])['stored']
        if stored > h['param_cap']:
            return {'val_acc': 0.0, 'pruned': True, 'progress': 0, 'over_cap': True, 'params': stored,
                    'hyperparameters': {**h, 'widths': list(h['widths'])}}
    torch.manual_seed(h['seed']); np.random.seed(h['seed'])
    dev = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
    torch.backends.cudnn.benchmark = True
    dtype = torch.float16 if h['precision'] == 'fp16' and dev.type == 'cuda' else torch.float32
    # 'amp': fp32 weights (the optimiser's state too), the arithmetic under bf16 autocast: for AdamW and for networks
    # whose fp16 activations overflow; 'fp16' is airbench's own (fp16 weights, batch norm fp32); 'fp32' neither
    import contextlib
    amp = (lambda: torch.autocast(dev.type, dtype=torch.bfloat16)) if h['precision'] == 'amp' else contextlib.nullcontext

    arrays = dataset.load(data)
    gpu = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    mean, std = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1), torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    images = lambda a: ((gpu(a).permute(0, 3, 1, 2).to(dtype) / 255 - mean.to(dtype)) / std.to(dtype)).contiguous(memory_format=torch.channels_last)
    x_train, y_train = images(arrays['x_train']), gpu(arrays['y_train']).long()
    x_val, y_val = images(arrays['x_val']), gpu(arrays['y_val']).long()
    x_test, y_test = images(arrays['x_test']), gpu(arrays['y_test']).long()
    if h['train_on'] == 'train+val':
        x_train, y_train = torch.cat([x_train, x_val]), torch.cat([y_train, y_val])
    elif h['train_on'] != 'train':
        raise ValueError(f"train_on: 'train' or 'train+val', not {h['train_on']!r}")

    net, whiten = build(h)
    biases, others = _groups(net, whiten, h['bn_scale'])
    counts = param_counts(net)
    net = net.to(dev).to(dtype).to(memory_format=torch.channels_last)
    for m in net.modules():
        if isinstance(m, nn_batchnorm()):
            m.float()
    proxy, proxy_info = None, {}
    if h['filter_keep'] < 1:
        # D1, data filtering (airbench96_faster's idea, with a proxy trained first and then fixed): a small network
        # scores each batch by its loss, and the student trains on the hardest ``filter_keep`` share only
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pr = train(data, save=tmp, device=device, widths=h['filter_proxy_widths'], depth=2, scaling_factor=1 / 9,
                       epochs=h['filter_proxy_epochs'] or h['epochs'], batch_size=h['batch_size'], tta_level=0,
                       test='sealed', seed=h['seed'] + 10_000)
            proxy, _ = load(tmp, device=dev)
        proxy = proxy.to(dtype).to(memory_format=torch.channels_last).eval()
        for m in proxy.modules():
            if isinstance(m, nn_batchnorm()):
                m.float()
        proxy_info = {'proxy': {'params': pr['params'], 'val_acc_notta': pr['val_acc_notta'], 'epochs': pr['epochs_charged']}}
    teachers = []
    if h['kd_teacher'] and h['kd_weight'] > 0:
        # one directory, or several separated by commas: an ensemble, its softened predictions averaged
        for d in str(h['kd_teacher']).split(','):
            t, _ = load(d.strip(), device=dev)
            # (the student's precision: on a GPU at fp16 the batches are fp16, which fp32 weights refuse)
            t = t.to(dtype).to(memory_format=torch.channels_last).eval()
            for m in t.modules():
                if isinstance(m, nn_batchnorm()):
                    m.float()
            teachers.append(t)

    bs, n = h['batch_size'], len(y_train)
    steps_per_epoch = n // bs
    total = math.ceil(steps_per_epoch * h['epochs'])
    kilostep = 1024 * (1 + 1 / (1 - h['momentum']))
    lr, wd = h['lr'] / kilostep, h['weight_decay'] * bs / kilostep
    lr_b = lr * h['bias_scaler']
    if h['optimizer'] == 'sgd':
        opt = torch.optim.SGD([dict(params=biases, lr=lr_b, weight_decay=wd / lr_b),
                               dict(params=others, lr=lr, weight_decay=wd / lr)], momentum=h['momentum'], nesterov=True)
    elif h['optimizer'] == 'adamw':                          # (lr and weight_decay as AdamW's own; the loss a batch mean)
        opt = torch.optim.AdamW([dict(params=biases, lr=h['lr'], weight_decay=0.0),
                                 dict(params=others, lr=h['lr'], weight_decay=h['weight_decay'])],
                                betas=(h['momentum'], h['beta2']), eps=1e-6 if dtype == torch.float16 else 1e-8)
    elif h['optimizer'] == 'muon':                           # (O4: Muon for the filters, the SGD above for the rest)
        from .methods import MuonSGD
        filters = [p for p in others if p.dim() == 4]
        rest = [p for p in others if p.dim() != 4]
        opt = MuonSGD([dict(params=biases, lr=lr_b, weight_decay=wd / lr_b),
                       dict(params=rest, lr=lr, weight_decay=wd / lr),
                       dict(params=filters, lr=h['muon_lr'], muon=True)], momentum=h['momentum'],
                      muon_momentum=h['muon_momentum'])
    else:
        raise ValueError(f"optimizer: 'sgd', 'adamw' or 'muon', not {h['optimizer']!r}")
    warm = int(total * h['warmup_frac'])

    def lr_factor(step):
        if step < warm:
            f = step / warm
            return 0.2 * (1 - f) + f
        f = (step - warm) / max(1, total - warm)
        return (1 - f) + h['final_lr_frac'] * f
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    alpha = 0.95 ** 5 * (torch.arange(total + 1) / total) ** 3
    ahead = {k: v.clone() for k, v in net.state_dict().items()} if h['lookahead'] else None

    def lookahead(decay):
        for (k, a), v in zip(ahead.items(), net.state_dict().values()):
            if v.dtype in (torch.half, torch.float, torch.bfloat16):
                a.lerp_(v, 1 - decay)
                v.copy_(a)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=h['label_smoothing'], reduction='none')

    def accuracy(x, y, level, gate: float = 0.0):
        """(accuracy at ``level``'s views, the share of images given them: 1 unless MT6's ``gate`` is set)."""
        from .methods import gated_logits
        net.eval()
        with torch.no_grad(), amp():
            if gate > 0 and level > 0:
                parts = [gated_logits(net, xb, level, gate, tta_logits) for xb in x.split(2000)]
                logits = torch.cat([p_[0] for p_ in parts])
                share = sum(p_[1] * len(xb) for p_, xb in zip(parts, x.split(2000))) / len(x)
            else:
                logits, share = torch.cat([tta_logits(net, xb, level) for xb in x.split(2000)]), 1.0 if level else 0.0
        net.train()
        return (logits.argmax(1) == y).float().mean().item(), share

    if report is None:
        stop_here = lambda step, value: False
    elif callable(report):
        stop_here = report
    else:
        from solve_queue.hpo import reporter
        stop_here = reporter(report)

    if dev.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    if whiten is not None:
        whiten_init(whiten, x_train[:5000], h['whiten_eps'])
    if h['pca_init']:                                        # (MT3)
        from .methods import pca_init
        pca_init(net, x_train[:2048])
    # alternating flip: each image flipped at random once, then all together every other epoch; translation from
    # reflect padding done once
    base = torch.where((torch.rand(n, device=dev) < 0.5).view(-1, 1, 1, 1), x_train.flip(-1), x_train) if h['flip'] else x_train
    padded = F.pad(base, (h['translate'],) * 4, 'reflect') if h['translate'] > 0 else None
    st = {'step': 0, 'history': [], 'pruned': False}

    def batch_loss(xb, yb):
        if proxy is not None:
            with torch.no_grad():
                hard = loss_fn(proxy(xb), yb).float()
            keep = hard.topk(max(1, int(round(h['filter_keep'] * len(yb))))).indices
            xb, yb = xb[keep], yb[keep]
        with amp():
            out = net(xb)
        per = loss_fn(out.float() if h['precision'] == 'amp' else out, yb)     # (fp16: airbench's own, in fp16)
        if teachers:
            with torch.no_grad(), amp():
                soft = sum(F.softmax(t(xb).float() / h['kd_temp'], 1) for t in teachers) / len(teachers)
            logp = F.log_softmax(out.float() / h['kd_temp'], 1)
            kl = (soft * (soft.clamp_min(1e-8).log() - logp)).sum(1) * h['kd_temp'] ** 2
            per = (1 - h['kd_weight']) * per.float() + h['kd_weight'] * kl
            if h['kd_weighting'] != 'none':                  # (MT4)
                from .methods import kd_weights
                per = per * kd_weights(kl, h['kd_weighting'])
        return per.sum() if h['optimizer'] in ('sgd', 'muon') else per.mean()

    def run(epochs):
        """Train over ``epochs`` (a range of epoch indices) of the schedule; False when the run should stop."""
        for epoch in epochs:
            if whiten is not None and whiten.bias is not None:
                whiten.bias.requires_grad = epoch < h['whiten_bias_epochs']
            ep = _crop(padded, 32) if padded is not None else base
            if h['augment'] != 'none':
                from .augment import apply as augment_policy
                ep = augment_policy(ep, h['augment'], h['augment_m'], mean, std)
            if h['flip'] and epoch % 2 == 1:
                ep = ep.flip(-1)
            if h['cutout'] > 0:
                ep = _cutout(ep, h['cutout'])
            perm = torch.randperm(n, device=dev)
            net.train()
            loss_sum = torch.zeros((), device=dev)
            for i in range(steps_per_epoch):
                idx = perm[i * bs:(i + 1) * bs]
                loss = batch_loss(ep[idx], y_train[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                st['step'] += 1
                loss_sum += loss.detach()
                if ahead is not None and st['step'] % 5 == 0:
                    lookahead(alpha[st['step']].item())
                if st['step'] >= total:
                    if ahead is not None:
                        lookahead(1.0)
                    break
            mean_loss = (loss_sum / max(1, i + 1) / (bs if h['optimizer'] in ('sgd', 'muon') else 1)).item()
            st['history'].append({'epoch': epoch + 1, 'loss': round(mean_loss, 4),
                                  'val_acc': round(accuracy(x_val, y_val, 0)[0], 4), 'seconds': round(time.time() - t0, 1)})
            # (a line an epoch: the job's log grows while it trains, so the queue's stall check means something)
            print(f"epoch {epoch + 1}/{n_epochs}: loss {mean_loss:.4f} val {st['history'][-1]['val_acc']:.4f} "
                  f"{st['history'][-1]['seconds']:.0f} s", flush=True)
            if not math.isfinite(mean_loss):
                return False
            if st['step'] < total and stop_here(round(100 * st['step'] / total), st['history'][-1]['val_acc']):
                st['pruned'] = True
                return False
        return True

    n_epochs = math.ceil(h['epochs'])
    if h['tails'] > 0:
        # MT2: a trunk over the schedule's first epochs, then ``tails`` continuations over its last ``tail_epochs``,
        # each from the trunk's state with its own augmentation draws, their weights averaged and batch norm's
        # statistics re-estimated; every tail's epochs count against the budget
        import copy
        split = n_epochs - int(h['tail_epochs'])
        ok = run(range(split))
        trunk = (copy.deepcopy(net.state_dict()), copy.deepcopy(opt.state_dict()), copy.deepcopy(sched.state_dict()),
                 copy.deepcopy(ahead), st['step'])
        finals = []
        for k in range(int(h['tails']) if ok else 0):
            net.load_state_dict(trunk[0]); opt.load_state_dict(trunk[1]); sched.load_state_dict(trunk[2])
            if ahead is not None:
                ahead.clear(); ahead.update(copy.deepcopy(trunk[3]))
            st['step'] = trunk[4]
            torch.manual_seed(h['seed'] * 1000 + k + 1)
            if not run(range(split, n_epochs)):
                break
            finals.append(copy.deepcopy(net.state_dict()))
        if finals:
            avg = {key: (sum(f[key].float() for f in finals) / len(finals)).to(v.dtype) if v.dtype.is_floating_point else v
                   for key, v in finals[0].items()}
            net.load_state_dict(avg)
            bns = [m for m in net.modules() if isinstance(m, nn_batchnorm())]
            for m in bns:
                m.reset_running_stats(); m.momentum = None    # (a cumulative average over the passes below)
            with torch.no_grad():
                net.train()
                for i in range(0, min(n, 20 * bs), bs):
                    net(base[i:i + bs])
            for m in bns:
                m.momentum = h['bn_momentum']
            st['history'].append({'epoch': 'tails-averaged', 'val_acc': round(accuracy(x_val, y_val, 0)[0], 4),
                                  'seconds': round(time.time() - t0, 1)})
    else:
        run(range(n_epochs))
    step, history, pruned = st['step'], st['history'], st['pruned']
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    seconds = time.time() - t0
    name = torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'
    charged = h['epochs'] + (max(0, int(h['tails']) - 1) * int(h['tail_epochs']) if h['tails'] else 0)
    charged *= h['filter_keep']                              # (filtering: the images back-propagated, PLAN §2.1)
    common = {'train_seconds': round(seconds, 1), 'images_per_s': round(step * bs / seconds), 'device': name,
              'params': counts['stored'], 'params_trainable': counts['trainable'], 'steps': step,
              'history': history, 'hyperparameters': {**h, 'widths': list(h['widths'])}, 'epochs_charged': charged,
              **proxy_info}
    if pruned:
        return {'val_acc': history[-1]['val_acc'], 'pruned': True, 'progress': round(100 * step / total), **common}
    val, val_share = accuracy(x_val, y_val, h['tta_level'], h['tta_gate'])
    views = {0: 1, 1: 2, 2: 6, 3: 18}[h['tta_level']]
    out = {'val_acc': round(val, 4), 'val_acc_notta': round(accuracy(x_val, y_val, 0)[0], 4),
           # (1, 2 and 6 views of the same network: the equivariance-vs-TTA comparisons, PLAN §5.1)
           'val_acc_by_views': {v: round(accuracy(x_val, y_val, lv)[0], 4) for lv, v in ((0, 1), (1, 2), (2, 6))}}
    if h['test'] == 'sealed':
        # (session 2026-10-30-equivariant-small: the test set is not evaluated during the session; final models only)
        share = val_share
    else:
        test, share = accuracy(x_test, y_test, h['tta_level'], h['tta_gate'])
        out.update(test_acc=round(test, 4), test_acc_notta=round(accuracy(x_test, y_test, 0)[0], 4))
    out.update(views_per_image=round(1 + views * share, 3) if h['tta_gate'] > 0 and h['tta_level'] else views, **common)
    if save:
        out['saved'] = str(save_model(net, h, save))
    return out


def save_model(net, h: dict, directory: str) -> Path:
    """The network's weights (``model.safetensors``, fp32) and the hyperparameters that rebuild it (``config.json``)."""
    from safetensors.torch import save_file
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().float().contiguous().cpu() for k, v in net.state_dict().items()}, str(d / 'model.safetensors'))
    (d / 'config.json').write_text(json.dumps({'family': 'airbench', **h, 'widths': list(h['widths']),
                                              'arch_kw': dict(h.get('arch_kw', {}))}, indent=1) + '\n')
    return d


def load(directory, device='cpu'):
    """(network in eval mode on ``device``, fp32, its hyperparameters) from ``save_model``'s directory."""
    import torch
    from safetensors.torch import load_file
    d = Path(directory)
    h = json.loads((d / 'config.json').read_text())
    net, _ = build(h)
    net.load_state_dict(load_file(str(d / 'model.safetensors')))
    return net.to(torch.device(device)).eval(), h


class Predictor:
    """The inference contract the footprint bench measures: ``predict(uint8 (n, 32, 32, 3)) -> probabilities (n, 10)``,
    normalisation and TTA included."""

    def __init__(self, directory, device='cpu', dtype=None, tta_level: int | None = None):
        import torch
        self.net, self.h = load(directory, device)
        self.device = torch.device(device)
        self.dtype = dtype or torch.float32
        if self.dtype != torch.float32:
            self.net = self.net.to(self.dtype)
            for m in self.net.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.float()
        self.tta = self.h['tta_level'] if tta_level is None else tta_level
        self.mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)

    def predict(self, images):
        import numpy as np
        import torch
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(images, dtype=np.uint8)).to(self.device).permute(0, 3, 1, 2).float() / 255
            x = ((x - self.mean) / self.std).to(self.dtype)
            if self.h.get('tta_gate', 0) > 0 and self.tta:      # (MT6: the full views only where the margin is small)
                from .methods import gated_logits
                logits = gated_logits(self.net, x, self.tta, self.h['tta_gate'], tta_logits)[0]
            else:
                logits = tta_logits(self.net, x, self.tta)
            return torch.softmax(logits.float(), 1).cpu().numpy()
