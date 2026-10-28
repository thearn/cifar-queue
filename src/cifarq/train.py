"""The model and its training, every choice a hyperparameter (``DEFAULTS``; 0 or 'none' turns a feature off):

- model: a ResNet-9-style CNN (the fast CIFAR-10 architecture of the DAWNBench entries) with a width multiplier,
  dropout before the classifier, a choice of activation (relu, gelu, silu, celu with its alpha), a choice of classifier
  head normalising its output (linear, cosine: normalised features and weights, bn: batch norm on the logits) with a
  scale on the logits,
  and optionally a fixed input whitening layer (27 filters from the eigenvectors of the training images' 3x3 patch
  covariance, as in Page's later DAWNBench versions);
- data: 45000 training images, 5000 held out (a fixed split) as the validation set the tuning optimises; the test set
  (10000) is reported, never tuned on;
- augmentation, random per image and done on the GPU: an affine transform (translation in pixels, rotation in degrees,
  scale), a horizontal flip with a probability, colour jitter (brightness, contrast, saturation), cutout (a square of a
  size, with a probability), and mixup or cutmix (Beta(alpha) mixing, with a probability per batch);
- optimisation: SGD (momentum, Nesterov) or AdamW (betas), decoupled weight decay (not on biases or batch norm),
  gradient clipping, an exponential moving average of the weights (evaluated in place of the weights), label smoothing;
- learning rate: a linear warmup over a fraction of the steps from a fraction of the peak, then a decay (cosine,
  linear, exponential, or none) to a final fraction of the peak;
- evaluation: optionally test-time flip averaging (each image's logits averaged with its mirror's).

The images live on the GPU as uint8; the arithmetic is bf16 under autocast. ``train(**hyperparameters)`` is what a
queue job calls (``solve_queue.function_job``); its return value, a dict of plain numbers, is the job's result.
"""
from __future__ import annotations

import math
import time

DEFAULTS = dict(
    # model
    width=32, dropout=0.1, activation='relu', celu_alpha=0.075, head='linear', logit_scale=1.0, whiten=False, whiten_eps=1e-2,
    # optimisation
    optimizer='sgd', lr=0.08, momentum=0.9, nesterov=True, beta1=0.9, beta2=0.999, weight_decay=5e-4,
    batch_size=512, epochs=10, label_smoothing=0.1, grad_clip=0.0, ema_decay=0.0,
    # learning rate schedule
    warmup_frac=0.25, warmup_start_frac=0.04, decay='cosine', final_lr_frac=0.01,
    # augmentation (0: off)
    aug_translate=4.0, aug_rotate=0.0, aug_scale=0.0, aug_flip=0.5,
    aug_brightness=0.0, aug_contrast=0.0, aug_saturation=0.0,
    cutout_size=0, cutout_prob=0.5, mix='none', mix_alpha=1.0, mix_prob=0.5,
    # evaluation
    tta_flip=False,
    seed=0)
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
VALIDATION = 5000


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


def lr_factor(step: int, total: int, h: dict) -> float:
    """The learning rate as a fraction of its peak at ``step`` of ``total``: warmup, then the decay."""
    warm = int(h['warmup_frac'] * total)
    if step < warm:
        return h['warmup_start_frac'] + (1 - h['warmup_start_frac']) * step / max(1, warm)
    t = (step - warm) / max(1, total - warm)
    end = h['final_lr_frac']
    return {'cosine': end + (1 - end) * 0.5 * (1 + math.cos(math.pi * t)), 'linear': 1 - (1 - end) * t,
            'exponential': end ** t, 'none': 1.0}[h['decay']]


def augment(x, h: dict):
    """A batch (float, 0-1, NCHW, on the GPU) augmented per image as ``h`` says; each part skipped when off."""
    import torch
    import torch.nn.functional as F
    n, dev = x.shape[0], x.device
    u = lambda: torch.rand(n, device=dev) * 2 - 1                  # uniform in [-1, 1], one per image
    if h['aug_brightness'] or h['aug_contrast'] or h['aug_saturation']:
        if h['aug_brightness']:
            x = x * (1 + h['aug_brightness'] * u()).view(-1, 1, 1, 1)
        if h['aug_contrast']:
            m = x.mean(dim=(1, 2, 3), keepdim=True)
            x = (x - m) * (1 + h['aug_contrast'] * u()).view(-1, 1, 1, 1) + m
        if h['aug_saturation']:
            g = (0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
            x = (x - g) * (1 + h['aug_saturation'] * u()).view(-1, 1, 1, 1) + g
        x = x.clamp(0, 1)
    if h['aug_translate'] or h['aug_rotate'] or h['aug_scale'] or h['aug_flip']:
        angle = h['aug_rotate'] * math.pi / 180 * u()
        scale = 1 + h['aug_scale'] * u()
        flip = torch.where(torch.rand(n, device=dev) < h['aug_flip'], -1.0, 1.0)
        shift = h['aug_translate'] * 2 / x.shape[-1]                  # pixels in normalised coordinates
        cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
        theta = torch.stack([torch.stack([flip * cos, -sin, shift * u()], 1), torch.stack([flip * sin, cos, shift * u()], 1)], 1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    if h['cutout_size']:
        s, size = x.shape[-1], int(h['cutout_size'])
        cy, cx = torch.randint(0, s, (n,), device=dev), torch.randint(0, s, (n,), device=dev)
        r = torch.arange(s, device=dev)
        box = ((r.view(1, s, 1) - cy.view(-1, 1, 1)).abs() <= size // 2) & ((r.view(1, 1, s) - cx.view(-1, 1, 1)).abs() <= size // 2)
        box &= (torch.rand(n, device=dev) < h['cutout_prob']).view(-1, 1, 1)
        x = x * (~box).unsqueeze(1)
    return x


class EMA:
    """An exponential moving average of a model's weights and batch-norm statistics, its decay warmed up
    (``min(decay, (1 + n) / (10 + n))`` at update n): without the warmup a short run's average is dominated by its
    first, untrained weights (0.13 validation accuracy at 0.999 over 350 steps, 2026-10-27)."""

    def __init__(self, net, decay: float):
        import copy
        self.module, self.decay, self.n = copy.deepcopy(net), decay, 0
        for p in self.module.parameters():
            p.requires_grad_(False)

    def update_parameters(self, net):
        import torch
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        self.n += 1
        with torch.no_grad():
            mine, theirs = self.module.state_dict(), net.state_dict()
            for k, v in mine.items():
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(theirs[k].detach(), alpha=1 - d)
                else:
                    v.copy_(theirs[k])


def train(data: str | None = None, save: str | None = None, **hyperparameters) -> dict:
    """Train one model with ``DEFAULTS`` updated by ``hyperparameters`` (an unknown name is an error); its validation
    and test accuracy, time and throughput, and the per-epoch history. ``save``: a directory (relative: in the job's
    directory, which comes back with the job) to write the trained network to as a ``hub.ResNet9`` (the EMA's weights
    when there is one)."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from . import data as D
    unknown = set(hyperparameters) - set(DEFAULTS)
    if unknown:
        raise TypeError(f'unknown hyperparameters: {sorted(unknown)}')
    h = {**DEFAULTS, **hyperparameters}
    for k in ('width', 'batch_size', 'epochs', 'cutout_size', 'seed'):
        h[k] = int(h[k])
    h['whiten'], h['tta_flip'] = bool(h['whiten']), bool(h['tta_flip'])

    torch.manual_seed(h['seed']); np.random.seed(h['seed'])
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    arrays = D.load(data)
    order = np.random.default_rng(1234).permutation(len(arrays['y_train']))   # the same split for every trial
    val_idx, tr_idx = order[:VALIDATION], order[VALIDATION:]
    gpu = lambda a: torch.from_numpy(a).to(dev)
    x_train, y_train = gpu(arrays['x_train'][tr_idx]).permute(0, 3, 1, 2).contiguous(), gpu(arrays['y_train'][tr_idx]).long()
    x_val, y_val = gpu(arrays['x_train'][val_idx]).permute(0, 3, 1, 2).contiguous(), gpu(arrays['y_train'][val_idx]).long()
    x_test, y_test = gpu(arrays['x_test']).permute(0, 3, 1, 2).contiguous(), gpu(arrays['y_test']).long()
    mean, std = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1), torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    norm = lambda x: ((x - mean) / std).contiguous(memory_format=torch.channels_last)

    whitening = None
    if h['whiten']:                                          # from 5000 training images (the validation images unseen)
        whitening = whitening_filters(norm(x_train[:5000].float() / 255), h['whiten_eps'])
    net = model(h['width'], h['dropout'], h['activation'], celu_alpha=h['celu_alpha'], head=h['head'], logit_scale=h['logit_scale'],
                whitening=whitening).to(dev).to(memory_format=torch.channels_last)
    trainable = [p for p in net.parameters() if p.requires_grad]
    groups = [{'params': [p for p in trainable if p.ndim > 1], 'weight_decay': h['weight_decay']},
              {'params': [p for p in trainable if p.ndim <= 1], 'weight_decay': 0.0}]
    if h['optimizer'] == 'sgd':
        opt = torch.optim.SGD(groups, lr=h['lr'], momentum=h['momentum'], nesterov=bool(h['nesterov']) and h['momentum'] > 0)
    else:
        opt = torch.optim.AdamW(groups, lr=h['lr'], betas=(h['beta1'], h['beta2']))
    steps_per_epoch = len(y_train) // h['batch_size']
    total = h['epochs'] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: lr_factor(step, total, h))
    ema = EMA(net, h['ema_decay']) if h['ema_decay'] else None

    def accuracy(x, y):
        m = ema.module if ema is not None else net
        m.eval()
        correct = 0
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i in range(0, len(y), 1000):
                xb = norm(x[i:i + 1000].float() / 255)
                logits = m(xb)
                if h['tta_flip']:                            # the mirror's logits too
                    logits = logits + m(xb.flip(3))
                correct += (logits.argmax(1) == y[i:i + 1000]).sum().item()
        m.train()
        return correct / len(y)

    history, t0 = [], time.time()
    for epoch in range(h['epochs']):
        perm = torch.randperm(len(y_train), device=dev)
        loss_sum = torch.zeros((), device=dev)
        for s in range(steps_per_epoch):
            idx = perm[s * h['batch_size']:(s + 1) * h['batch_size']]
            xb, yb = augment(x_train[idx].float() / 255, h), y_train[idx]
            mixed = h['mix'] != 'none' and np.random.rand() < h['mix_prob']
            if mixed:
                lam = float(np.random.beta(h['mix_alpha'], h['mix_alpha']))
                other = torch.randperm(len(idx), device=dev)
                if h['mix'] == 'mixup':
                    xb = lam * xb + (1 - lam) * xb[other]
                else:                                               # cutmix: a box of area 1 - lam from another image
                    side = int(32 * math.sqrt(1 - lam))
                    y0, x0 = np.random.randint(0, 33 - side, 2) if side else (0, 0)
                    xb = xb.clone(); xb[:, :, y0:y0 + side, x0:x0 + side] = xb[other, :, y0:y0 + side, x0:x0 + side]
                    lam = 1 - side * side / 1024
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = net(norm(xb))
                loss = F.cross_entropy(out, yb, label_smoothing=h['label_smoothing'])
                if mixed:
                    loss = lam * loss + (1 - lam) * F.cross_entropy(out, yb[other], label_smoothing=h['label_smoothing'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if h['grad_clip']:
                torch.nn.utils.clip_grad_norm_(trainable, h['grad_clip'])
            opt.step()
            sched.step()
            if ema is not None:
                ema.update_parameters(net)
            loss_sum += loss.detach()
        mean_loss = (loss_sum / steps_per_epoch).item()
        history.append({'epoch': epoch + 1, 'loss': round(mean_loss, 4), 'val_acc': round(accuracy(x_val, y_val), 4),
                        'lr': round(sched.get_last_lr()[0], 6), 'seconds': round(time.time() - t0, 1)})
        if not math.isfinite(mean_loss):                           # diverged: no use training on
            break
    torch.cuda.synchronize()
    seconds = time.time() - t0
    out = {'val_acc': history[-1]['val_acc'], 'test_acc': round(accuracy(x_test, y_test), 4), 'train_seconds': round(seconds, 1),
           'images_per_s': round(len(history) * steps_per_epoch * h['batch_size'] / seconds),
           'params': sum(p.numel() for p in net.parameters()), 'device': torch.cuda.get_device_name(0),
           'history': history, 'hyperparameters': h}
    if save:
        from .hub import save as save_model
        out['saved'] = str(save_model(ema.module if ema is not None else net, h, save))
    return out
