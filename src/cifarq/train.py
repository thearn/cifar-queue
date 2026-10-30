"""The model and its training, every choice a hyperparameter (``DEFAULTS``; 0 or 'none' turns a feature off):

- model: a ResNet-9-style CNN (the fast CIFAR-10 architecture of the DAWNBench entries) with a width multiplier,
  dropout before the classifier, a choice of activation (relu, gelu, silu, celu with its alpha), a choice of classifier
  head normalising its output (linear, cosine: normalised features and weights, bn: batch norm on the logits) with a
  scale on the logits,
  and optionally a fixed input whitening layer (27 filters from the eigenvectors of the training images' 3x3 patch
  covariance, as in Page's later DAWNBench versions);
- data: the dataset's 45000 training images and 5000 validation images (``data.dataset``: a fixed split), the
  validation set the one the tuning optimises; the test set (10000) is reported, never tuned on;
- augmentation, random per image and done on the GPU: an affine transform (translation in pixels, rotation in degrees,
  scale), a horizontal flip with a probability, colour jitter (brightness, contrast, saturation), cutout (a square of a
  size, with a probability), and mixup or cutmix (Beta(alpha) mixing, with a probability per batch);
- optimisation: SGD (momentum, Nesterov) or AdamW (betas), decoupled weight decay (not on biases or batch norm),
  gradient clipping, an exponential moving average of the weights (evaluated in place of the weights), label smoothing;
- learning rate: a linear warmup over a fraction of the steps from a fraction of the peak, then a decay (cosine,
  linear, exponential, or none) to a final fraction of the peak;
- evaluation: optionally test-time flip averaging (each image's logits averaged with its mirror's).

The model is in ``model``, the accuracy in ``evaluate``. The images live on the GPU as uint8; the arithmetic is bf16
under autocast. ``device='cpu'`` (or ``auto`` without a GPU) runs the same code on the CPU, for the tests. ``train(**hyperparameters)`` is what a
queue job calls (``solve_queue.function_job``); its return value, a dict of plain numbers, is the job's result.
"""
from __future__ import annotations

import contextlib
import math
import time

from . import evaluate
from .model import MEAN, STD, model, whitening_filters  # noqa: F401  (train.model etc.: the names before the split)

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


def train(data: str | None = None, save: str | None = None, report=None, device: str = 'auto', **hyperparameters) -> dict:
    """Train one model with ``DEFAULTS`` updated by ``hyperparameters`` (an unknown name is an error); its validation
    and test accuracy, time and throughput, and the per-epoch history. ``save``: a directory (relative: in the job's
    directory, which comes back with the job) to write the trained network to as a ``hub.ResNet9`` (the EMA's weights
    when there is one). ``report``: early stopping (``solve_queue.hpo.reporter``: a study's spec, or a callable
    ``(step, value) -> stop``): after each epoch the validation accuracy is reported at the percent of training done,
    and a trial told to stop returns at once with ``pruned: True`` (no test evaluation). ``device``: ``cuda``, ``cpu``
    or ``auto`` (the GPU when there is one); not a hyperparameter, so not in the result's ``hyperparameters``."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .data import dataset
    unknown = set(hyperparameters) - set(DEFAULTS)
    if unknown:
        raise TypeError(f'unknown hyperparameters: {sorted(unknown)}')
    h = {**DEFAULTS, **hyperparameters}
    for k in ('width', 'batch_size', 'epochs', 'cutout_size', 'seed'):
        h[k] = int(h[k])
    h['whiten'], h['tta_flip'] = bool(h['whiten']), bool(h['tta_flip'])

    torch.manual_seed(h['seed']); np.random.seed(h['seed'])
    dev = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
    torch.backends.cudnn.benchmark = True
    arrays = dataset.load(data)                              # (the split is the dataset's: the same for every trial)
    gpu = lambda a: torch.from_numpy(a).to(dev)
    x_train, y_train = gpu(arrays['x_train']).permute(0, 3, 1, 2).contiguous(), gpu(arrays['y_train']).long()
    x_val, y_val = gpu(arrays['x_val']).permute(0, 3, 1, 2).contiguous(), gpu(arrays['y_val']).long()
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
    steps_per_epoch = len(y_train) // h['batch_size']
    total = h['epochs'] * steps_per_epoch
    # schedule-free (Defazio et al., 2024; ``schedulefree``): no decay phase, the averaged weights good at any point, so
    # accuracy part-way through predicts the final (early stopping's signal); its own warm-up, no LambdaLR, no EMA
    schedule_free = h['optimizer'].endswith('_sf')
    if schedule_free:
        import schedulefree
        warm = max(1, int(h['warmup_frac'] * total))
        if h['optimizer'] == 'sgd_sf':
            opt = schedulefree.SGDScheduleFree(groups, lr=h['lr'], momentum=h['momentum'] or 0.9, warmup_steps=warm)
        else:
            opt = schedulefree.AdamWScheduleFree(groups, lr=h['lr'], betas=(h['beta1'], h['beta2']), warmup_steps=warm)
        sched = None
    else:
        if h['optimizer'] == 'sgd':
            opt = torch.optim.SGD(groups, lr=h['lr'], momentum=h['momentum'], nesterov=bool(h['nesterov']) and h['momentum'] > 0)
        else:
            opt = torch.optim.AdamW(groups, lr=h['lr'], betas=(h['beta1'], h['beta2']))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: lr_factor(step, total, h))
    ema = EMA(net, h['ema_decay']) if h['ema_decay'] and not schedule_free else None

    def averaged():
        """Schedule-free: switch to the averaged weights and refresh batch norm's statistics for them (a few training
        batches, no gradients), as the method's authors advise for networks with batch norm."""
        if not schedule_free:
            return
        opt.eval()
        net.train()
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
            for i in range(0, min(len(y_train), 25 * h['batch_size']), h['batch_size']):
                net(norm(x_train[i:i + h['batch_size']].float() / 255))

    def accuracy(x, y):
        return evaluate.accuracy(ema.module if ema is not None else net, x, y, norm, h['tta_flip'])

    try:                                                     # (taking turns at the card with other trials: opt-in)
        from solve_queue.gpu import device_turn
    except ImportError:
        device_turn = contextlib.nullcontext
    turn = device_turn(); turn.__enter__()
    if report is None:
        stop_here = lambda step, value: False
    elif callable(report):
        stop_here = report
    else:
        from solve_queue.hpo import reporter                 # a study's spec: only inside the queue's trials
        stop_here = reporter(report)
    pruned = False
    history, t0 = [], time.time()
    for epoch in range(h['epochs']):
        if schedule_free:
            opt.train()
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
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                out = net(norm(xb))
                loss = F.cross_entropy(out, yb, label_smoothing=h['label_smoothing'])
                if mixed:
                    loss = lam * loss + (1 - lam) * F.cross_entropy(out, yb[other], label_smoothing=h['label_smoothing'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if h['grad_clip']:
                torch.nn.utils.clip_grad_norm_(trainable, h['grad_clip'])
            opt.step()
            if sched is not None:
                sched.step()
            if ema is not None:
                ema.update_parameters(net)
            loss_sum += loss.detach()
        mean_loss = (loss_sum / steps_per_epoch).item()
        averaged()
        history.append({'epoch': epoch + 1, 'loss': round(mean_loss, 4), 'val_acc': round(accuracy(x_val, y_val), 4),
                        'lr': round(sched.get_last_lr()[0] if sched is not None else opt.param_groups[0]['lr'], 6),
                        'seconds': round(time.time() - t0, 1)})
        if not math.isfinite(mean_loss):                           # diverged: no use training on
            break
        if epoch + 1 < h['epochs'] and stop_here(round(100 * (epoch + 1) / h['epochs']), history[-1]['val_acc']):
            pruned = True                                         # the study's pruner: worse than its rung's best
            break
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    seconds = time.time() - t0
    test_acc = None if pruned else round(accuracy(x_test, y_test), 4)
    turn.__exit__(None, None, None)
    name = torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'
    if pruned:
        return {'val_acc': history[-1]['val_acc'], 'pruned': True, 'progress': round(100 * len(history) / h['epochs']),
                'epochs_run': len(history), 'train_seconds': round(seconds, 1),
                'images_per_s': round(len(history) * steps_per_epoch * h['batch_size'] / seconds),
                'device': name, 'history': history, 'hyperparameters': h}
    out = {'val_acc': history[-1]['val_acc'], 'test_acc': test_acc, 'train_seconds': round(seconds, 1),
           'images_per_s': round(len(history) * steps_per_epoch * h['batch_size'] / seconds),
           'params': sum(p.numel() for p in net.parameters()), 'device': name,
           'history': history, 'hyperparameters': h}
    if save:
        from .hub import save as save_model
        out['saved'] = str(save_model(ema.module if ema is not None else net, h, save))
    return out
