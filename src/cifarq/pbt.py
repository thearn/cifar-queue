"""Population-based training as a second stage after the search (docs/PLAN-mode2.md step 7), and the probes that went
with it on the G4: the best configuration over seeds, batch sizes, and one against two trainings at once.

PBT (Ray Tune's ``PopulationBasedTraining``): a population trains at once; every few epochs the weaker members copy a
stronger member's weights and settings ("exploit") and perturb the settings ("explore"). The architecture is fixed
(members copy each other's weights): what moves are the learning rate, weight decay, label smoothing and augmentation
strengths, over the run. ``Trainer`` is ``train.train`` taken apart so a run can stop and resume at an epoch boundary
(Ray's checkpoints); ``train.train`` itself is unchanged.
"""
from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path

from .train import DEFAULTS, MEAN, STD, augment, model, whitening_filters

# what PBT may change during a run, and the ranges it resamples from
MUTABLE = {'lr': (1e-4, 3e-2), 'weight_decay': (1e-3, 0.5), 'label_smoothing': (0.0, 0.2),
           'aug_translate': (0.0, 5.0), 'aug_rotate': (0.0, 15.0), 'aug_scale': (0.0, 0.3), 'aug_contrast': (0.0, 0.4)}


class Trainer:
    """One training run, an epoch at a time: ``epoch()`` trains one, ``accuracy('val'|'test')``, ``state()`` and
    ``load(state)`` for a checkpoint, ``set(**hparams)`` for the settings PBT moves. Schedule-free optimisers only
    (no fixed horizon to break when the settings change mid-run)."""

    def __init__(self, data: str | None, **hyperparameters):
        import numpy as np
        import torch
        import schedulefree
        from .data import dataset
        h = {**DEFAULTS, **hyperparameters}
        for k in ('width', 'batch_size', 'epochs', 'seed'):
            h[k] = int(h[k])
        if not h['optimizer'].endswith('_sf'):
            raise ValueError('PBT here takes a schedule-free optimiser (sgd_sf, adamw_sf)')
        self.h, self.torch = h, torch
        torch.manual_seed(h['seed']); np.random.seed(h['seed'])
        dev = self.dev = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        arrays = dataset.load(data)
        gpu = lambda a: torch.from_numpy(a).to(dev)
        self.x_train, self.y_train = gpu(arrays['x_train']).permute(0, 3, 1, 2).contiguous(), gpu(arrays['y_train']).long()
        self.x = {s: gpu(arrays[f'x_{s}']).permute(0, 3, 1, 2).contiguous() for s in ('val', 'test')}
        self.y = {s: gpu(arrays[f'y_{s}']).long() for s in ('val', 'test')}
        mean, std = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1), torch.tensor(STD, device=dev).view(1, 3, 1, 1)
        self.norm = lambda x: ((x - mean) / std).contiguous(memory_format=torch.channels_last)
        whitening = whitening_filters(self.norm(self.x_train[:5000].float() / 255), h['whiten_eps']) if h['whiten'] else None
        self.net = model(h['width'], h['dropout'], h['activation'], celu_alpha=h['celu_alpha'], head=h['head'],
                         logit_scale=h['logit_scale'], whitening=whitening).to(dev).to(memory_format=torch.channels_last)
        trainable = [p for p in self.net.parameters() if p.requires_grad]
        groups = [{'params': [p for p in trainable if p.ndim > 1], 'weight_decay': h['weight_decay']},
                  {'params': [p for p in trainable if p.ndim <= 1], 'weight_decay': 0.0}]
        self.steps_per_epoch = len(self.y_train) // h['batch_size']
        warm = max(1, int(h['warmup_frac'] * h['epochs'] * self.steps_per_epoch))
        if h['optimizer'] == 'sgd_sf':
            self.opt = schedulefree.SGDScheduleFree(groups, lr=h['lr'], momentum=h['momentum'] or 0.9, warmup_steps=warm)
        else:
            self.opt = schedulefree.AdamWScheduleFree(groups, lr=h['lr'], betas=(h['beta1'], h['beta2']), warmup_steps=warm)
        self.done = 0

    def set(self, **hp):
        """The settings PBT moves, applied from the next step: learning rate, weight decay, label smoothing, augmentation."""
        for k, v in hp.items():
            if k not in MUTABLE:
                raise TypeError(f'{k} is not a setting PBT moves here')
            self.h[k] = float(v)
        for g in self.opt.param_groups:
            g['lr'] = self.h['lr']
            if g['weight_decay']:
                g['weight_decay'] = self.h['weight_decay']

    def epoch(self) -> float:
        import torch.nn.functional as F
        torch, h = self.torch, self.h
        self.opt.train(); self.net.train()
        perm = torch.randperm(len(self.y_train), device=self.dev)
        loss_sum = torch.zeros((), device=self.dev)
        for s in range(self.steps_per_epoch):
            idx = perm[s * h['batch_size']:(s + 1) * h['batch_size']]
            xb, yb = augment(self.x_train[idx].float() / 255, h), self.y_train[idx]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = F.cross_entropy(self.net(self.norm(xb)), yb, label_smoothing=h['label_smoothing'])
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            loss_sum += loss.detach()
        self.done += 1
        return (loss_sum / self.steps_per_epoch).item()

    def accuracy(self, split: str) -> float:
        """On the averaged weights, batch norm's statistics refreshed for them (as ``train`` does)."""
        torch, h = self.torch, self.h
        self.opt.eval(); self.net.train()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i in range(0, min(len(self.y_train), 25 * h['batch_size']), h['batch_size']):
                self.net(self.norm(self.x_train[i:i + h['batch_size']].float() / 255))
        self.net.eval()
        correct = 0
        x, y = self.x[split], self.y[split]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i in range(0, len(y), 1000):
                xb = self.norm(x[i:i + 1000].float() / 255)
                logits = self.net(xb)
                if h['tta_flip']:
                    logits = logits + self.net(xb.flip(3))
                correct += (logits.argmax(1) == y[i:i + 1000]).sum().item()
        return correct / len(y)

    def state(self) -> dict:
        return {'net': self.net.state_dict(), 'opt': self.opt.state_dict(), 'done': self.done,
                'h': {k: self.h[k] for k in MUTABLE}}

    def load(self, state: dict):
        self.net.load_state_dict(state['net']); self.opt.load_state_dict(state['opt']); self.done = state['done']


def _trainable(config: dict):
    """Ray Tune's PBT trial: one ``Trainer``, an epoch per report, a checkpoint each report (Ray restores another
    member's when it exploits); the settings from ``config`` applied at every (re)start."""
    import torch
    from ray import tune as rtune
    from ray.tune import Checkpoint
    config = dict(config)
    data, base = config.pop('data'), config.pop('base')
    t = Trainer(data, **{**base, **{k: config[k] for k in MUTABLE}})
    ck = rtune.get_checkpoint()
    if ck:
        with ck.as_directory() as d:
            t.load(torch.load(Path(d) / 'state.pt', weights_only=False))
    t.set(**{k: config[k] for k in MUTABLE})
    while t.done < t.h['epochs']:
        loss = t.epoch()
        val = t.accuracy('val')
        metrics = {'val_acc': val, 'loss': loss, 'epoch': t.done}
        if t.done == t.h['epochs']:
            metrics['test_acc'] = t.accuracy('test')
        with tempfile.TemporaryDirectory() as d:
            torch.save(t.state(), Path(d) / 'state.pt')
            rtune.report(metrics, checkpoint=Checkpoint.from_directory(d))


def run_pbt(base: dict, data: str | None = None, population: int = 8, concurrent: int = 4, interval: int = 3,
            seed: int = 0) -> dict:
    """PBT over ``base`` (a configuration in ``train``'s names, schedule-free): ``population`` members, ``concurrent``
    at once on the GPU, the settings in ``MUTABLE`` explored every ``interval`` epochs; members start from ``base``'s
    values perturbed. The best member's validation and test accuracy, every member's final, the settings' history."""
    import random
    import ray
    from ray import tune as rtune
    from ray.tune.schedulers import PopulationBasedTraining
    t0 = time.time()
    rng = random.Random(seed)
    ray.init(include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)

    def start(k):
        lo, hi = MUTABLE[k]
        v = base.get(k, (lo + hi) / 2) * rng.uniform(0.7, 1.4)
        return min(hi, max(lo, v))
    space = {k: rtune.sample_from(lambda spec, k=k: start(k)) for k in MUTABLE}
    log_uniform = lambda lo, hi: (lambda: math.exp(rng.uniform(math.log(lo), math.log(hi))))
    mutations = {k: (log_uniform(max(lo, 1e-5), hi) if k in ('lr', 'weight_decay') else (lambda lo=lo, hi=hi: rng.uniform(lo, hi)))
                 for k, (lo, hi) in MUTABLE.items()}
    pbt = PopulationBasedTraining(time_attr='training_iteration', perturbation_interval=interval,
                                  hyperparam_mutations=mutations, quantile_fraction=0.25, resample_probability=0.25)
    tuner = rtune.Tuner(
        rtune.with_resources(_trainable, {'cpu': 1, 'gpu': 1 / concurrent}),
        param_space={'data': data, 'base': base, **space},
        tune_config=rtune.TuneConfig(metric='val_acc', mode='max', scheduler=pbt, num_samples=population,
                                     max_concurrent_trials=concurrent),
        run_config=rtune.RunConfig(storage_path=str(Path.cwd() / 'ray'), name='pbt', verbose=0,
                                   stop={'training_iteration': int(base['epochs'])}))
    results = tuner.fit()
    rows = [r for r in results if r.metrics]
    final = [{'val_acc': r.metrics.get('val_acc'), 'test_acc': r.metrics.get('test_acc'),
              'settings': {k: r.config.get(k) for k in MUTABLE}} for r in rows]
    best = max(final, key=lambda f: f['val_acc'] or 0)
    out = {'population': population, 'concurrent': concurrent, 'interval': interval, 'best': best, 'members': final,
           'seconds': round(time.time() - t0, 1)}
    Path('pbt.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    print(json.dumps({'best': best, 'seconds': out['seconds']}, default=str), flush=True)
    ray.shutdown()
    return out


def probe(base: dict, data: str | None = None, seeds=(0, 1, 2), batches=(256, 512, 1024, 2048, 4096)) -> dict:
    """On one card, for the configuration ``base`` (``train``'s names): its accuracy over ``seeds`` (the baseline PBT is
    measured against), each batch size with the learning rate scaled by the square root of the batch ratio (Adam's
    usual rule), and the throughput of one training against two at once (``parallel.GPUProcessExecutor``)."""
    from .train import train
    from solve_queue.parallel import GPUProcessExecutor
    out = {'seeds': [], 'batches': [], 'concurrency': []}
    for s in seeds:
        r = train(data=data, **{**base, 'seed': s})
        out['seeds'].append({'seed': s, 'val_acc': r['val_acc'], 'test_acc': r['test_acc'], 'train_seconds': r['train_seconds']})
        print('seed', out['seeds'][-1], flush=True)
    b0 = int(base['batch_size'])
    for b in batches:
        r = train(data=data, **{**base, 'batch_size': b, 'lr': base['lr'] * math.sqrt(b / b0)})
        out['batches'].append({'batch_size': b, 'val_acc': r['val_acc'], 'test_acc': r['test_acc'],
                               'train_seconds': r['train_seconds'], 'images_per_s': r['images_per_s']})
        print('batch', out['batches'][-1], flush=True)
    for k in (1, 2):
        with GPUProcessExecutor(k) as ex:
            ex.submit('cifarq.train:train', data=data, **{**base, 'epochs': 2}).result()     # (warm: start, imports)
            t0 = time.time()
            futures = [ex.submit('cifarq.train:train', data=data, **base) for _ in range(k)]
            rs = [f.result() for f in futures]
        wall = time.time() - t0
        out['concurrency'].append({'at_once': k, 'wall_seconds': round(wall, 1), 'trainings_per_hour': round(3600 * k / wall, 1),
                                   'val_acc': [r['val_acc'] for r in rs]})
        print('concurrency', out['concurrency'][-1], flush=True)
    Path('probe.json').write_text(json.dumps(out, indent=1) + '\n')
    return out
