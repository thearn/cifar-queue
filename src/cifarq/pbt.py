"""Population-based training as a second stage after the search (docs/PLAN-mode2.md step 7), and the probes that went
with it on the G4: the best configuration over seeds, batch sizes, and one against two trainings at once.

PBT (Ray Tune's ``PopulationBasedTraining``): a population trains at once; every few epochs the weaker members copy a
stronger member's weights and settings ("exploit") and perturb the settings ("explore"). The architecture is fixed
(members copy each other's weights): what moves are the learning rate, weight decay, label smoothing and augmentation
strengths, over the run. ``Trainer`` is ``train.Run`` (what ``train.train`` loops over) at one seed, so a run can stop
and resume at an epoch boundary (Ray's checkpoints).
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .train import DEFAULTS, Run

if TYPE_CHECKING:
    from collections.abc import Iterable

# what PBT may change during a run, and the ranges it resamples from
MUTABLE = {
    'lr': (1e-4, 3e-2),
    'weight_decay': (1e-3, 0.5),
    'label_smoothing': (0.0, 0.2),
    'aug_translate': (0.0, 5.0),
    'aug_rotate': (0.0, 15.0),
    'aug_scale': (0.0, 0.3),
    'aug_contrast': (0.0, 0.4),
}


class Trainer:
    """One training run, an epoch at a time: ``epoch()`` trains one, ``accuracy('val'|'test')``, ``state()`` and
    ``load(state)`` for a checkpoint, ``set(**hparams)`` for the settings PBT moves. Schedule-free optimisers only
    (no fixed horizon to break when the settings change mid-run). A one-seed ``train.Run``: the settings PBT moves are
    traced arguments of its compiled epoch, so a move costs no compilation."""

    def __init__(self, data: str | None, device: str = 'auto', **hyperparameters) -> None:
        h = {**DEFAULTS, **hyperparameters}
        for k in ('width', 'batch_size', 'epochs', 'seed', 'cutout_size'):
            h[k] = int(h[k])
        if not h['optimizer'].endswith('_sf'):
            raise ValueError('PBT here takes a schedule-free optimiser (sgd_sf, adamw_sf)')
        self.run = Run(h, [h['seed']], data, device)
        self.h = self.run.h

    @property
    def done(self) -> int:
        return self.run.done

    def set(self, **hp) -> None:
        """The settings PBT moves, applied from the next step: learning rate, weight decay, label smoothing, augmentation."""
        for k in hp:
            if k not in MUTABLE:
                raise TypeError(f'{k} is not a setting PBT moves here')
        self.run.set(**hp)

    def epoch(self) -> float:
        return float(self.run.epoch()[0])

    def accuracy(self, split: str) -> float:
        """On the averaged weights, batch norm's statistics refreshed for them (as ``train`` does)."""
        return self.run.accuracies(split)[0]

    def state(self) -> dict:
        return {**self.run.state(), 'h': {k: self.h[k] for k in MUTABLE}}

    def load(self, state: dict) -> None:
        self.run.load(state)


def _trainable(config: dict) -> None:
    """Ray Tune's PBT trial: one ``Trainer``, an epoch per report, a checkpoint each report (Ray restores another
    member's when it exploits); the settings from ``config`` applied at every (re)start."""
    import pickle

    from ray import tune as rtune
    from ray.tune import Checkpoint

    config = dict(config)
    data, base = config.pop('data'), config.pop('base')
    t = Trainer(data, **{**base, **{k: config[k] for k in MUTABLE}})
    ck = rtune.get_checkpoint()
    if ck:
        with ck.as_directory() as d:
            t.load(pickle.loads((Path(d) / 'state.pkl').read_bytes()))
    t.set(**{k: config[k] for k in MUTABLE})
    while t.done < t.h['epochs']:
        loss = t.epoch()
        val = t.accuracy('val')
        metrics = {'val_acc': val, 'loss': loss, 'epoch': t.done}
        if t.done == t.h['epochs']:
            metrics['test_acc'] = t.accuracy('test')
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'state.pkl').write_bytes(pickle.dumps(t.state()))
            rtune.report(metrics, checkpoint=Checkpoint.from_directory(d))


def run_pbt(base: dict, data: str | None = None, population: int = 8, concurrent: int = 4, interval: int = 3, seed: int = 0) -> dict:
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

    def start(k: str) -> float:
        lo, hi = MUTABLE[k]
        v = base.get(k, (lo + hi) / 2) * rng.uniform(0.7, 1.4)
        return min(hi, max(lo, v))

    space = {k: rtune.sample_from(lambda spec, k=k: start(k)) for k in MUTABLE}
    log_uniform = lambda lo, hi: (lambda: math.exp(rng.uniform(math.log(lo), math.log(hi))))
    mutations = {
        k: (log_uniform(max(lo, 1e-5), hi) if k in ('lr', 'weight_decay') else (lambda lo=lo, hi=hi: rng.uniform(lo, hi)))
        for k, (lo, hi) in MUTABLE.items()
    }
    pbt = PopulationBasedTraining(
        time_attr='training_iteration',
        perturbation_interval=interval,
        hyperparam_mutations=mutations,
        quantile_fraction=0.25,
        resample_probability=0.25,
    )
    tuner = rtune.Tuner(
        rtune.with_resources(_trainable, {'cpu': 1, 'gpu': 1 / concurrent}),
        param_space={'data': data, 'base': base, **space},
        tune_config=rtune.TuneConfig(metric='val_acc', mode='max', scheduler=pbt, num_samples=population, max_concurrent_trials=concurrent),
        run_config=rtune.RunConfig(storage_path=str(Path.cwd() / 'ray'), name='pbt', verbose=0, stop={'training_iteration': int(base['epochs'])}),
    )
    results = tuner.fit()
    rows = [r for r in results if r.metrics]
    final = [
        {'val_acc': r.metrics.get('val_acc'), 'test_acc': r.metrics.get('test_acc'), 'settings': {k: r.config.get(k) for k in MUTABLE}} for r in rows
    ]
    best = max(final, key=lambda f: f['val_acc'] or 0)
    out = {
        'population': population,
        'concurrent': concurrent,
        'interval': interval,
        'best': best,
        'members': final,
        'seconds': round(time.time() - t0, 1),
    }
    Path('pbt.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    print(json.dumps({'best': best, 'seconds': out['seconds']}, default=str), flush=True)
    ray.shutdown()
    return out


def probe(base: dict, data: str | None = None, seeds: Iterable[int] = (0, 1, 2), batches: Iterable[int] = (256, 512, 1024, 2048, 4096)) -> dict:
    """On one card, for the configuration ``base`` (``train``'s names): its accuracy over ``seeds`` (the baseline PBT is
    measured against), each batch size with the learning rate scaled by the square root of the batch ratio (Adam's
    usual rule), and the throughput of one training against two at once (``parallel.GPUProcessExecutor``)."""
    from runq.search.parallel import GPUProcessExecutor

    from .train import train

    out = {'seeds': [], 'batches': [], 'concurrency': []}
    for s in seeds:
        r = train(data=data, **{**base, 'seed': s})
        out['seeds'].append({'seed': s, 'val_acc': r['val_acc'], 'test_acc': r['test_acc'], 'train_seconds': r['train_seconds']})
        print('seed', out['seeds'][-1], flush=True)
    b0 = int(base['batch_size'])
    for b in batches:
        r = train(data=data, **{**base, 'batch_size': b, 'lr': base['lr'] * math.sqrt(b / b0)})
        out['batches'].append(
            {
                'batch_size': b,
                'val_acc': r['val_acc'],
                'test_acc': r['test_acc'],
                'train_seconds': r['train_seconds'],
                'images_per_s': r['images_per_s'],
            }
        )
        print('batch', out['batches'][-1], flush=True)
    for k in (1, 2):
        with GPUProcessExecutor(k) as ex:
            ex.submit('cifarq.train:train', data=data, **{**base, 'epochs': 2}).result()  # (warm: start, imports)
            t0 = time.time()
            futures = [ex.submit('cifarq.train:train', data=data, **base) for _ in range(k)]
            rs = [f.result() for f in futures]
        wall = time.time() - t0
        out['concurrency'].append(
            {'at_once': k, 'wall_seconds': round(wall, 1), 'trainings_per_hour': round(3600 * k / wall, 1), 'val_acc': [r['val_acc'] for r in rs]}
        )
        print('concurrency', out['concurrency'][-1], flush=True)
    Path('probe.json').write_text(json.dumps(out, indent=1) + '\n')
    return out
