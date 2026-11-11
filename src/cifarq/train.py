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

In JAX (Flax NNX, Optax; the shared pieces in ``jx``): an epoch is one jitted ``lax.scan`` over its steps, with the
images resident on the device as uint8 and augmented inside the step. Every run is ``vmap``-ed over a leading seed
axis, so ``train_seeds`` trains several seeds of one configuration as one program on one card (the champion's
seeds at the cost of a little more than one), and ``train`` is its one-seed case. Float hyperparameters are traced
arguments (``jx.dynamic``): a search's trials differing only in them share a compiled program (and JAX's persistent
cache, ``jx.cache``, keeps it across processes). ``device='cpu'`` runs the same code on the CPU, for the tests.
``train(**hyperparameters)`` is what a queue job calls (``runq.function_job``); its return value, a dict of plain
numbers, is the job's result.
"""

from __future__ import annotations

import contextlib
import math
import time
from typing import TYPE_CHECKING

from . import evaluate
from .model import MEAN, STD, model, whitening_filters  # noqa: F401  (train.model etc.: the names before the split)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    import jax
    import numpy as np

DEFAULTS = dict(
    # model
    width=32,
    dropout=0.1,
    activation='relu',
    celu_alpha=0.075,
    head='linear',
    logit_scale=1.0,
    whiten=False,
    whiten_eps=1e-2,
    # optimisation
    optimizer='sgd',
    lr=0.08,
    momentum=0.9,
    nesterov=True,
    beta1=0.9,
    beta2=0.999,
    weight_decay=5e-4,
    batch_size=512,
    epochs=10,
    label_smoothing=0.1,
    grad_clip=0.0,
    ema_decay=0.0,
    # learning rate schedule
    warmup_frac=0.25,
    warmup_start_frac=0.04,
    decay='cosine',
    final_lr_frac=0.01,
    # augmentation (0: off)
    aug_translate=4.0,
    aug_rotate=0.0,
    aug_scale=0.0,
    aug_flip=0.5,
    aug_brightness=0.0,
    aug_contrast=0.0,
    aug_saturation=0.0,
    cutout_size=0,
    cutout_prob=0.5,
    mix='none',
    mix_alpha=1.0,
    mix_prob=0.5,
    # evaluation
    tta_flip=False,
    seed=0,
)

# traced in the step: a change of these is not a recompile
DYNAMIC = (
    'logit_scale',
    'label_smoothing',
    'ema_decay',
    'aug_translate',
    'aug_rotate',
    'aug_scale',
    'aug_flip',
    'aug_brightness',
    'aug_contrast',
    'aug_saturation',
    'cutout_size',
    'cutout_prob',
    'mix_alpha',
    'mix_prob',
)
BN_REFRESH = 25  # batches refreshing batch norm's statistics for schedule-free's averaged weights


def lr_factor(step: int, total: int, h: dict) -> float:
    """The learning rate as a fraction of its peak at ``step`` of ``total``: warmup, then the decay (the Python form
    of ``jx.warmup_decay``, for reports and tests)."""
    warm = int(h['warmup_frac'] * total)
    if step < warm:
        return h['warmup_start_frac'] + (1 - h['warmup_start_frac']) * step / max(1, warm)
    t = (step - warm) / max(1, total - warm)
    end = h['final_lr_frac']
    return {'cosine': end + (1 - end) * 0.5 * (1 + math.cos(math.pi * t)), 'linear': 1 - (1 - end) * t, 'exponential': end**t, 'none': 1.0}[
        h['decay']
    ]


def augment(key: jax.Array, x: jax.Array, h: dict, hp: dict) -> jax.Array:
    """A batch (float, 0-1, NHWC) augmented per image: ``h`` (Python) says which parts are on, ``hp`` (traced) how
    strongly; each part skipped when off."""
    import jax
    import jax.numpy as jnp

    from . import jx

    n = x.shape[0]
    keys = iter(jax.random.split(key, 12))  # (11 drawn with every augmentation on; 10 raised StopIteration while tracing)
    u = lambda: jax.random.uniform(next(keys), (n,), minval=-1.0, maxval=1.0)  # one per image
    col = lambda v: v[:, None, None, None]
    if h['aug_brightness'] or h['aug_contrast'] or h['aug_saturation']:
        if h['aug_brightness']:
            x = x * col(1 + hp['aug_brightness'] * u())
        if h['aug_contrast']:
            m = x.mean(axis=(1, 2, 3), keepdims=True)
            x = (x - m) * col(1 + hp['aug_contrast'] * u()) + m
        if h['aug_saturation']:
            g = 0.299 * x[..., :1] + 0.587 * x[..., 1:2] + 0.114 * x[..., 2:3]
            x = (x - g) * col(1 + hp['aug_saturation'] * u()) + g
        x = jnp.clip(x, 0, 1)
    if h['aug_translate'] or h['aug_rotate'] or h['aug_scale'] or h['aug_flip']:
        angle = hp['aug_rotate'] * jnp.pi / 180 * u()
        scale = 1 + hp['aug_scale'] * u()
        flip = jnp.where(jax.random.uniform(next(keys), (n,)) < hp['aug_flip'], -1.0, 1.0)
        shift = hp['aug_translate'] * 2 / x.shape[2]  # pixels in normalised coordinates
        cos, sin = jnp.cos(angle) / scale, jnp.sin(angle) / scale
        theta = jnp.stack([jnp.stack([flip * cos, -sin, shift * u()], 1), jnp.stack([flip * sin, cos, shift * u()], 1)], 1)
        x = jx.affine(x, theta)
    if h['cutout_size']:
        s = x.shape[1]
        half = jnp.floor(hp['cutout_size'] / 2)
        cy, cx = jax.random.randint(next(keys), (n,), 0, s), jax.random.randint(next(keys), (n,), 0, s)
        r = jnp.arange(s)
        box = (jnp.abs(r[None, :, None] - cy[:, None, None]) <= half) & (jnp.abs(r[None, None, :] - cx[:, None, None]) <= half)
        box &= (jax.random.uniform(next(keys), (n,)) < hp['cutout_prob'])[:, None, None]
        x = x * (~box)[..., None]
    return x


def _mix(key: jax.Array, xb: jax.Array, h: dict, hp: dict) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mixup or cutmix on a batch with probability ``mix_prob``: the mixed batch, the partner permutation and the
    weight of the batch's own labels (1 when not mixed)."""
    import jax
    import jax.numpy as jnp

    k1, k2, k3, k4 = jax.random.split(key, 4)
    n = xb.shape[0]
    other = jax.random.permutation(k2, n)
    on = jax.random.uniform(k1) < hp['mix_prob']
    lam = jax.random.beta(k3, hp['mix_alpha'], hp['mix_alpha'])
    if h['mix'] == 'mixup':
        mixed = lam * xb + (1 - lam) * xb[other]
    else:  # cutmix: a box of area 1 - lam from another image
        side = jnp.floor(32 * jnp.sqrt(1 - lam)).astype(jnp.int32)
        y0, x0 = jax.random.randint(k4, (2,), 0, 33 - side)
        r = jnp.arange(32)
        box = ((r >= y0) & (r < y0 + side))[:, None] & ((r >= x0) & (r < x0 + side))[None, :]
        mixed = jnp.where(box[None, :, :, None], xb[other], xb)
        lam = 1 - side * side / 1024
    return jnp.where(on, mixed, xb), other, jnp.where(on, lam, 1.0)


class Run:
    """One configuration's training at several seeds (vmapped over a leading axis), an epoch at a time: ``epoch()``
    trains one and returns the seeds' mean losses, ``weights()`` the weights to evaluate (the schedule-free average
    with batch norm refreshed for it, the EMA, or the weights), ``accuracies(split)`` per seed, ``state()``/``load()``
    for a checkpoint (host arrays), ``set(**floats)`` changes traced hyperparameters from the next step without a
    recompile (PBT's moves), ``save(i, directory)`` one seed's network. ``train`` and ``pbt.Trainer`` are both this."""

    def __init__(self, h: dict, seeds: list[int], data: str | None = None, device: str = 'auto') -> None:
        import jax
        import numpy as np

        from . import jx
        from .data import dataset

        jx.cache()
        self.h, self.seeds, self.device = h, list(seeds), device
        self.dev = jx.pick(device)
        arrays = dataset.load(data)  # (the split is the dataset's: the same for every trial)
        classes = len(arrays['classes'])
        put = lambda a: jax.device_put(np.asarray(a), self.dev)
        x_train, y_train = put(arrays['x_train']), put(arrays['y_train'].astype(np.int32))
        # (the training set reaches the compiled programs as arguments: closed over, it was baked into each program as
        # a 138 MB constant, which slowed every compilation and filled the persistent cache)
        self.x_train, self.y_train = x_train, y_train
        self.x = {s: put(arrays[f'x_{s}']) for s in ('val', 'test')}
        self.y = {s: np.asarray(arrays[f'y_{s}']) for s in ('val', 'test')}
        S, steps, bs = self._build_state(h, arrays, classes)
        self._compile(h, S, arrays, classes, steps, bs)
        self.done, self._w = 0, None

    def _build_state(self, h: dict, arrays: dict, classes: int) -> tuple[Any, int, int]:
        """The models of every seed, stacked: the training state (parameters, the rest, optimizer state, keys, step
        counts, the EMA copy) on the device; (the first model's split, steps per epoch, batch size)."""
        import jax
        import jax.numpy as jnp
        import numpy as np

        from . import jx

        with jax.default_device(self.dev):
            whitening = None
            if h['whiten']:  # from 5000 training images (the validation images unseen)
                x5 = (arrays['x_train'][:5000].astype(np.float32) / 255 - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)
                whitening = whitening_filters(x5, h['whiten_eps'])
            splits = [
                jx.Split(
                    model(
                        h['width'],
                        h['dropout'],
                        h['activation'],
                        classes=classes,
                        celu_alpha=h['celu_alpha'],
                        head=h['head'],
                        whitening=whitening,
                        seed=s,
                    )
                )
                for s in self.seeds
            ]
            S = self.split = splits[0]
            stack = lambda trees: jax.tree.map(lambda *a: jnp.stack(a), *trees)
            params, rest = stack([s.params for s in splits]), stack([s.rest for s in splits])
            k, bs = len(self.seeds), h['batch_size']
            self.steps_per_epoch = steps = len(arrays['y_train']) // bs
            self.total = total = h['epochs'] * steps
            # (the step count traced too: trials differing only in epochs share a compiled program)
            self.hp = {**jx.dynamic(h, jx.OPTIMIZER_FLOATS), **jx.dynamic(h, DYNAMIC), **jx.dynamic({'total': total}, ['total'])}
            tx0, self.schedule_free = jx.optimizer(h, self.hp, S.params, total)
            use_ema = self.use_ema = bool(h['ema_decay']) and not self.schedule_free
            self.st = {
                'params': params,
                'rest': rest,
                'opt': jax.vmap(tx0.init)(params),
                'key': jnp.stack([jax.random.key_data(jax.random.key(s)) for s in self.seeds]),
                'n': jnp.zeros((k,), jnp.float32),
            }
            if use_ema:
                self.st['ema'] = jax.tree.map(jnp.copy, (params, rest))  # (its own buffers: the state is donated)
        return S, steps, bs

    def _compile(self, h: dict, S: Any, arrays: dict, classes: int, steps: int, bs: int) -> None:
        """The compiled epoch (a scan of training steps, vmapped over the seeds, the state donated) and the weights to
        evaluate (schedule-free's averaged ones with batch norm refreshed, the EMA, or the live ones)."""
        import jax
        import jax.numpy as jnp
        import optax

        from . import jx

        use_ema = self.use_ema
        norm = lambda xb: (xb - jnp.asarray(MEAN)) / jnp.asarray(STD)
        mixing = h['mix'] != 'none'

        def step(carry: dict, idx: jax.Array, hp: dict, x_data: jax.Array, y_data: jax.Array) -> tuple[dict, jax.Array]:
            params, rest, opt, key, n = carry['params'], carry['rest'], carry['opt'], carry['key'], carry['n']
            key, ka, km = jax.random.split(jax.random.wrap_key_data(key), 3)
            xb = augment(ka, x_data[idx].astype(jnp.float32) / 255, h, hp)
            yb = y_data[idx]
            ce = lambda out, y: jx.cross_entropy(out, y, classes, hp['label_smoothing'])
            if mixing:
                xb, other, lam = _mix(km, xb, h, hp)
                objective = lambda out: lam * ce(out, yb) + (1 - lam) * ce(out, yb[other])
            else:
                objective = lambda out: ce(out, yb)

            def loss_fn(p: Any, r: Any) -> tuple[jax.Array, Any]:
                out, new_rest = S.apply(p, r, norm(xb), train=True)
                return objective(out * hp['logit_scale']), new_rest

            (loss, rest), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, rest)
            tx, _ = jx.optimizer(h, hp, params, hp['total'])
            updates, opt = tx.update(grads, opt, params)
            params = optax.apply_updates(params, updates)
            new = {'params': params, 'rest': rest, 'opt': opt, 'key': jax.random.key_data(key), 'n': n + 1}
            if use_ema:
                new['ema'] = jx.ema_update(carry['ema'], (params, rest), n, hp['ema_decay'])
            return new, loss

        def epoch(carry: dict, hp: dict, x_data: jax.Array, y_data: jax.Array) -> tuple[dict, jax.Array]:
            key, kp = jax.random.split(jax.random.wrap_key_data(carry['key']))
            carry = {**carry, 'key': jax.random.key_data(key)}
            perm = jax.random.permutation(kp, y_data.shape[0])[: steps * bs].reshape(steps, bs)
            carry, losses = jax.lax.scan(lambda c, i: step(c, i, hp, x_data, y_data), carry, perm)
            return carry, losses.mean()

        def refresh(params: Any, rest: Any, x_data: jax.Array) -> Any:
            """Schedule-free: batch norm's statistics refreshed for the averaged weights (a few training batches, no
            gradients), as the method's authors advise for networks with batch norm. A scan over the batches: one
            forward pass compiled, not 25 unrolled (which compiled as long as a whole training epoch)."""
            n = min(len(arrays['y_train']) // bs, BN_REFRESH)
            idx = jnp.arange(n * bs).reshape(n, bs)

            def one(r: Any, i: jax.Array) -> tuple[Any, None]:
                return S.apply(params, r, norm(x_data[i].astype(jnp.float32) / 255), train=True)[1], None

            return jax.lax.scan(one, rest, idx)[0]

        def weights(st: dict, x_data: jax.Array) -> tuple[Any, Any]:
            if self.schedule_free:
                p = jx.eval_params(st['opt'], st['params'], True)
                return p, refresh(p, st['rest'], x_data)
            return st['ema'] if use_ema else (st['params'], st['rest'])

        self._epoch = jax.jit(jax.vmap(epoch, in_axes=(0, None, None, None)), donate_argnums=0)
        self._weights = jax.jit(jax.vmap(weights, in_axes=(0, None)))

    def set(self, **floats) -> None:
        import jax.numpy as jnp

        for k, v in floats.items():
            if k not in self.hp:
                raise TypeError(f'{k} is not a traced hyperparameter')
            self.h[k] = float(v)
            self.hp[k] = jnp.asarray(float(v), jnp.float32)

    def epoch(self) -> np.ndarray:
        import numpy as np

        self.st, loss = self._epoch(self.st, self.hp, self.x_train, self.y_train)
        self.done += 1
        self._w = None
        return np.asarray(loss)

    def weights(self) -> tuple[Any, Any]:
        if self._w is None:
            self._w = self._weights(self.st, self.x_train)
        return self._w

    def seed_weights(self, i: int) -> tuple[Any, Any]:
        import jax

        p, r = self.weights()
        return jax.tree.map(lambda a: a[i], p), jax.tree.map(lambda a: a[i], r)

    def accuracies(self, split: str) -> list[float]:
        return [
            evaluate.accuracy(self.split, *self.seed_weights(i), self.x[split], self.y[split], self.h['tta_flip']) for i in range(len(self.seeds))
        ]

    def lr(self) -> float:
        """The learning rate the last epoch ended at."""
        if self.schedule_free:
            return self.h['lr']
        return self.h['lr'] * lr_factor(min(self.done * self.steps_per_epoch, self.total - 1), self.total, self.h)

    def state(self) -> dict:
        from . import jx

        return {'st': jx.to_host(self.st), 'done': self.done, 'h': dict(self.h)}

    def load(self, state: dict) -> None:
        from . import jx

        self.st = jx.from_host(state['st'], self.dev)
        self.done, self._w = state['done'], None

    def params(self) -> int:
        from . import jx

        return jx.count(self.split.params) + self.split.frozen

    def save(self, i: int, directory: str) -> str:
        from .hub import save as save_model

        return str(save_model(self.split, *self.seed_weights(i), {**self.h, 'seed': self.seeds[i]}, directory))


def _run(h: dict, seeds: list[int], data: str | None, report: Any, device: str, save_dirs: list[str | None] | None = None) -> list[dict]:
    import jax
    import numpy as np

    from . import jx

    run = Run(h, seeds, data, device)
    k = len(seeds)
    try:  # (taking turns at the card with other trials: opt-in)
        from runq.core.gpu import device_turn
    except ImportError:
        device_turn = contextlib.nullcontext
    if report is None:
        stop_here = lambda step, value: False
    elif callable(report):
        stop_here = report
    else:
        from runq.search.hpo import reporter  # a study's spec: only inside the queue's trials

        stop_here = reporter(report)
    pruned = False
    histories = [[] for _ in seeds]
    with device_turn():
        t0 = time.time()
        for e in range(h['epochs']):
            loss = run.epoch()
            accs = run.accuracies('val')
            for i in range(k):
                histories[i].append(
                    {
                        'epoch': e + 1,
                        'loss': round(float(loss[i]), 4),
                        'val_acc': round(accs[i], 4),
                        'lr': round(run.lr(), 6),
                        'seconds': round(time.time() - t0, 1),
                    }
                )
            if not np.all(np.isfinite(loss)):  # diverged: no use training on
                break
            if e + 1 < h['epochs'] and stop_here(round(100 * (e + 1) / h['epochs']), float(np.mean(accs))):
                pruned = True  # the study's pruner: worse than its rung's best
                break
        jax.block_until_ready(run.st['params'])
        seconds = time.time() - t0
        test = None if pruned else run.accuracies('test')
    name = jx.device_name(device)
    out = []
    for i, s in enumerate(seeds):
        history = histories[i]
        row = {
            'val_acc': history[-1]['val_acc'],
            'train_seconds': round(seconds, 1),
            'images_per_s': round(k * len(history) * run.steps_per_epoch * h['batch_size'] / seconds),
            'device': name,
            'history': history,
            'hyperparameters': {**h, 'seed': s},
        }
        if test is None:  # (pruned)
            row.update(pruned=True, progress=round(100 * len(history) / h['epochs']), epochs_run=len(history))
        else:
            row.update(test_acc=round(test[i], 4), params=run.params())
            if save_dirs and save_dirs[i]:
                row['saved'] = run.save(i, save_dirs[i])
        out.append(row)
    return out


def _check(hyperparameters: dict) -> dict:
    unknown = set(hyperparameters) - set(DEFAULTS)
    if unknown:
        raise TypeError(f'unknown hyperparameters: {sorted(unknown)}')
    h = {**DEFAULTS, **hyperparameters}
    for k in ('width', 'batch_size', 'epochs', 'cutout_size', 'seed'):
        h[k] = int(h[k])
    h['whiten'], h['tta_flip'] = bool(h['whiten']), bool(h['tta_flip'])
    return h


def train(data: str | None = None, save: str | None = None, report: Any = None, device: str = 'auto', **hyperparameters) -> dict:
    """Train one model with ``DEFAULTS`` updated by ``hyperparameters`` (an unknown name is an error); its validation
    and test accuracy, time and throughput, and the per-epoch history. ``save``: a directory (relative: in the job's
    directory, which comes back with the job) to write the trained network to as a ``hub.ResNet9`` (the EMA's or
    schedule-free average's weights when there is one). ``report``: early stopping (``runq.hpo.reporter``: a study's
    spec, or a callable ``(step, value) -> stop``): after each epoch the validation accuracy is reported at the percent
    of training done, and a trial told to stop returns at once with ``pruned: True`` (no test evaluation). ``device``:
    ``cuda``, ``cpu`` or ``auto`` (the GPU when there is one); not a hyperparameter, so not in the result's
    ``hyperparameters``."""
    h = _check(hyperparameters)
    return _run(h, [h['seed']], data, report, device, [save])[0]


def train_seeds(
    seeds: Iterable[int],
    data: str | None = None,
    save: str | None = None,
    report: Any = None,
    device: str = 'auto',
    together: bool = False,
    **hyperparameters,
) -> list[dict]:
    """``train`` of one configuration at each of ``seeds`` in one process; a list of ``train``'s results in the seeds'
    order. ``save``: a directory, each seed's network written to ``<save>/seed<s>``. ``seed`` in ``hyperparameters`` is
    ignored. By default the seeds train one after another, sharing one compiled program (the persistent cache serves
    the later ones). ``together``: one program vmapped over the seeds instead (``report`` then sees their mean); on the
    RTX PRO 6000 that was slower for this ResNet-9 (1.02 s an epoch a seed against 0.61 s alone, 2026-11-09): XLA
    batches the seeds' convolutions less well than it runs one."""
    from pathlib import Path

    h = _check({k: v for k, v in hyperparameters.items() if k != 'seed'})
    seeds = [int(s) for s in seeds]
    dirs = [str(Path(save) / f'seed{s}') for s in seeds] if save else [None] * len(seeds)
    if together:
        return _run(h, seeds, data, report, device, dirs)
    return [_run({**h, 'seed': s}, [s], data, report, device, [d])[0] for s, d in zip(seeds, dirs, strict=True)]
