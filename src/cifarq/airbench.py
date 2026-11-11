"""The airbench family (K. Jordan, "94 % on CIFAR-10 in 3.29 seconds on a single GPU", 2024; github.com/KellerJordan/
cifar10-airbench at 4c1b6d1, ``legacy/airbench94.py`` and ``legacy/airbench96.py``) ported to this project's dataset
and job contract, every choice a hyperparameter (``DEFAULTS``; ``AIRBENCH94`` and ``AIRBENCH96`` are the two published
recipes as points of it). JAX (Flax NNX, Optax) on NHWC images, the shared pieces in ``jx``.

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
translation from reflect padding, cutout. Evaluation averages ``tta_level`` views: 0 the image, 1 and its mirror, 2 and
both shifted one pixel up-left and down-right (6 views, the shifted ones at half weight).

Additions, off by default: ``train_on='train+val'`` (all 50000 images, the published setting, for a record-format
run: the validation accuracy is then on trained images and only reported), and distillation from a teacher saved by
``train(save=...)`` (``kd_teacher``: its directory, or several separated by commas for an ensemble; ``kd_weight`` of
the loss is the KL divergence to its softened predictions at ``kd_temp``, the teacher run on each augmented batch).

In JAX: an epoch is one jitted ``lax.scan`` over its steps (a step past the schedule's last is skipped, so a
fractional last epoch is the same program); the epoch's augmented images are made on the device first. Optimiser and
loss scalars are traced arguments (``jx.dynamic``); the architecture and switches are static. Parameter groups are
labels over the parameter tree (``labels``): batch-norm biases, batch-norm scales (frozen unless ``bn_scale``), the
whitening weight (frozen) and bias (trained for ``whiten_bias_epochs``), the filters (Muon's, under ``optimizer=
'muon'``) and the rest.

Differences from the Torch port, all numerical: ``precision`` 'fp16' and 'amp' both mean float32 weights and
optimiser state with bfloat16 arithmetic on a GPU (Torch's 'fp16' kept the weights themselves in float16); on a CPU
the arithmetic is float32, as before. 'auto' picks bfloat16 where the card runs it natively and float16 where it
does not (a Tesla T4), and 'f16' asks for float16 outright: see :mod:`cifarq.precision`. Batch norm's running
variance is the biased batch variance (Torch's is unbiased, a factor n/(n-1) at n = batch x pixels). The whitening
and PCA eigenvectors are computed in float64 (Torch: float32). The loss is computed in float32 under every precision.
"""

from __future__ import annotations

import functools
import json
import math
import operator
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from types import ModuleType
    from typing import Any

    import jax
    import numpy as np
    import optax
    from flax import nnx
    from jax.typing import DTypeLike
    from numpy.typing import ArrayLike as NPArrayLike

DEFAULTS = dict(
    # network
    widths=(64, 256, 256),
    depth=2,
    scaling_factor=1 / 9,
    bn_momentum=0.4,
    whiten_kernel=2,
    whiten_eps=5e-4,
    # optimisation
    epochs=9.9,
    batch_size=1024,
    lr=11.5,
    momentum=0.85,
    weight_decay=0.0153,
    bias_scaler=64.0,
    label_smoothing=0.2,
    whiten_bias_epochs=3,
    warmup_frac=0.23,
    final_lr_frac=0.07,
    lookahead=True,
    # augmentation
    flip=True,
    translate=2,
    cutout=0,
    # evaluation
    tta_level=2,
    # additions
    arch='airbench',
    arch_kw={},
    bn_scale=False,
    param_cap=0,
    tie=0,
    flip_eq=0,
    pca_init=False,
    tails=0,
    tail_epochs=3,
    kd_weighting='none',
    tta_gate=0.0,
    optimizer='sgd',
    beta2=0.99,
    augment='none',
    augment_m=9,
    train_on='train',
    kd_teacher='',
    kd_weight=0.0,
    kd_temp=4.0,
    precision='fp16',
    eq_group='d1',
    blur_pool=False,
    se=0,
    srf=False,
    filter_keep=1.0,
    filter_proxy_widths=(32, 64, 64),
    filter_proxy_epochs=0.0,
    muon_lr=0.24,
    muon_momentum=0.6,
    eval_batch=0,
    test='open',
    seed=0,
)

AIRBENCH94 = dict(DEFAULTS)
AIRBENCH96 = {
    **DEFAULTS,
    'widths': (128, 384, 512),
    'depth': 3,
    'bn_momentum': 0.1,
    'epochs': 37.0,
    'lr': 9.0,
    'weight_decay': 0.012,
    'warmup_frac': 0.1,
    'final_lr_frac': 0.0,
    'translate': 4,
    'cutout': 12,
}

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

# traced in the step: a change of these is not a recompile
DYNAMIC = (
    'lr',
    'momentum',
    'weight_decay',
    'bias_scaler',
    'label_smoothing',
    'warmup_frac',
    'final_lr_frac',
    'kd_weight',
    'kd_temp',
    'muon_lr',
    'muon_momentum',
    'beta2',
)


def _modules() -> dict[str, type]:
    """The module classes (built on first use, so importing this file costs no JAX import)."""
    global _M
    if '_M' in globals():
        return _M
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from . import jx, methods

    gelu = lambda x: jax.nn.gelu(x, approximate=False)

    class Conv(methods.Kernel3):
        """A convolution initialised as airbench's: PyTorch's default, a zero bias, then a partial identity (dirac) on
        its first ``cin`` filters."""

        def __init__(self, cin: int, cout: int, k: int = 3, padding: int | str = 1, bias: bool = False, *, dtype: DTypeLike, rngs: nnx.Rngs) -> None:
            w = jx.torch_kernel_init()(rngs.params(), (k, k, cin, cout))
            self.weight = nnx.Param(methods.dirac(w, cin))
            self.bias = nnx.Param(jnp.zeros((cout,), jnp.float32)) if bias else None
            self.padding = [(padding, padding)] * 2 if isinstance(padding, int) else padding
            self.dtype, self.srf = dtype, False
            self.srf_able = k == 3

        def __call__(self, x: jax.Array) -> jax.Array:
            y = methods._conv(x, self._kernel(), self.dtype, self.padding)
            return y + self.bias[...].astype(y.dtype) if self.bias is not None else y

    class PairNorm(nnx.Module):
        """Batch norm with one set of statistics for all ``k`` members of a field (group-major channels)."""

        def __init__(self, c: int, k: int, momentum: float, *, rngs: nnx.Rngs) -> None:
            self.k, self.bn = k, jx.batchnorm(c, momentum=momentum, eps=1e-12, rngs=rngs)

        def __call__(self, x: jax.Array) -> jax.Array:
            n, hh, ww, kc = x.shape
            y = x.reshape(n, hh, ww, self.k, kc // self.k).transpose(0, 3, 1, 2, 4).reshape(n * self.k, hh, ww, kc // self.k)
            y = self.bn(y)
            return y.reshape(n, self.k, hh, ww, kc // self.k).transpose(0, 2, 3, 1, 4).reshape(n, hh, ww, kc)

    class Group(nnx.Module):
        def __init__(self, cin: int, cout: int, eq: str, cfg: dict, *, rngs: nnx.Rngs) -> None:
            depth, tie, dtype, mom = cfg['depth'], cfg['tie'], cfg['dtype'], cfg['bn_momentum']
            eq_conv = {'d1': methods.FlipConv, 'd4': methods.P4MConv}[cfg['eq_group']]
            k = eq_conv.order
            if eq:  # 'lift' (plain in, pairs out) or 'pair' (pairs in and out)
                self.conv1, self.norm1 = eq_conv(cin, cout, lift=eq == 'lift', dtype=dtype, rngs=rngs), PairNorm(cout, k, mom, rngs=rngs)
                self.convs = nnx.List([eq_conv(cout, cout, dtype=dtype, rngs=rngs) for _ in range(depth - 1)])
                norm = lambda: PairNorm(cout, k, mom, rngs=rngs)
            else:
                self.conv1, self.norm1 = Conv(cin, cout, dtype=dtype, rngs=rngs), jx.batchnorm(cout, momentum=mom, eps=1e-12, rngs=rngs)
                self.convs = nnx.List([Conv(cout, cout, dtype=dtype, rngs=rngs) for _ in range(depth - 1)])
                norm = lambda: jx.batchnorm(cout, momentum=mom, eps=1e-12, rngs=rngs)
            self.pool = methods.MaxBlurPool() if cfg['blur_pool'] else None
            self.se = methods.FieldSE(cout, k if eq else 1, cfg['se'], dtype=dtype, rngs=rngs) if cfg['se'] else None
            self.norms = nnx.List([norm() for _ in range(depth - 1 + (tie if depth > 1 else 0))])
            self.depth, self.tie = depth, tie

        def _conv(self, conv: Callable[[jax.Array], jax.Array], x: jax.Array, record: dict | None) -> jax.Array:
            if record is not None:
                record[id(conv)] = x
            return conv(x)

        def __call__(self, x: jax.Array, record: dict | None = None) -> jax.Array:
            y = self._conv(self.conv1, x, record)
            y = self.pool(y) if self.pool is not None else jx.max_pool(y)
            x = gelu(self.norm1(y))
            x0 = x
            convs = list(self.convs) + ([self.convs[-1]] * self.tie if len(self.convs) else [])
            for i, (conv, norm) in enumerate(zip(convs, self.norms, strict=False)):
                x = norm(self._conv(conv, x, record))
                if self.depth >= 3 and i == len(convs) - 1:
                    x = x + x0  # (depth 3: the residual over the last two)
                x = gelu(x)
            return self.se(x) if self.se is not None else x

    class Airbench(nnx.Module):
        def __init__(self, cfg: dict, *, rngs: nnx.Rngs) -> None:
            ww = 2 * 3 * cfg['whiten_kernel'] ** 2
            w1, w2, w3 = cfg['widths']
            fe = cfg['flip_eq']
            eqs = [''] * (3 - fe) + (['lift'] + ['pair'] * (fe - 1) if fe else [])
            self.whiten = Conv(3, ww, cfg['whiten_kernel'], padding=0, bias=True, dtype=cfg['dtype'], rngs=rngs)
            self.whiten.srf_able = False  # (frozen: not a coefficient layer)
            self.groups = nnx.List(
                [Group(ww, w1, eqs[0], cfg, rngs=rngs), Group(w1, w2, eqs[1], cfg, rngs=rngs), Group(w2, w3, eqs[2], cfg, rngs=rngs)]
            )
            k = {'d1': 2, 'd4': 8}[cfg['eq_group']]
            self.gpool = methods.GroupMaxPool(k) if fe else None
            self.head = nnx.Linear(w3, cfg['classes'], use_bias=False, kernel_init=jx.torch_kernel_init(), dtype=cfg['dtype'], rngs=rngs)
            self.scaling_factor = float(cfg['scaling_factor'])

        def convs_in_order(self) -> list[nnx.Module]:
            out = []
            for g in self.groups:
                for c in [g.conv1, *g.convs]:
                    if isinstance(c, Conv) and c.srf_able:
                        out.append(c)
            return out

        def __call__(self, x: jax.Array, record: dict | None = None) -> jax.Array:
            x = gelu(self.whiten(x))
            for g in self.groups:
                x = g(x, record)
            x = self.gpool(x) if self.gpool is not None else jx.max_pool(x, 3).reshape(x.shape[0], -1)
            return self.head(x).astype(jnp.float32) * self.scaling_factor

    _M = {'Conv': Conv, 'Airbench': Airbench, 'PairNorm': PairNorm, 'Group': Group}
    return _M


def model(
    widths: Iterable[int] = (64, 256, 256),
    depth: int = 2,
    scaling_factor: float = 1 / 9,
    bn_momentum: float = 0.4,
    whiten_kernel: int = 2,
    classes: int = 10,
    tie: int = 0,
    flip_eq: int = 0,
    eq_group: str = 'd1',
    blur_pool: bool = False,
    se: int = 0,
    srf: bool = False,
    dtype: DTypeLike | None = None,
    seed: int = 0,
) -> nnx.Module:
    """The network (Flax NNX, float32 weights; ``dtype`` the arithmetic's, float32 by default). ``bn_momentum`` is
    PyTorch's (the weight of the new batch). ``tie`` (method MT5): each group's last convolution applied ``tie`` more
    times, its weights shared, a batch norm of its own each time (depth at no stored convolution weights). ``flip_eq``
    (method MT1): the last ``flip_eq`` groups made equivariant to horizontal flips (``methods.FlipConv``: channel pairs,
    half the free weights of a plain convolution of twice the width), the features pooled over each pair at the end
    (invariance by construction, not by test-time mirroring). ``eq_group`` 'd4': those groups equivariant to the flips
    and quarter turns instead (``methods.P4MConv``, fields of 8, 1/8 of the expanded convolution's weights).
    ``blur_pool``: each group's downsampling anti-aliased (``methods.MaxBlurPool``) instead of a 2x2 max pool. ``se``: a
    squeeze-and-excitation block at the end of each group with this reduction (``methods.FieldSE``, shared over a field;
    0: none). ``srf``: every trainable 3x3 kernel stored as structured-receptive-field coefficients
    (``methods.use_srf``)."""
    import jax.numpy as jnp
    from flax import nnx

    from . import methods

    cfg = dict(
        widths=tuple(int(w) for w in widths),
        depth=int(depth),
        scaling_factor=scaling_factor,
        bn_momentum=bn_momentum,
        whiten_kernel=int(whiten_kernel),
        classes=classes,
        tie=int(tie),
        flip_eq=int(flip_eq),
        eq_group=eq_group,
        blur_pool=bool(blur_pool),
        se=int(se),
        dtype=dtype or jnp.float32,
    )
    net = _modules()['Airbench'](cfg, rngs=nnx.Rngs(params=seed))
    if srf:
        methods.use_srf(net)
    return net


def build(h: dict, dtype: DTypeLike | None = None, abstract: bool = False) -> tuple[nnx.Module, nnx.Module | None]:
    """The network ``h`` describes, and its whitening convolution (or None): ``arch='airbench'`` is ``model``, any
    other a builder of ``cifarq.nets`` with ``arch_kw``. ``abstract``: shapes only, no arrays (``nnx.eval_shape``: for
    counting parameters and MACs of a large network cheaply)."""
    from flax import nnx

    def make() -> nnx.Module:
        if h.get('arch', 'airbench') == 'airbench':
            return model(
                tuple(h['widths']),
                h['depth'],
                h['scaling_factor'],
                h['bn_momentum'],
                h['whiten_kernel'],
                tie=int(h.get('tie', 0)),
                flip_eq=int(h.get('flip_eq', 0)),
                eq_group=h.get('eq_group', 'd1'),
                blur_pool=bool(h.get('blur_pool', False)),
                se=int(h.get('se', 0)),
                srf=bool(h.get('srf', False)),
                dtype=dtype,
                seed=int(h.get('seed', 0)),
            )
        from . import nets

        net = nets.build(h['arch'], dtype=dtype, seed=int(h.get('seed', 0)), **h.get('arch_kw', {}))
        for _, node in nnx.iter_graph(net):
            if isinstance(node, nnx.BatchNorm):
                node.momentum = 1 - h['bn_momentum']
        return net

    net = nnx.eval_shape(make) if abstract else make()
    return net, getattr(net, 'whiten', None)


def _path(path: Sequence[Any]) -> tuple:
    out = []
    for p in path:
        if hasattr(p, 'key'):
            out.append(p.key)
        elif hasattr(p, 'idx'):
            out.append(p.idx)
        elif getattr(p, 'name', None) != 'value':
            out.append(p.name)
    return tuple(out)


def labels(net: nnx.Module, params: nnx.State, bn_scale: bool = False) -> Any:
    """Each parameter's group, a tree like ``params``: 'bias' (batch-norm biases), 'frozen' (batch-norm scales unless
    ``bn_scale``; the whitening weight), 'whiten_b' (the whitening bias), 'filter' (other 4-d kernels), 'other'."""
    import jax
    from flax import nnx

    bn, whiten = set(), set()
    for path, node in nnx.iter_graph(net):
        if isinstance(node, nnx.BatchNorm):
            bn.add(tuple(path))
    w = getattr(net, 'whiten', None)
    if w is not None:
        whiten.add(('whiten',))

    def one(path: Sequence[Any], leaf: jax.Array) -> str:
        p = _path(path)
        parent, name = p[:-1], p[-1]
        if parent in bn:
            return 'bias' if name == 'bias' else ('other' if bn_scale else 'frozen')
        if parent in whiten:
            return 'whiten_b' if name == 'bias' else 'frozen'
        return 'filter' if leaf.ndim == 4 else 'other'

    return jax.tree_util.tree_map_with_path(one, params)


def param_counts(net: nnx.Module, bn_scale: bool = False) -> dict:
    """{'stored': every parameter array's size, frozen ones included; 'trainable': those training updates} (batch-norm
    running statistics are not counted; batch norm's frozen scale is a parameter and is counted). A ``jx.Frozen``
    array is stored and never trained, so it belongs to the first count and not the second."""
    import jax
    from flax import nnx

    from . import jx

    params = nnx.state(net, nnx.Param)
    lab = labels(net, params, bn_scale)
    sizes = jax.tree.map(lambda a: int(math.prod(a.shape)), params)
    never = jx.count(nnx.state(net, jx.Frozen))
    stored = sum(jax.tree.leaves(sizes)) + never
    frozen = never + sum(s for s, l in zip(jax.tree.leaves(sizes), jax.tree.leaves(lab), strict=True) if l == 'frozen')
    return {'stored': stored, 'trainable': stored - frozen}


def tta_logits(apply: Callable[[jax.Array], jax.Array], x: jax.Array, level: int) -> jax.Array:
    """Logits of normalised images ``x`` (NHWC) through ``apply`` averaged over ``level``'s views (0: 1 view, 1: 2,
    2: 6; 3: 18, every shift of -2, 0 or 2 pixels each way, mirrored: the session's P2, inference compute being
    uncapped)."""
    import jax.numpy as jnp

    mirror = lambda v: 0.5 * apply(v) + 0.5 * apply(v[:, :, ::-1])
    if level == 0:
        return apply(x)
    if level == 1:
        return mirror(x)
    if level >= 3:
        padded = jnp.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='reflect')
        return functools.reduce(operator.add, (mirror(padded[:, dy : dy + 32, dx : dx + 32]) for dy in (0, 2, 4) for dx in (0, 2, 4))) / 9
    padded = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='reflect')
    shifted = 0.5 * (mirror(padded[:, 0:32, 0:32]) + mirror(padded[:, 2:34, 2:34]))
    return 0.5 * mirror(x) + 0.5 * shifted


def _check(hyperparameters: dict) -> dict:
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
    return h


def _transform(h: dict, hp: dict, bs: int, eps: float) -> dict[str, optax.GradientTransformation]:
    """The optimiser over ``labels``' groups, without the schedule (``train`` multiplies the updates by the schedule's
    factor at each step). ``hp`` may be traced."""
    import optax

    from . import methods

    frozen = optax.set_to_zero()
    if h['optimizer'] in ('sgd', 'muon'):
        mu = hp['momentum']
        kilostep = 1024 * (1 + 1 / (1 - mu))
        lr, wd = hp['lr'] / kilostep, hp['weight_decay'] * bs / kilostep
        lr_b = lr * hp['bias_scaler']
        sgd = lambda lr_g: optax.chain(optax.add_decayed_weights(wd / lr_g), optax.trace(decay=mu, nesterov=True), optax.scale(-lr_g))
        groups = {'bias': sgd(lr_b), 'other': sgd(lr), 'whiten_b': sgd(lr), 'filter': sgd(lr), 'frozen': frozen}
        if h['optimizer'] == 'muon':  # (O4: Muon for the filters, the SGD above for the rest)
            groups['filter'] = methods.MuonSGD(hp['muon_lr'], hp['muon_momentum'])
        return groups
    if h['optimizer'] == 'adamw':  # (lr and weight_decay as AdamW's own; the loss a batch mean)
        adam = lambda wd: optax.chain(
            optax.scale_by_adam(b1=hp['momentum'], b2=hp['beta2'], eps=eps), optax.add_decayed_weights(wd), optax.scale(-hp['lr'])
        )
        return {
            'bias': adam(0.0),
            'other': adam(hp['weight_decay']),
            'whiten_b': adam(hp['weight_decay']),
            'filter': adam(hp['weight_decay']),
            'frozen': frozen,
        }
    raise ValueError(f"optimizer: 'sgd', 'adamw' or 'muon', not {h['optimizer']!r}")


def _floating(a: Any) -> bool:
    import jax.numpy as jnp

    return hasattr(a, 'dtype') and jnp.issubdtype(a.dtype, jnp.floating)


def train(data: str | None = None, save: str | None = None, report: Any = None, device: str = 'auto', **hyperparameters) -> dict:
    """Train one airbench network with ``DEFAULTS`` updated by ``hyperparameters`` (an unknown name is an error); the
    same contract as ``cifarq.train.train``: validation and test accuracy (``test_acc`` at ``tta_level``, and without
    TTA beside it), time, throughput, per-epoch history, the parameter counts; ``report`` early stopping after each
    epoch; ``save`` a directory for ``load``."""
    h = _check(hyperparameters)
    if h['param_cap']:  # over the cap: pruned before any work (the search's bound)
        stored = param_counts(build(h, abstract=True)[0])['stored']
        if stored > h['param_cap']:
            return {
                'val_acc': 0.0,
                'pruned': True,
                'progress': 0,
                'over_cap': True,
                'params': stored,
                'hyperparameters': {**h, 'widths': list(h['widths'])},
            }
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    from . import jx, precision
    from .data import dataset

    jx.cache()
    dev = jx.pick(device)
    dtype = precision.compute_dtype(h, device)
    with jax.default_device(dev):
        return _train(h, data, save, report, device, dev, dtype, jax, jnp, np, optax, jx, dataset)


def _train(
    h: dict,
    data: str | None,
    save: str | None,
    report: Any,
    device: str,
    dev: jax.Device,
    dtype: DTypeLike,
    jax: ModuleType,
    jnp: ModuleType,
    np: ModuleType,
    optax: ModuleType,
    jx: ModuleType,
    dataset: ModuleType,
) -> dict:
    from flax import nnx

    from . import augment, methods, precision

    arrays = dataset.load(data)
    mean_, std_ = np.asarray(MEAN, np.float32), np.asarray(STD, np.float32)
    images = lambda a: (np.asarray(a, np.float32) / 255 - mean_) / std_
    x_train, y_train = images(arrays['x_train']), np.asarray(arrays['y_train'], np.int32)
    x_val, y_val = images(arrays['x_val']), np.asarray(arrays['y_val'])
    x_test, y_test = images(arrays['x_test']), np.asarray(arrays['y_test'])
    if h['train_on'] == 'train+val':
        x_train, y_train = np.concatenate([x_train, x_val]), np.concatenate([y_train, y_val.astype(np.int32)])
    elif h['train_on'] != 'train':
        raise ValueError(f"train_on: 'train' or 'train+val', not {h['train_on']!r}")
    classes = len(arrays['classes'])

    net, whiten = build(h, dtype)
    counts = param_counts(net, h['bn_scale'])
    sgd_like = h['optimizer'] in ('sgd', 'muon')
    proxy, proxy_info = None, {}
    if h['filter_keep'] < 1:
        # D1, data filtering (airbench96_faster's idea, with a proxy trained first and then fixed): a small network
        # scores each batch by its loss, and the student trains on the hardest ``filter_keep`` share only
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            pr = train(
                data,
                save=tmp,
                device=device,
                widths=h['filter_proxy_widths'],
                depth=2,
                scaling_factor=1 / 9,
                epochs=h['filter_proxy_epochs'] or h['epochs'],
                batch_size=h['batch_size'],
                tta_level=0,
                test='sealed',
                precision=h['precision'],
                seed=h['seed'] + 10_000,
            )
            pnet, _ = load(tmp, dtype=dtype)
        pS = jx.Split(pnet)
        proxy = (pS, pS.params, pS.rest)
        proxy_info = {'proxy': {'params': pr['params'], 'val_acc_notta': pr['val_acc_notta'], 'epochs': pr['epochs_charged']}}
    teachers = []
    if h['kd_teacher'] and h['kd_weight'] > 0:
        # one directory, or several separated by commas: an ensemble, its softened predictions averaged
        for d in str(h['kd_teacher']).split(','):
            tnet, _ = load(d.strip(), dtype=dtype)
            tS = jx.Split(tnet)
            teachers.append((tS, tS.params, tS.rest))

    bs, n = h['batch_size'], len(y_train)
    steps_per_epoch = n // bs
    total = math.ceil(steps_per_epoch * h['epochs'])
    eps = 1e-6 if dtype == jnp.bfloat16 else 1e-8
    hp = jx.dynamic(h, DYNAMIC)

    if report is None:
        stop_here = lambda step, value: False
    elif callable(report):
        stop_here = report
    else:
        from runq.search.hpo import reporter

        stop_here = reporter(report)

    t0 = time.time()
    if whiten is not None:
        methods.whiten_init(whiten, x_train[:5000], h['whiten_eps'])
    if h['pca_init']:  # (MT3)
        from .methods import pca_init

        pca_init(net, x_train[:2048], seed=h['seed'])
    S = jx.Split(net)
    lab = labels(net, S.params, h['bn_scale'])
    is_wb = jax.tree.map(lambda l: l == 'whiten_b', lab)
    label_fn = lambda params: lab
    tx0 = optax.multi_transform(_transform(h, hp, bs, eps), label_fn)

    x_dev = jax.device_put(jnp.asarray(x_train, dtype), dev)
    y_dev = jax.device_put(jnp.asarray(y_train), dev)
    key = jax.random.key(h['seed'])
    key, kflip = jax.random.split(key)
    # alternating flip: each image flipped at random once, then all together every other epoch; translation from
    # reflect padding done once
    if h['flip']:
        base = jnp.where((jax.random.uniform(kflip, (n,)) < 0.5)[:, None, None, None], x_dev[:, :, ::-1], x_dev)
    else:
        base = x_dev
    tr = h['translate']
    padded = jnp.pad(base, ((0, 0), (tr, tr), (tr, tr), (0, 0)), mode='reflect') if tr > 0 else None
    mean_j, std_j = jnp.asarray(mean_), jnp.asarray(std_)

    src = padded if padded is not None else base

    @jax.jit
    def make_epoch(key: jax.Array, src: jax.Array) -> tuple[jax.Array, jax.Array]:
        kc, kq = jax.random.split(key)
        ep = augment.crop(kc, src, 32) if tr > 0 else src
        return ep, kq

    @jax.jit
    def finish_epoch(ep: jax.Array, key: jax.Array, epoch: jax.Array) -> jax.Array:
        if h['flip']:
            ep = jnp.where(epoch % 2 == 1, ep[:, :, ::-1], ep)
        if h['cutout'] > 0:
            ep = augment.cutout(key, ep, h['cutout'])
        return ep

    def epoch_images(key: jax.Array, epoch: jax.Array) -> jax.Array:
        ep, kq = make_epoch(key, src)
        kp, kcut = jax.random.split(kq)
        if h['augment'] != 'none':
            from .augment import apply as augment_policy

            ep = augment_policy(kp, ep, h['augment'], h['augment_m'], mean_j, std_j).astype(dtype)
        return finish_epoch(ep, kcut, epoch)

    keep = max(1, int(round(h['filter_keep'] * bs))) if proxy is not None else bs

    def factor(step: jax.Array, hp: dict) -> jax.Array:
        warm = jnp.floor(total * hp['warmup_frac'])
        s = step.astype(jnp.float32)
        f_up = s / jnp.maximum(warm, 1.0)
        f_down = (s - warm) / jnp.maximum(1.0, total - warm)
        return jnp.where(s < warm, 0.2 * (1 - f_up) + f_up, (1 - f_down) + hp['final_lr_frac'] * f_down)

    def per_example(logits: jax.Array, y: jax.Array, hp: dict) -> jax.Array:
        target = jax.nn.one_hot(y, classes) * (1 - hp['label_smoothing']) + hp['label_smoothing'] / classes
        return optax.softmax_cross_entropy(logits.astype(jnp.float32), target)

    def real_step(carry: dict, xb: jax.Array, yb: jax.Array, hp: dict, aux: dict, wb_on: jax.Array) -> tuple[dict, jax.Array]:
        params, rest, opt, step = carry['params'], carry['rest'], carry['opt'], carry['step']
        if proxy is not None:
            pS, pp, pr = aux['proxy']
            hard = per_example(pS.apply(pp, pr, xb, train=False)[0], yb, hp)
            idx = jax.lax.top_k(hard, keep)[1]
            xb, yb = xb[idx], yb[idx]

        def loss_fn(p: Any, r: Any) -> tuple[jax.Array, Any]:
            out, r2 = S.apply(p, r, xb, train=True)
            per = per_example(out, yb, hp)
            if teachers:
                from .methods import kd_weights

                T = hp['kd_temp']
                soft = sum(jax.nn.softmax(tS.apply(tp, tr_, xb, train=False)[0].astype(jnp.float32) / T, 1) for tS, tp, tr_ in aux['teachers']) / len(
                    teachers
                )
                soft = jax.lax.stop_gradient(soft)
                logp = jax.nn.log_softmax(out.astype(jnp.float32) / T, 1)
                kl = (soft * (jnp.log(jnp.maximum(soft, 1e-8)) - logp)).sum(1) * T**2
                per = (1 - hp['kd_weight']) * per + hp['kd_weight'] * kl
                if h['kd_weighting'] != 'none':  # (MT4)
                    per = per * kd_weights(kl, h['kd_weighting'])
            return (per.sum() if sgd_like else per.mean()), r2

        (loss, rest), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, rest)
        tx = optax.multi_transform(_transform(h, hp, bs, eps), label_fn)
        updates, opt = tx.update(grads, opt, params)
        f = factor(step, hp)
        updates = jax.tree.map(lambda u, w: u * f * (wb_on if w else 1.0), updates, is_wb)
        params = optax.apply_updates(params, updates)
        step = step + 1
        new = {'params': params, 'rest': rest, 'opt': opt, 'step': step}
        if h['lookahead']:
            ahead = carry['ahead']
            decay = 0.95**5 * (step.astype(jnp.float32) / total) ** 3
            on = (step % 5) == 0
            cur = (params, rest)
            ahead2 = jax.tree.map(lambda a, v: jnp.where(on, a + (1 - decay) * (v - a), a) if _floating(v) else a, ahead, cur)
            cur = jax.tree.map(lambda a, v: jnp.where(on | (step >= total), a, v) if _floating(v) else v, ahead2, cur)
            new.update(params=cur[0], rest=cur[1], ahead=ahead2)
        return new, loss

    def scan_epoch(
        carry: dict, ep: jax.Array, y: jax.Array, perm: jax.Array, hp: dict, arr: dict, wb_on: jax.Array
    ) -> tuple[dict, tuple[jax.Array, jax.Array]]:
        aux = {
            'proxy': (proxy[0], *arr['proxy']) if proxy else None,
            'teachers': [(t[0], *ta) for t, ta in zip(teachers, arr['teachers'], strict=True)],
        }

        def body(c: dict, idx: jax.Array) -> tuple[dict, tuple[jax.Array, jax.Array]]:
            active = c['step'] < total
            new, loss = jax.lax.cond(active, lambda: real_step(c, ep[idx], y[idx], hp, aux, wb_on), lambda: (c, jnp.zeros((), jnp.float32)))
            return new, (loss.astype(jnp.float32), active)

        return jax.lax.scan(body, carry, perm)

    epoch_fn = jax.jit(scan_epoch, donate_argnums=0)
    aux_arrays = {'proxy': (proxy[1], proxy[2]) if proxy else None, 'teachers': [(t[1], t[2]) for t in teachers]}

    def run_epoch(carry: dict, ep: jax.Array, perm: jax.Array, wb_on: bool) -> tuple[dict, tuple[jax.Array, jax.Array]]:
        return epoch_fn(carry, ep, y_dev, perm, hp, aux_arrays, jnp.asarray(wb_on, jnp.float32))

    def fwd(params: Any, rest: Any, x: jax.Array, level: int, gate: float) -> tuple[jax.Array, jax.Array]:
        apply = lambda v: S.apply(params, rest, v, train=False)[0]
        if gate > 0 and level > 0:
            from .methods import gated_logits

            return gated_logits(apply, x, level, gate, tta_logits)
        return tta_logits(apply, x, level), jnp.asarray(1.0 if level else 0.0)

    fwd_jit = jax.jit(fwd, static_argnums=(3, 4))

    def accuracy(carry: dict, x: np.ndarray, y: np.ndarray, level: int, gate: float = 0.0) -> tuple[float, float]:
        """(accuracy at ``level``'s views, the share of images given them: 1 unless MT6's ``gate`` is set).

        ``eval_batch`` images a forward pass (0: chosen from the device's memory, see ``eval_batch_size``). The batch
        only divides the work: every image's logits are computed from itself alone, so the value does not depend on it.
        """
        eb = precision.eval_batch_size(h, dev)
        correct, shares, nn_ = 0, 0.0, len(y)
        for a, b in jx.batches(nn_, eb):
            xb = jnp.asarray(x[a:b], dtype)
            if b - a < eb and nn_ > eb:
                xb = jnp.concatenate([xb, jnp.zeros((eb - (b - a), *xb.shape[1:]), xb.dtype)])
            z, share = fwd_jit(carry['params'], carry['rest'], xb, level, float(gate))
            z = np.asarray(z)[: b - a]
            correct += int((z.argmax(1) == np.asarray(y[a:b])).sum())
            shares += float(share) * (b - a)
        return correct / nn_, shares / nn_

    carry = {'params': S.params, 'rest': S.rest, 'opt': tx0.init(S.params), 'step': jnp.zeros((), jnp.int32)}
    if h['lookahead']:
        # (float arrays copied; the others are never read from it, a placeholder keeps the structure: the carry is donated)
        carry['ahead'] = jax.tree.map(lambda a: jnp.copy(a) if _floating(a) else jnp.zeros((), jnp.float32), (S.params, S.rest))
    st = {'history': [], 'pruned': False, 'key': key}
    n_epochs = math.ceil(h['epochs'])

    def run(carry: dict, epochs: Iterable[int]) -> tuple[dict, bool]:
        """Train over ``epochs`` (a range of epoch indices) of the schedule; (carry, False when the run should stop)."""
        for epoch in epochs:
            wb_on = whiten is not None and epoch < h['whiten_bias_epochs']
            st['key'], ke, kp = jax.random.split(st['key'], 3)
            ep = epoch_images(ke, jnp.asarray(epoch))
            perm = jax.random.permutation(kp, n)[: steps_per_epoch * bs].reshape(steps_per_epoch, bs)
            carry, (losses, active) = run_epoch(carry, ep, perm, wb_on)
            losses, active = np.asarray(losses), np.asarray(active)
            mean_loss = float(losses[active].sum() / max(1, active.sum()) / (bs if sgd_like else 1))
            st['history'].append(
                {
                    'epoch': epoch + 1,
                    'loss': round(mean_loss, 4),
                    'val_acc': round(accuracy(carry, x_val, y_val, 0)[0], 4),
                    'seconds': round(time.time() - t0, 1),
                }
            )
            # (a line an epoch: the job's log grows while it trains, so the queue's stall check means something)
            print(
                f'epoch {epoch + 1}/{n_epochs}: loss {mean_loss:.4f} val {st["history"][-1]["val_acc"]:.4f} {st["history"][-1]["seconds"]:.0f} s',
                flush=True,
            )
            step = int(carry['step'])
            if not math.isfinite(mean_loss):
                return carry, False
            if step < total and stop_here(round(100 * step / total), st['history'][-1]['val_acc']):
                st['pruned'] = True
                return carry, False
        return carry, True

    def copy(tree: Any) -> Any:
        """Fresh buffers for every array (random keys too): a copy that outlives the donated carry."""

        def one(a: Any) -> Any:
            if not hasattr(a, 'dtype'):
                return a
            if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
                return jax.random.wrap_key_data(jnp.copy(jax.random.key_data(a)), impl=jax.random.key_impl(a))
            return jnp.copy(a)

        return jax.tree.map(one, tree)

    if h['tails'] > 0:
        # MT2: a trunk over the schedule's first epochs, then ``tails`` continuations over its last ``tail_epochs``,
        # each from the trunk's state with its own augmentation draws, their weights averaged and batch norm's
        # statistics re-estimated; every tail's epochs count against the budget
        split = n_epochs - int(h['tail_epochs'])
        carry, ok = run(carry, range(split))
        trunk = copy(carry)
        finals = []
        for k in range(int(h['tails']) if ok else 0):
            carry = copy(trunk)
            st['key'] = jax.random.key(h['seed'] * 1000 + k + 1)
            carry, ok = run(carry, range(split, n_epochs))
            if not ok:
                break
            finals.append(copy((carry['params'], carry['rest'])))
        if finals:
            avg = jax.tree.map(lambda *v: sum(v) / len(v) if _floating(v[0]) else v[0], *finals)
            params, rest = avg
            # batch norm's statistics as a cumulative average over 20 training batches (Torch's momentum None): each
            # pass at momentum 0 gives its batch's statistics, which are then averaged
            tmp = S.merge(params, rest, train=True)
            for _, node in nnx.iter_graph(tmp):
                if isinstance(node, nnx.BatchNorm):
                    node.momentum = 0.0
            g0 = nnx.graphdef(tmp)

            @jax.jit
            def stats(params: Any, rest: Any, xb: jax.Array) -> Any:
                m = nnx.merge(g0, params, rest, copy=True)
                m(xb)
                return nnx.split(m, nnx.Param, ...)[2]

            passes = [stats(params, rest, base[i : i + bs]) for i in range(0, min(n, 20 * bs), bs)]
            rest = jax.tree.map(lambda *v: sum(v) / len(v) if _floating(v[0]) else v[0], *passes)
            carry = {**carry, 'params': params, 'rest': rest}
            st['history'].append(
                {'epoch': 'tails-averaged', 'val_acc': round(accuracy(carry, x_val, y_val, 0)[0], 4), 'seconds': round(time.time() - t0, 1)}
            )
    else:
        carry, _ = run(carry, range(n_epochs))
    jax.block_until_ready(carry['params'])
    step, history, pruned = int(carry['step']), st['history'], st['pruned']
    seconds = time.time() - t0
    charged = h['epochs'] + (max(0, int(h['tails']) - 1) * int(h['tail_epochs']) if h['tails'] else 0)
    charged *= h['filter_keep']  # (filtering: the images back-propagated, PLAN §2.1)
    common = {
        'train_seconds': round(seconds, 1),
        'images_per_s': round(step * bs / seconds),
        'device': jx.device_name(device),
        'params': counts['stored'],
        'params_trainable': counts['trainable'],
        'steps': step,
        'history': history,
        'hyperparameters': {**h, 'widths': list(h['widths'])},
        'epochs_charged': charged,
        **proxy_info,
    }
    if pruned:
        return {'val_acc': history[-1]['val_acc'], 'pruned': True, 'progress': round(100 * step / total), **common}
    val, val_share = accuracy(carry, x_val, y_val, h['tta_level'], h['tta_gate'])
    views = {0: 1, 1: 2, 2: 6, 3: 18}[h['tta_level']]
    out = {
        'val_acc': round(val, 4),
        'val_acc_notta': round(accuracy(carry, x_val, y_val, 0)[0], 4),
        # (1, 2 and 6 views of the same network: the equivariance-vs-TTA comparisons, PLAN §5.1)
        'val_acc_by_views': {v: round(accuracy(carry, x_val, y_val, lv)[0], 4) for lv, v in ((0, 1), (1, 2), (2, 6))},
    }
    if h['test'] == 'sealed':
        # (session 2026-10-30-equivariant-small: the test set is not evaluated during the session; final models only)
        share = val_share
    else:
        test, share = accuracy(carry, x_test, y_test, h['tta_level'], h['tta_gate'])
        out.update(test_acc=round(test, 4), test_acc_notta=round(accuracy(carry, x_test, y_test, 0)[0], 4))
    out.update(views_per_image=round(1 + views * share, 3) if h['tta_gate'] > 0 and h['tta_level'] else views, **common)
    if save:
        out['saved'] = str(save_model(S.merge(carry['params'], carry['rest']), h, save))
    return out


def save_model(net: nnx.Module, h: dict, directory: str) -> Path:
    """The network's arrays (``model.safetensors``, float32: parameters, batch statistics) and the hyperparameters that
    rebuild it (``config.json``)."""
    from safetensors.numpy import save_file

    from .hub import flat_weights

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    save_file(flat_weights(net), str(d / 'model.safetensors'))
    cfg = {
        'family': 'airbench',
        **h,
        'widths': list(h['widths']),
        'filter_proxy_widths': list(h['filter_proxy_widths']),
        'arch_kw': dict(h.get('arch_kw', {})),
    }
    (d / 'config.json').write_text(json.dumps(cfg, indent=1) + '\n')
    return d


def load(directory: str | Path, device: str = 'cpu', dtype: DTypeLike | None = None) -> tuple[nnx.Module, dict]:
    """(network in eval mode, float32 weights, ``dtype`` its arithmetic (float32 by default), its hyperparameters) from
    ``save_model``'s directory. ``device`` is where ``Predictor`` runs it (kept for the Torch signature)."""
    from safetensors.numpy import load_file

    from .hub import load_weights

    d = Path(directory)
    h = json.loads((d / 'config.json').read_text())
    net, _ = build(h, dtype)
    load_weights(net, load_file(str(d / 'model.safetensors')))
    net.eval()
    return net, h


class Predictor:
    """The inference contract the footprint bench measures: ``predict(uint8 (n, 32, 32, 3)) -> probabilities (n, 10)``,
    normalisation and TTA included."""

    def __init__(self, directory: str | Path, device: str = 'cpu', dtype: DTypeLike | None = None, tta_level: int | None = None) -> None:
        import jax
        import jax.numpy as jnp

        from . import jx

        self.device = jx.pick(device)
        self.dtype = jnp.dtype(dtype) if dtype is not None else jnp.float32
        self.net, self.h = load(directory, device, self.dtype)
        self.split = jx.Split(self.net)
        self.params, self.rest = jax.device_put((self.split.params, self.split.rest), self.device)
        self.tta = self.h['tta_level'] if tta_level is None else tta_level
        gate = float(self.h.get('tta_gate', 0))
        S, tta = self.split, self.tta

        def fwd(params: Any, rest: Any, images: jax.Array) -> jax.Array:
            x = jx.normalise(images, MEAN, STD).astype(self.dtype)
            apply = lambda v: S.apply(params, rest, v, train=False)[0]
            if gate > 0 and tta:  # (MT6: the full views only where the margin is small)
                from .methods import gated_logits

                logits = gated_logits(apply, x, tta, gate, tta_logits)[0]
            else:
                logits = tta_logits(apply, x, tta)
            return jax.nn.softmax(logits.astype(jnp.float32), 1)

        self._fwd = jax.jit(fwd)

    def predict(self, images: NPArrayLike) -> np.ndarray:
        import jax
        import numpy as np

        x = jax.device_put(np.ascontiguousarray(images, dtype=np.uint8), self.device)
        return np.asarray(self._fwd(self.params, self.rest, x))
