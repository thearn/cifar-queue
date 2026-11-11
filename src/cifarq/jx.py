"""The JAX pieces every trainer here shares (``train``, ``airbench``, ``pbt``): layers initialised as PyTorch's are (the
recipes and their tuned hyperparameters were found with those initialisations), batch norm with PyTorch's momentum
convention, a frozen variable for fixed filters, the per-image affine resample of ``grid_sample``, the optimisers with
decoupled or coupled weight decay masked off biases and batch norm, a weight average, and the functional split of a
Flax NNX network into a static graph (one for training, one for evaluation) and its arrays, so a training step is a
pure function that ``jax.jit``, ``jax.lax.scan`` and ``jax.vmap`` (several seeds at once) take as it is.

Images are NHWC throughout (JAX's native layout). Convolutions and linear layers compute in bfloat16 with float32
weights, batch norm and the loss in float32, the counterpart of the Torch code's bf16 autocast.

Compilation: a trial compiles its step once per process; the persistent cache (``cache``) keeps the compiled programs
across processes on a machine, so a search's trials that share a structure (width, batch, switches) compile once.
Float hyperparameters are passed traced (``dynamic``), so they never cause a compile.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

if TYPE_CHECKING:
    from typing import Any

    from jax.typing import ArrayLike

COMPUTE = jnp.bfloat16


def cache(path: str | None = None) -> None:
    """Turn on JAX's persistent compilation cache (``JAX_COMPILATION_CACHE_DIR``, else ``~/.cache/cifarq-jax``)."""
    path = path or os.environ.get('JAX_COMPILATION_CACHE_DIR') or os.path.expanduser('~/.cache/cifarq-jax')
    jax.config.update('jax_compilation_cache_dir', path)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)


def device_name(device: str = 'auto') -> str:
    """The name of the device a computation runs on (``cpu`` or the GPU's model)."""
    d = pick(device)
    return getattr(d, 'device_kind', 'cpu') if d.platform == 'gpu' else 'cpu'


def pick(device: str = 'auto') -> jax.Device:
    """A JAX device: ``auto`` the first GPU when there is one, else the CPU; ``cuda``/``gpu`` or ``cpu`` as named."""
    if device == 'auto':
        return jax.devices()[0]
    return jax.devices('cpu' if device == 'cpu' else 'gpu')[0]


# ---- layers --------------------------------------------------------------------------------------------------


def torch_kernel_init() -> Callable:
    """PyTorch's default for convolutions and linear layers: kaiming-uniform with a = sqrt(5), i.e. uniform within
    1 / sqrt(fan_in)."""
    return nnx.initializers.variance_scaling(1 / 3, 'fan_in', 'uniform')


def torch_bias_init(fan_in: int) -> Callable:
    bound = 1 / math.sqrt(fan_in) if fan_in else 0.0
    return lambda key, shape, dtype=jnp.float32: jax.random.uniform(key, shape, dtype, -bound, bound)


def conv(
    cin: int, cout: int, k: int = 3, *, stride: int = 1, padding: int | str = 'SAME', bias: bool = False, groups: int = 1, rngs: nnx.Rngs
) -> nnx.Conv:
    """A convolution initialised as ``torch.nn.Conv2d``; ``padding`` an int is symmetric padding of that many pixels."""
    pad = [(padding, padding)] * 2 if isinstance(padding, int) else padding
    return nnx.Conv(
        cin,
        cout,
        (k, k),
        strides=stride,
        padding=pad,
        use_bias=bias,
        feature_group_count=groups,
        kernel_init=torch_kernel_init(),
        bias_init=torch_bias_init(cin // groups * k * k),
        dtype=COMPUTE,
        rngs=rngs,
    )


def linear(cin: int, cout: int, *, bias: bool = True, rngs: nnx.Rngs) -> nnx.Linear:
    """A linear layer initialised as ``torch.nn.Linear``."""
    return nnx.Linear(cin, cout, use_bias=bias, kernel_init=torch_kernel_init(), bias_init=torch_bias_init(cin), dtype=COMPUTE, rngs=rngs)


def batchnorm(c: int, *, momentum: float = 0.1, eps: float = 1e-5, affine: bool = True, scale: bool = True, rngs: nnx.Rngs) -> nnx.BatchNorm:
    """Batch norm with PyTorch's conventions: ``momentum`` the weight of the new batch (Flax's is the old's), output
    in float32. ``scale``: whether the scale is a learned parameter (airbench freezes it at 1)."""
    return nnx.BatchNorm(c, momentum=1 - momentum, epsilon=eps, use_bias=affine, use_scale=affine and scale, dtype=jnp.float32, rngs=rngs)


class Frozen(nnx.Variable):
    """An array that is part of the network but not trained (a whitening filter bank): kept out of ``nnx.Param`` so
    gradients, optimisers and weight decay never see it."""


class FixedConv(nnx.Module):
    """A convolution with fixed weights (HWIO), no bias unless ``bias`` (a trained ``nnx.Param``, zeros)."""

    def __init__(self, kernel: np.ndarray, *, padding: int | str = 'VALID', bias: bool = False) -> None:
        self.kernel = Frozen(jnp.asarray(kernel, jnp.float32))
        self.padding = [(padding, padding)] * 2 if isinstance(padding, int) else padding
        self.bias = nnx.Param(jnp.zeros(kernel.shape[-1], jnp.float32)) if bias else None

    def __call__(self, x: jax.Array) -> jax.Array:
        y = jax.lax.conv_general_dilated(
            x.astype(COMPUTE), self.kernel[...].astype(COMPUTE), (1, 1), self.padding, dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        return y + self.bias[...].astype(y.dtype) if self.bias is not None else y


def activation(name: str, celu_alpha: float = 0.075) -> Callable:
    return {
        'relu': jax.nn.relu,
        'gelu': lambda x: jax.nn.gelu(x, approximate=False),
        'silu': jax.nn.silu,
        'celu': lambda x: jax.nn.celu(x, alpha=celu_alpha),
    }[name]


def max_pool(x: jax.Array, k: int = 2) -> jax.Array:
    return nnx.max_pool(x, (k, k), strides=(k, k))


def global_max(x: jax.Array) -> jax.Array:
    return x.max(axis=(1, 2))


# ---- the functional split -----------------------------------------------------------------------------------


class Split:
    """A network as static graphs and arrays: ``train_graph`` and ``eval_graph`` (batch norm on batch statistics and
    dropout on, or running statistics and dropout off), ``params`` (``nnx.Param``: what the optimiser updates) and
    ``rest`` (batch statistics, frozen filters, random-number state). ``apply(params, rest, x, train)`` returns the
    logits and the updated ``rest``; ``merge`` rebuilds the module (for saving)."""

    def __init__(self, net: nnx.Module) -> None:
        net.train()
        self.frozen = count(nnx.state(net, Frozen))  # (counted as parameters, as the Torch model counted them)
        self.train_graph, self.params, self.rest = nnx.split(net, nnx.Param, ...)
        net.eval()
        self.eval_graph = nnx.graphdef(net)
        net.train()

    def apply(self, params: nnx.State, rest: nnx.State, x: jax.Array, train: bool = True) -> tuple[jax.Array, nnx.State]:
        net = nnx.merge(
            self.train_graph if train else self.eval_graph, params, rest, copy=True
        )  # (copy: else batch norm's update fails under jax.grad)
        out = net(x)
        _, _, rest = nnx.split(net, nnx.Param, ...)
        return out, rest

    def merge(self, params: nnx.State, rest: nnx.State, train: bool = False) -> nnx.Module:
        return nnx.merge(self.train_graph if train else self.eval_graph, params, rest, copy=True)


def count(tree: Any) -> int:
    return int(sum(np.prod(a.shape) for a in jax.tree.leaves(tree)))


def decay_mask(params: Any) -> Any:
    """True for weights with two or more dimensions (convolution and linear kernels): weight decay applies to them only,
    not to biases or batch norm, as the Torch parameter groups did."""
    return jax.tree.map(lambda a: a.ndim > 1, params)


# ---- optimisers ----------------------------------------------------------------------------------------------


def dynamic(h: dict, names: Iterable[str]) -> dict:
    """The float hyperparameters ``names`` of ``h`` as float32 arrays: passed to a jitted function as arguments, so a
    change of value is not a change of program."""
    return {k: jnp.asarray(float(h[k]), jnp.float32) for k in names if k in h}


def warmup_decay(step: ArrayLike, total: int, hp: dict, decay: str) -> jax.Array:
    """The learning rate at ``step`` of ``total`` (traced): a linear warm-up over ``warmup_frac`` of the steps from
    ``warmup_start_frac`` of the peak ``lr``, then ``decay`` (cosine, linear, exponential, none) to ``final_lr_frac``
    of it. ``decay`` is static; the fractions may be traced."""
    step = jnp.asarray(step, jnp.float32)
    warm = jnp.floor(hp['warmup_frac'] * total)
    up = hp['warmup_start_frac'] + (1 - hp['warmup_start_frac']) * step / jnp.maximum(1.0, warm)
    t = (step - warm) / jnp.maximum(1.0, total - warm)
    end = hp['final_lr_frac']
    down = {
        'cosine': lambda: end + (1 - end) * 0.5 * (1 + jnp.cos(jnp.pi * t)),
        'linear': lambda: 1 - (1 - end) * t,
        'exponential': lambda: end**t,
        'none': lambda: jnp.ones_like(t),
    }[decay]()
    return hp['lr'] * jnp.where(step < warm, up, down)


OPTIMIZER_FLOATS = ('lr', 'momentum', 'beta1', 'beta2', 'weight_decay', 'grad_clip', 'warmup_frac', 'warmup_start_frac', 'final_lr_frac')


def optimizer(h: dict, hp: dict, params: Any, total: int, schedule: Callable | None = None) -> tuple[optax.GradientTransformation, bool]:
    """The optimiser of ``h`` (``optimizer``: sgd, adamw, sgd_sf, adamw_sf) with its weight decay masked off biases and
    batch norm, gradient clipping when ``grad_clip``; and whether it is schedule-free (its weights for evaluation are
    then ``eval_params``). ``h`` (Python values) makes the structural choices; the values come from ``hp`` (``dynamic``
    of ``OPTIMIZER_FLOATS``), which may be traced: build the optimiser inside the jitted step from its ``hp`` argument
    and its ``init`` outside with the same ``h``, and a new learning rate is the same compiled program.
    ``schedule``: step -> learning rate (default ``warmup_decay`` with ``h['decay']``); unused when schedule-free.

    sgd: PyTorch's SGD (L2 weight decay added to the gradient, then momentum, Nesterov when on); adamw: decoupled;
    the schedule-free pair (Defazio et al., 2024) as ``schedulefree``'s, a linear warm-up from 0 over ``warmup_frac``
    of the steps and no decay."""
    mask = decay_mask(params)
    wd, name = hp['weight_decay'], h['optimizer']
    clip = [optax.clip_by_global_norm(hp['grad_clip'])] if h.get('grad_clip') else []
    if name.endswith('_sf'):
        warm = jnp.maximum(1.0, jnp.floor(hp['warmup_frac'] * total))
        lr = lambda step: hp['lr'] * jnp.minimum(1.0, step / warm)
        if name == 'sgd_sf':
            base = optax.chain(optax.add_decayed_weights(wd, mask), optax.scale_by_learning_rate(lr))
            b1 = hp['momentum'] if h.get('momentum') else 0.9
        else:
            base = optax.chain(
                optax.scale_by_rms(decay=hp['beta2'], eps=1e-8, eps_in_sqrt=False, bias_correction=True),
                optax.add_decayed_weights(wd, mask),
                optax.scale_by_learning_rate(lr),
            )
            b1 = hp['beta1']
        return optax.chain(*clip, optax.contrib.schedule_free(base, learning_rate=lr, b1=b1)), True
    schedule = schedule or (lambda step: warmup_decay(step, total, hp, h['decay']))
    if name == 'sgd':
        on = float(h['momentum']) > 0
        core = optax.chain(
            optax.add_decayed_weights(wd, mask),
            optax.sgd(schedule, momentum=hp['momentum'] if on else None, nesterov=bool(h.get('nesterov', False)) and on),
        )
    elif name == 'adamw':
        core = optax.adamw(schedule, b1=hp['beta1'], b2=hp['beta2'], eps=1e-8, weight_decay=wd, mask=mask)
    else:
        raise ValueError(f'unknown optimizer {name!r}')
    return optax.chain(*clip, core), False


class HostKey:
    """A typed PRNG key on the host: its key data and implementation, for ``to_host``/``from_host``."""

    def __init__(self, data: np.ndarray, impl: str) -> None:
        self.data, self.impl = data, impl


def to_host(tree: Any) -> Any:
    """A pytree of device arrays as NumPy arrays (a checkpoint that pickles); typed PRNG keys as ``HostKey``."""

    def one(a: jax.Array) -> np.ndarray | HostKey:
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            return HostKey(np.asarray(jax.random.key_data(a)), str(jax.random.key_impl(a)))
        return np.asarray(a)

    return jax.tree.map(one, tree)


def from_host(tree: Any, device: jax.Device | None = None) -> Any:
    """``to_host``'s tree back on ``device``."""

    def one(a: Any) -> jax.Array:
        if isinstance(a, HostKey):
            return jax.device_put(jax.random.wrap_key_data(a.data, impl=a.impl), device)
        return jax.device_put(a, device)

    return jax.tree.map(one, tree, is_leaf=lambda a: isinstance(a, HostKey))


def eval_params(opt_state: Any, params: Any, schedule_free: bool) -> Any:
    """The weights to evaluate: the schedule-free average where the optimiser keeps one, else the weights."""
    if not schedule_free:
        return params
    sf = next(
        s
        for s in jax.tree.leaves(opt_state, is_leaf=lambda s: isinstance(s, optax.contrib.ScheduleFreeState))
        if isinstance(s, optax.contrib.ScheduleFreeState)
    )
    return optax.contrib.schedule_free_eval_params(sf, params)


def ema_update(avg: Any, new: Any, n: ArrayLike, decay: float) -> Any:
    """One step of a weight average with its decay warmed up (``min(decay, (1 + n) / (10 + n))`` at update n):
    without the warm-up a short run's average is its first, untrained weights (0.13 validation accuracy at 0.999 over
    350 steps, 2026-10-27). Floating arrays are averaged, others copied."""
    d = jnp.minimum(decay, (1.0 + n) / (10.0 + n))
    return jax.tree.map(lambda a, b: d * a + (1 - d) * b if jnp.issubdtype(a.dtype, jnp.floating) else b, avg, new)


def cross_entropy(logits: jax.Array, y: jax.Array, classes: int, smoothing: ArrayLike = 0.0) -> jax.Array:
    """Mean cross-entropy with label smoothing (PyTorch's: ``smoothing`` spread over all classes)."""
    z = logits.astype(jnp.float32)
    target = jax.nn.one_hot(y, classes) * (1 - smoothing) + smoothing / classes
    return optax.softmax_cross_entropy(z, target).mean()


# ---- images --------------------------------------------------------------------------------------------------


def normalise(x_uint8: ArrayLike, mean: ArrayLike, std: ArrayLike) -> jax.Array:
    """uint8 NHWC images to float32, per-channel standardised."""
    return (x_uint8.astype(jnp.float32) / 255 - jnp.asarray(mean)) / jnp.asarray(std)


def affine(x: jax.Array, theta: jax.Array) -> jax.Array:
    """``torch.nn.functional.grid_sample(x, affine_grid(theta))``, bilinear, reflection padding, align_corners off:
    each image of ``x`` (N, H, W, C) resampled through its own 2x3 ``theta`` (N, 2, 3), which maps output to input
    positions in [-1, 1] coordinates (x along the width)."""
    n, hgt, wid, c = x.shape
    xs = (2 * jnp.arange(wid) + 1) / wid - 1
    ys = (2 * jnp.arange(hgt) + 1) / hgt - 1
    gx, gy = jnp.meshgrid(xs, ys)  # (H, W)
    base = jnp.stack([gx, gy, jnp.ones_like(gx)], -1)  # (H, W, 3)
    grid = jnp.einsum('hwk,nik->nhwi', base, theta)  # (N, H, W, 2): x, y
    col = ((grid[..., 0] + 1) * wid - 1) / 2
    row = ((grid[..., 1] + 1) * hgt - 1) / 2

    def one(img: jax.Array, r: jax.Array, cl: jax.Array) -> jax.Array:
        return jax.vmap(lambda ch: jax.scipy.ndimage.map_coordinates(ch, [r, cl], order=1, mode='reflect'), in_axes=2, out_axes=2)(img)

    return jax.vmap(one)(x, row, col)


def patches(x: np.ndarray, k: int) -> np.ndarray:
    """Every unpadded k x k patch of NHWC images as rows (n * (H-k+1) * (W-k+1), k * k * C), in (dy, dx, c) order,
    so a row reshapes to an HWC filter."""
    n, hgt, wid, c = x.shape
    rows = [x[:, dy : hgt - k + 1 + dy, dx : wid - k + 1 + dx, :] for dy in range(k) for dx in range(k)]
    return np.stack(rows, 3).reshape(-1, k * k * c)


def whitening_kernel(x: ArrayLike, k: int, eps: float, negatives: bool = False) -> np.ndarray:
    """The patch-whitening filter bank of standardised images ``x`` (NHWC, NumPy): the eigenvectors of their k x k
    patches' covariance, each divided by the square root of its eigenvalue plus ``eps``, as an HWIO kernel
    (k, k, C, k*k*C); with ``negatives`` the filters' negatives appended (airbench). Computed in float64."""
    p = patches(np.asarray(x, np.float64), k)
    p = p - p.mean(0)
    cov = p.T @ p / len(p)
    vals, vecs = np.linalg.eigh(cov)
    filt = (vecs.T / np.sqrt(vals + eps)[:, None]).reshape(-1, k, k, x.shape[-1])  # (O, kh, kw, C)
    if negatives:
        filt = np.concatenate([filt, -filt])
    return np.transpose(filt, (1, 2, 3, 0)).astype(np.float32)


def batches(n: int, size: int) -> list[tuple[int, int]]:
    """(start, stop) pairs covering ``n`` rows in ``size`` blocks."""
    return [(i, min(n, i + size)) for i in range(0, n, size)]


def padded_apply(fn: Callable[[jax.Array], jax.Array], x: jax.Array, size: int) -> jax.Array:
    """``fn`` over ``x`` in blocks of ``size`` rows, the last block padded to ``size`` (one compiled shape), the
    outputs concatenated and trimmed."""
    out = []
    for a, b in batches(len(x), size):
        xb = x[a:b]
        if b - a < size:
            xb = jnp.concatenate([xb, jnp.zeros((size - (b - a), *xb.shape[1:]), xb.dtype)])
        out.append(fn(xb)[: b - a])
    return jnp.concatenate(out)
