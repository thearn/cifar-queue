"""The inference footprint and runtime of a saved model, measured the same way for every family (the autoresearch
session's evaluator, docs/autoresearch/2026-10-29-footprint-arch/PLAN.md section 2.2).

A model is a saved directory: ``config.json`` names its family (``airbench``; no ``family`` is a ``hub.ResNet9``), and
``predictor(directory, device)`` gives an object whose ``predict(uint8 (n, 32, 32, 3))`` returns class probabilities
(n, 10), normalisation and test-time augmentation included. ``static`` counts its parameters, stored bytes and
multiply-accumulates per image; ``measure`` runs one setting (device, threads, batch) in a fresh process so memory
peaks start from the interpreter's floor; ``bench`` does the whole table. Run on the local machine only, so timings
compare.

MACs are read from the network's traced program (``count_macs``): every convolution and matrix product that combines
the input with the network's arrays, so expanded equivariant kernels count as the convolution they run, while
products of weights alone (structured-receptive-field kernels built from their coefficients) and convolutions with
constant filters (the blur pool's) are not counted, as the Torch counter's module hooks did not count them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import jax
    import numpy as np
    from flax import nnx
    from jax.extend import core
    from numpy.typing import ArrayLike

    from .airbench import Predictor


def family(directory: str | Path) -> str:
    cfg = json.loads((Path(directory) / 'config.json').read_text())
    return cfg.get('family', 'resnet9')


class _ResNet9Predictor:
    def __init__(self, directory: str | Path, device: str = 'cpu', dtype: str | None = None) -> None:
        from .hub import ResNet9

        self.net = ResNet9.from_pretrained(str(directory))
        self.views = 2 if self.net.tta_flip else 1
        self.device = device

    def predict(self, images: ArrayLike) -> np.ndarray:
        import numpy as np

        return self.net.predict(np.asarray(images, dtype=np.uint8))


def predictor(directory: str | Path, device: str = 'cpu', dtype: str | None = None) -> _ResNet9Predictor | Predictor:
    fam = family(directory)
    if fam == 'airbench':
        from .airbench import Predictor

        return Predictor(directory, device, dtype)
    if fam == 'resnet9':
        return _ResNet9Predictor(directory, device, dtype)
    raise ValueError(f'unknown family {fam!r}')


def views(directory: str | Path) -> int:
    """Forward passes per image at inference (test-time augmentation's views)."""
    cfg = json.loads((Path(directory) / 'config.json').read_text())
    if family(directory) == 'resnet9':
        return 2 if cfg.get('tta_flip') else 1
    return {0: 1, 1: 2, 2: 6, 3: 18}[int(cfg.get('tta_level', 0))]  # (level 3: added 2026-10-30, P2)


def _macs(jaxpr: core.Jaxpr, depends: dict[Any, set[str]]) -> int:
    """MACs of the equations of ``jaxpr`` whose operands mix the input with the arrays: ``depends`` maps each invar to
    its set of sources ({'x'}, {'w'} or both), propagated through every equation (sub-programs recursively)."""
    from jax.extend import core

    src = dict(depends)
    get = lambda v: src.get(v, set()) if not isinstance(v, core.Literal) else set()
    total = 0
    for eqn in jaxpr.eqns:
        ins = [get(v) for v in eqn.invars]
        name = eqn.primitive.name
        if name in ('conv_general_dilated', 'dot_general') and len(ins) >= 2 and ({'x'} <= ins[0] | ins[1]) and ins[0] and ins[1]:
            mixed = ('x' in ins[0] and 'w' in ins[1]) or ('w' in ins[0] and 'x' in ins[1])
            if mixed:
                out = eqn.outvars[0].aval
                lhs, rhs = eqn.invars[0].aval, eqn.invars[1].aval
                if name == 'conv_general_dilated':
                    o = eqn.params['dimension_numbers'].rhs_spec[0]
                    per = 1
                    for i, d in enumerate(rhs.shape):
                        if i != o:
                            per *= d
                    total += int(out.size) * per
                else:
                    (lc, _), _ = eqn.params['dimension_numbers']
                    per = 1
                    for i in lc:
                        per *= lhs.shape[i]
                    total += int(out.size) * per
        for p in eqn.params.values():
            sub = getattr(p, 'jaxpr', p)
            if hasattr(sub, 'eqns') and hasattr(sub, 'invars') and len(sub.invars) == len(eqn.invars):
                total += _macs(sub, dict(zip(sub.invars, ins, strict=True)))
        union = set().union(*ins) if ins else set()
        for v in eqn.outvars:
            src[v] = union
    return total


def count_macs(net: nnx.Module, shape: tuple[int, ...] = (1, 32, 32, 3)) -> int:
    """Multiply-accumulates of one forward pass of ``net`` (a Flax NNX module, evaluation mode) on an NHWC input of
    ``shape``, per image: convolutions (grouped ones included) and matrix products of the input with the network's
    arrays, which carry all but a negligible share of the arithmetic here. Traced only: nothing runs (an abstract
    network from ``nnx.eval_shape`` works)."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    net.eval()
    graph, state = nnx.split(net)

    def f(state: nnx.State, x: jax.Array) -> jax.Array:
        return nnx.merge(graph, state, copy=True)(x)

    spec = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), state)
    closed = jax.make_jaxpr(f)(spec, jax.ShapeDtypeStruct(shape, jnp.float32))
    jaxpr = closed.jaxpr
    n_state = len(jax.tree.leaves(spec))
    depends = {v: ({'w'} if i < n_state else {'x'}) for i, v in enumerate(jaxpr.invars)}
    for v in jaxpr.constvars:
        depends[v] = set()
    return _macs(jaxpr, depends) // shape[0]


def static(directory: str | Path) -> dict:
    """Parameters (stored: every parameter array, frozen ones included; trainable), stored bytes (every file of the
    directory the predictor needs), and MACs per image (one view, and times the views)."""
    from flax import nnx

    from . import jx

    d = Path(directory)
    p = predictor(d)
    if family(d) == 'resnet9':
        net = p.net.net
        frozen = jx.count(nnx.state(net, jx.Frozen))
        stored = jx.count(nnx.state(net, nnx.Param)) + frozen
        trainable = stored - frozen
    else:
        from .airbench import build, param_counts

        net = p.net
        c = param_counts(build(p.h, abstract=True)[0], p.h.get('bn_scale', False))
        stored, trainable = c['stored'], c['trainable']
    macs = count_macs(net)
    v = views(d)
    return {
        'family': family(d),
        'params_stored': stored,
        'params_trainable': trainable,
        'bytes': sum(f.stat().st_size for f in (d / 'model.safetensors', d / 'config.json')),
        'macs_per_view': macs,
        'views': v,
        'macs_per_image': macs * v,
    }


class _Null:
    """The null model of the memory baseline: the same inputs, a constant output, no network."""

    def __init__(self, device: str) -> None:
        from . import jx

        self.device = jx.pick(device)

    def predict(self, images: ArrayLike) -> np.ndarray:
        import jax
        import jax.numpy as jnp
        import numpy as np

        x = jax.device_put(np.ascontiguousarray(images), self.device).astype(jnp.float32)
        return np.asarray(jnp.full((len(images), 10), 0.1) + 0 * x.mean())


def _vmrss_mb() -> float:
    """This process's resident memory now (Linux ``/proc``), MB."""
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return round(int(line.split()[1]) / 1024, 1)
    return 0.0


def _measure(directory: str, device: str, threads: int, batch: int, warmup: int, timed: int) -> dict:
    """One setting, in this (fresh) process: the process's peak memory (RSS, and the GPU's peak bytes in use), and
    latency or throughput. ``directory`` ``null`` measures the baseline (``_Null``), which ``measure`` subtracts.
    ``threads`` is set by ``_fresh`` (XLA's CPU thread flags, before JAX starts)."""
    import resource
    import time

    import jax
    import numpy as np

    from . import jx

    dev = jx.pick(device)
    gpu = dev.platform == 'gpu'
    dtype = 'bfloat16' if gpu else None
    p = _Null(device) if directory == 'null' else predictor(directory, device, dtype)
    images = np.random.default_rng(0).integers(0, 256, size=(batch, 32, 32, 3), dtype=np.uint8)
    for _ in range(warmup):
        p.predict(images)
    times = []
    for _ in range(timed):
        t = time.perf_counter()
        p.predict(images)  # (returns NumPy: synchronised)
        times.append(time.perf_counter() - t)
    times = np.asarray(times)
    out = {
        'device': device,
        'threads': threads or os.cpu_count(),
        'batch': batch,
        'p50_ms': round(1e3 * float(np.median(times)), 3),
        'p95_ms': round(1e3 * float(np.percentile(times, 95)), 3),
        'images_per_s': round(batch / float(np.median(times)), 1),
        'peak_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        'rss_mb': _vmrss_mb(),
    }
    if gpu:
        stats = dev.memory_stats() or {}
        out['peak_gpu_mb'] = round(stats.get('peak_bytes_in_use', 0) / 2**20, 1)
        out['gpu'] = dev.device_kind
    del jax
    return out


def _fresh(directory: str | Path, device: str, threads: int, batch: int, warmup: int, timed: int) -> dict:
    env = dict(os.environ)
    if device == 'cpu':
        env['JAX_PLATFORMS'] = 'cpu'
        if threads:
            env['XLA_FLAGS'] = (env.get('XLA_FLAGS', '') + f' --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={threads}').strip()
    cmd = [sys.executable, '-m', 'cifarq.footprint', str(directory), device, str(threads), str(batch), str(warmup), str(timed)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


def measure(directory: str | Path, device: str = 'cpu', threads: int = 0, batch: int = 1, warmup: int = 20, timed: int = 200) -> dict:
    """``_measure`` in a fresh interpreter, and the null model's in another: peak memory is the model's process peak
    minus the null's (the same imports, inputs and device runtime: what the model itself adds), latency the model's."""
    r = _fresh(directory, device, threads, batch, warmup, timed)
    null = _fresh('null', device, threads, batch, 2, 3)
    r['peak_rss_total_mb'], r['rss_total_mb'] = r['peak_rss_mb'], r['rss_mb']
    r['peak_rss_mb'] = round(max(0.0, r['peak_rss_mb'] - null['peak_rss_mb']), 1)
    r['rss_mb'] = round(max(0.0, r['rss_mb'] - null['rss_mb']), 1)  # (resident after the timed loop: what it holds)
    if 'peak_gpu_mb' in r:
        r['peak_gpu_total_mb'] = r['peak_gpu_mb']
        r['peak_gpu_mb'] = round(max(0.0, r['peak_gpu_mb'] - null.get('peak_gpu_mb', 0.0)), 1)
    return r


SETTINGS = [  # (name, device, threads, batch, warmup, timed)
    ('gpu_b1', 'cuda', 0, 1, 200, 500),
    ('gpu_b256', 'cuda', 0, 256, 5, 30),
    ('gpu_b1024', 'cuda', 0, 1024, 3, 20),
    ('cpu1_b1', 'cpu', 1, 1, 30, 300),
    ('cpuall_b1', 'cpu', 0, 1, 30, 300),
    ('cpuall_b256', 'cpu', 0, 256, 2, 10),
]


def _gpu_present() -> bool:
    try:
        import jax

        return bool(jax.devices('gpu'))
    except RuntimeError:
        return False


def bench(directory: str | Path, settings: list[tuple[str, str, int, int, int, int]] | None = None, gpu: bool = True) -> dict:
    """The whole table: ``static`` and every setting of ``SETTINGS`` (the GPU ones only when ``gpu`` and a GPU is there)."""
    out = {'model': str(directory), **static(directory), 'runs': {}}
    has_gpu = gpu and _gpu_present()
    for name, device, threads, batch, warmup, timed in settings or SETTINGS:
        if device == 'cuda' and not has_gpu:
            continue
        out['runs'][name] = measure(directory, device, threads, batch, warmup, timed)
    return out


def classify(row: dict, ref: dict, param_cap: int | None = None) -> str:
    """'S' (strict: at most ``ref`` on parameters and bytes, at most 1.05 x on every peak memory reading), 'P'
    (parameter-matched: at most on parameters and bytes) or 'out'. ``param_cap`` replaces ``ref``'s parameters as the
    bound (Goal A: 2,000,000)."""
    cap = param_cap if param_cap is not None else ref['params_stored']
    if row['params_stored'] > cap or row['bytes'] > max(ref['bytes'], 4 * cap + 4096):
        return 'out'
    for name, r in row['runs'].items():
        base = ref['runs'].get(name)
        if base is None:
            continue
        # GPU: peak allocations, deterministic (+1 MB: granularity). CPU: resident memory at batch 1 only, +5 MB, the
        # spread of repeats (E0.1-a3: 43 +- 3 MB for B0; at batch 256 the allocator's per-thread caches vary 17 %)
        if 'peak_gpu_mb' in r and 'peak_gpu_mb' in base and r['peak_gpu_mb'] > 1.05 * base['peak_gpu_mb'] + 1.0:
            return 'P'
        if name.endswith('_b1') and 'rss_mb' in r and 'rss_mb' in base and r['rss_mb'] > 1.05 * base['rss_mb'] + 5.0:
            return 'P'
    return 'S'


if __name__ == '__main__':
    d, device, threads, batch, warmup, timed = sys.argv[1:7]
    print(json.dumps(_measure(d, device, int(threads), int(batch), int(warmup), int(timed))))
