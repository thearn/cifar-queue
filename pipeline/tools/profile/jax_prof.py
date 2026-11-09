"""JAX side: python jax_prof.py CONFIG [e2e|components|trace] [--tag T] [--xla FLAGS] [--set k=v ...]. Run with
~/.venvs/cifarq-jax/bin/python. Writes results/profile/jax-<config>-<mode>[-tag].json. ``--xla`` sets XLA_FLAGS before
JAX loads (a variant to compare); ``--set`` overrides hyperparameters."""

import argparse
import gzip
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from configs import CONFIGS, DATA, write  # noqa: E402

N_WARM, N_TIME = 10, 50


def gpu_kernels(events, top=25):
    """Kernel time by name from a JAX Perfetto trace: X events in the processes named after a GPU stream."""
    gpu = {e['pid'] for e in events if e.get('ph') == 'M' and e.get('name') == 'process_name' and 'GPU' in str(e.get('args', {}).get('name', ''))}
    agg = {}
    for e in events:
        if e.get('ph') == 'X' and e.get('pid') in gpu and 'dur' in e:
            a = agg.setdefault(str(e.get('name', ''))[:120], [0.0, 0])
            a[0] += float(e['dur'])
            a[1] += 1
    total = sum(v[0] for v in agg.values()) or 1.0
    rows = sorted(
        ({'kernel': k, 'ms': round(v[0] / 1e3, 2), 'calls': v[1], 'share': round(v[0] / total, 4)} for k, v in agg.items()), key=lambda r: -r['ms']
    )
    return [{'total_kernel_ms': round(total / 1e3, 1)}] + rows[:top]


def run_for(cfg, sets):
    from cifarq import train

    h = train._check({**CONFIGS[cfg], **sets, 'seed': 0})
    return h, train.Run(h, [0], DATA)


def timed(fn):
    import jax

    for _ in range(N_WARM):
        out = fn()
    jax.block_until_ready(out)
    t = time.perf_counter()
    for _ in range(N_TIME):
        out = fn()
    jax.block_until_ready(out)
    return round((time.perf_counter() - t) / N_TIME * 1e3, 3)


def components(cfg, sets, tag):
    import jax
    import jax.numpy as jnp

    from cifarq import jx, train
    from cifarq.model import MEAN, STD

    h, run = run_for(cfg, sets)
    S, hp, bs = run.split, run.hp, h['batch_size']
    p = jax.tree.map(lambda a: a[0], run.st['params'])
    r = jax.tree.map(lambda a: a[0], run.st['rest'])
    x_all = jax.device_put(__import__('cifarq.data.dataset', fromlist=['load']).load(DATA)['x_train'][:bs])
    y = jnp.zeros((bs,), jnp.int32)
    key = jax.random.key(0)
    norm = lambda xb: (xb - jnp.asarray(MEAN)) / jnp.asarray(STD)
    aug = jax.jit(lambda k, xb: train.augment(k, xb.astype(jnp.float32) / 255, h, hp))
    fwd = jax.jit(lambda p, r, xb: S.apply(p, r, norm(xb.astype(jnp.float32) / 255), train=True)[0])

    def loss(p, r, xb):
        out, r2 = S.apply(p, r, norm(xb.astype(jnp.float32) / 255), train=True)
        return jx.cross_entropy(out * hp['logit_scale'], y, 10, hp['label_smoothing']), r2

    fb = jax.jit(jax.value_and_grad(loss, has_aux=True))
    ms = {'augment': timed(lambda: aug(key, x_all)), 'forward': timed(lambda: fwd(p, r, x_all)), 'forward_backward': timed(lambda: fb(p, r, x_all))}
    run.epoch()  # (compile)
    t = time.perf_counter()
    for _ in range(3):
        run.epoch()
    jax.block_until_ready(run.st['params'])
    epoch_s = (time.perf_counter() - t) / 3
    ms['train_step'] = round(epoch_s / run.steps_per_epoch * 1e3, 3)
    run.accuracies('val')
    t = time.perf_counter()
    for _ in range(3):
        run._w = None
        run.accuracies('val')
    ms['eval_val_5000'] = round((time.perf_counter() - t) / 3 * 1e3, 1)
    stats = jax.devices()[0].memory_stats() or {}
    write(
        f'jax-{cfg}-components{tag}',
        {
            'framework': 'jax',
            'config': cfg,
            'batch': bs,
            'device': jx.device_name(),
            'xla_flags': os.environ.get('XLA_FLAGS'),
            'sets': sets,
            'ms': ms,
            'steps_per_epoch': run.steps_per_epoch,
            'epoch_s': round(epoch_s, 3),
            'peak_gpu_mb': round(stats.get('peak_bytes_in_use', 0) / 2**20),
        },
    )


def e2e(cfg, sets, tag):
    import jax

    from cifarq import train

    t = time.time()
    r = train.train(data=DATA, **{**CONFIGS[cfg], **sets, 'epochs': 4})
    hist = [e['seconds'] for e in r['history']]
    stats = jax.devices()[0].memory_stats() or {}
    write(
        f'jax-{cfg}-e2e{tag}',
        {
            'framework': 'jax',
            'config': cfg,
            'xla_flags': os.environ.get('XLA_FLAGS'),
            'sets': sets,
            'first_epoch_s': hist[0],
            'steady_epoch_s': round((hist[-1] - hist[0]) / 3, 3),
            'process_wall_s': round(time.time() - t, 1),
            'val_acc_4ep': r['val_acc'],
            'images_per_s': r['images_per_s'],
            'peak_gpu_mb': round(stats.get('peak_bytes_in_use', 0) / 2**20),
        },
    )


def trace(cfg, sets, tag):
    import jax

    h, run = run_for(cfg, sets)
    run.epoch()  # (compile)
    jax.block_until_ready(run.st['params'])
    d = tempfile.mkdtemp()
    with jax.profiler.trace(d, create_perfetto_trace=True):
        run.epoch()
        jax.block_until_ready(run.st['params'])
    f = next(Path(d).rglob('perfetto_trace.json.gz'))
    events = json.loads(gzip.decompress(f.read_bytes()))
    events = events['traceEvents'] if isinstance(events, dict) else events
    write(
        f'jax-{cfg}-trace{tag}',
        {
            'framework': 'jax',
            'config': cfg,
            'steps': run.steps_per_epoch,
            'xla_flags': os.environ.get('XLA_FLAGS'),
            'kernels': gpu_kernels(events),
            'trace_dir': d,
        },
    )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('mode')
    ap.add_argument('--tag', default='')
    ap.add_argument('--xla', default=None)
    ap.add_argument('--set', nargs='*', default=[])
    a = ap.parse_args()
    if a.xla:
        os.environ['XLA_FLAGS'] = a.xla
    sets = {k: json.loads(v) for k, v in (s.split('=', 1) for s in a.set)}
    {'e2e': e2e, 'components': components, 'trace': trace}[a.mode](a.config, sets, f'-{a.tag}' if a.tag else '')
