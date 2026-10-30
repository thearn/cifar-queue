"""The inference footprint and runtime of a saved model, measured the same way for every family (the autoresearch
session's evaluator, docs/autoresearch/2026-10-29-footprint-arch/PLAN.md section 2.2).

A model is a saved directory: ``config.json`` names its family (``airbench``; no ``family`` is a ``hub.ResNet9``), and
``predictor(directory, device)`` gives an object whose ``predict(uint8 (n, 32, 32, 3))`` returns class probabilities
(n, 10), normalisation and test-time augmentation included. ``static`` counts its parameters, stored bytes and
multiply-accumulates per image; ``measure`` runs one setting (device, threads, batch) in a fresh process so memory
peaks start from the interpreter's floor; ``bench`` does the whole table. Run on the local machine only, so timings
compare.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def family(directory) -> str:
    cfg = json.loads((Path(directory) / 'config.json').read_text())
    return cfg.get('family', 'resnet9')


class _ResNet9Predictor:
    def __init__(self, directory, device='cpu', dtype=None):
        import torch
        from .hub import ResNet9
        self.net = ResNet9.from_pretrained(str(directory)).to(device).eval()
        self.dtype = dtype or torch.float32
        if self.dtype != torch.float32:
            self.net = self.net.to(self.dtype)
            for m in self.net.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    m.float()
        self.views = 2 if self.net.tta_flip else 1
        self.device = torch.device(device)

    def predict(self, images):
        import numpy as np
        import torch
        with torch.no_grad(), torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            return self.net.predict(np.asarray(images, dtype=np.uint8))


def predictor(directory, device='cpu', dtype=None):
    fam = family(directory)
    if fam == 'airbench':
        from .airbench import Predictor
        return Predictor(directory, device, dtype)
    if fam == 'resnet9':
        return _ResNet9Predictor(directory, device, dtype)
    raise ValueError(f'unknown family {fam!r}')


def views(directory) -> int:
    """Forward passes per image at inference (test-time augmentation's views)."""
    cfg = json.loads((Path(directory) / 'config.json').read_text())
    if family(directory) == 'resnet9':
        return 2 if cfg.get('tta_flip') else 1
    return {0: 1, 1: 2, 2: 6, 3: 18}[int(cfg.get('tta_level', 0))]       # (level 3: added 2026-10-30, P2)


def count_macs(net, shape=(1, 3, 32, 32)) -> int:
    """Multiply-accumulates of one forward pass of ``net`` (fp32, CPU) on an input of ``shape``: convolutions (grouped
    ones included) and linear layers, which carry all but a negligible share of the arithmetic here."""
    import torch
    total = [0]

    def hook(m, inputs, out):
        if isinstance(m, torch.nn.Conv2d):
            total[0] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, torch.nn.Linear):
            total[0] += out.numel() * m.in_features
    handles = [m.register_forward_hook(hook) for m in net.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    try:
        net = net.float().eval()
        with torch.no_grad():
            net(torch.zeros(shape))
    finally:
        for h in handles:
            h.remove()
    return total[0] // shape[0]


def static(directory) -> dict:
    """Parameters (stored: every parameter tensor, frozen ones included; trainable), stored bytes (every file of the
    directory the predictor needs), and MACs per image (one view, and times the views)."""
    d = Path(directory)
    p = predictor(d)
    net = p.net.net if family(d) == 'resnet9' else p.net
    stored = sum(t.numel() for t in net.parameters())
    trainable = sum(t.numel() for t in net.parameters() if t.requires_grad)
    if family(d) == 'resnet9':
        frozen = sum(m.weight.numel() for m in net.modules()
                     if hasattr(m, 'weight') and m.weight is not None and not m.weight.requires_grad)
        trainable = stored - frozen
    elif family(d) == 'airbench':
        from .airbench import _groups, build, param_counts
        fresh, whiten = build(p.h)
        _groups(fresh, whiten, p.h.get('bn_scale', False))
        trainable = param_counts(fresh)['trainable']
    macs = count_macs(net)
    v = views(d)
    return {'family': family(d), 'params_stored': stored, 'params_trainable': trainable,
            'bytes': sum(f.stat().st_size for f in (d / 'model.safetensors', d / 'config.json')),
            'macs_per_view': macs, 'views': v, 'macs_per_image': macs * v}


class _Null:
    """The null model of the memory baseline: the same inputs, a constant output, no network."""

    def __init__(self, device):
        import torch
        self.device = torch.device(device)

    def predict(self, images):
        import numpy as np
        import torch
        x = torch.from_numpy(np.ascontiguousarray(images)).to(self.device).float()
        return torch.full((len(images), 10), 0.1, device=self.device).add_(0 * x.mean()).cpu().numpy()


def _vmrss_mb() -> float:
    """This process's resident memory now (Linux ``/proc``), MB."""
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return round(int(line.split()[1]) / 1024, 1)
    return 0.0


def _measure(directory: str, device: str, threads: int, batch: int, warmup: int, timed: int) -> dict:
    """One setting, in this (fresh) process: the process's peak memory (RSS, and GPU allocations), and latency or
    throughput. ``directory`` ``null`` measures the baseline (``_Null``), which ``measure`` subtracts."""
    import resource
    import time
    import numpy as np
    import torch
    if threads:
        torch.set_num_threads(threads)
    dev = torch.device(device)
    if dev.type == 'cuda':
        torch.zeros(1, device=dev)
        torch.cuda.synchronize()
    dtype = torch.bfloat16 if dev.type == 'cuda' else torch.float32
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(dev)
    p = _Null(device) if directory == 'null' else predictor(directory, device, dtype)
    images = np.random.default_rng(0).integers(0, 256, size=(batch, 32, 32, 3), dtype=np.uint8)
    sync = torch.cuda.synchronize if dev.type == 'cuda' else (lambda: None)
    for _ in range(warmup):
        p.predict(images)
    sync()
    times = []
    for _ in range(timed):
        t = time.perf_counter()
        p.predict(images)
        sync()
        times.append(time.perf_counter() - t)
    times = np.asarray(times)
    out = {'device': device, 'threads': threads or torch.get_num_threads(), 'batch': batch,
           'p50_ms': round(1e3 * float(np.median(times)), 3), 'p95_ms': round(1e3 * float(np.percentile(times, 95)), 3),
           'images_per_s': round(batch / float(np.median(times)), 1),
           'peak_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
           'rss_mb': _vmrss_mb()}
    if dev.type == 'cuda':
        out['peak_gpu_mb'] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 20, 1)
        out['gpu'] = torch.cuda.get_device_name(dev)
    return out


def _fresh(directory, device, threads, batch, warmup, timed) -> dict:
    cmd = [sys.executable, '-m', 'cifarq.footprint', str(directory), device, str(threads), str(batch), str(warmup), str(timed)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def measure(directory, device='cpu', threads: int = 0, batch: int = 1, warmup: int = 20, timed: int = 200) -> dict:
    """``_measure`` in a fresh interpreter, and the null model's in another: peak memory is the model's process peak
    minus the null's (the same imports, inputs and CUDA context: what the model itself adds), latency the model's."""
    r = _fresh(directory, device, threads, batch, warmup, timed)
    null = _fresh('null', device, threads, batch, 2, 3)
    r['peak_rss_total_mb'], r['rss_total_mb'] = r['peak_rss_mb'], r['rss_mb']
    r['peak_rss_mb'] = round(max(0.0, r['peak_rss_mb'] - null['peak_rss_mb']), 1)
    r['rss_mb'] = round(max(0.0, r['rss_mb'] - null['rss_mb']), 1)       # (resident after the timed loop: what it holds)
    if 'peak_gpu_mb' in r:
        r['peak_gpu_total_mb'] = r['peak_gpu_mb']
        r['peak_gpu_mb'] = round(max(0.0, r['peak_gpu_mb'] - null.get('peak_gpu_mb', 0.0)), 1)
    return r


SETTINGS = [  # (name, device, threads, batch, warmup, timed)
    ('gpu_b1', 'cuda', 0, 1, 200, 500), ('gpu_b256', 'cuda', 0, 256, 5, 30), ('gpu_b1024', 'cuda', 0, 1024, 3, 20),
    ('cpu1_b1', 'cpu', 1, 1, 30, 300), ('cpuall_b1', 'cpu', 0, 1, 30, 300), ('cpuall_b256', 'cpu', 0, 256, 2, 10)]


def bench(directory, settings=None, gpu: bool = True) -> dict:
    """The whole table: ``static`` and every setting of ``SETTINGS`` (the GPU ones only when ``gpu`` and CUDA is there)."""
    import torch
    out = {'model': str(directory), **static(directory), 'runs': {}}
    for name, device, threads, batch, warmup, timed in settings or SETTINGS:
        if device == 'cuda' and not (gpu and torch.cuda.is_available()):
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
