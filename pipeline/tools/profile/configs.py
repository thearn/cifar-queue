"""Shared profiling configurations for the Torch and JAX harnesses (train's argument names)."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = str(sorted((REPO / 'data' / 'dataset').glob('dataset-*.npz'))[0])
OUT = REPO / 'results' / 'profile'

CHAMPION = json.loads((REPO / 'results' / 'g4-best.json').read_text())
PAGE = {
    'epochs': 24,
    'width': 64,
    'activation': 'relu',
    'batch_size': 512,
    'head': 'linear',
    'logit_scale': 0.125,
    'tta_flip': False,
    'dropout': 0.0,
    'label_smoothing': 0.2,
    'optimizer': 'sgd',
    'lr': 0.4,
    'momentum': 0.9,
    'nesterov': True,
    'weight_decay': 0.0005,
    'grad_clip': 0.0,
    'ema_decay': 0.0,
    'warmup_frac': 0.20833333333333334,
    'warmup_start_frac': 0.005,
    'decay': 'linear',
    'final_lr_frac': 0.001,
    'aug_translate': 4.0,
    'aug_rotate': 0.0,
    'aug_scale': 0.0,
    'aug_brightness': 0.0,
    'aug_contrast': 0.0,
    'aug_saturation': 0.0,
    'aug_flip': 0.5,
    'cutout_size': 8,
    'cutout_prob': 1.0,
    'mix': 'none',
}  # tune.train_params(tune.PAGE)
DEFAULTS = {}
CONFIGS = {'champion': CHAMPION, 'page': PAGE, 'defaults': DEFAULTS}


def write(name: str, rec: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f'{name}.json'
    f.write_text(json.dumps(rec, indent=1, default=str) + '\n')
    print(json.dumps(rec, default=str)[:2000], flush=True)
    return f


def kernel_table(events, top: int = 25) -> list[dict]:
    """GPU kernel time by name from Chrome/Perfetto trace events (``ph`` X with a duration on a GPU stream)."""
    agg = {}
    for e in events:
        if e.get('ph') != 'X' or 'dur' not in e:
            continue
        cat = str(e.get('cat', '')).lower()
        name = str(e.get('name', ''))
        if cat not in ('kernel', 'gpu_memcpy', 'gpu_memset') and not cat.startswith('kernel'):
            continue
        a = agg.setdefault(name[:120], [0.0, 0])
        a[0] += float(e['dur'])
        a[1] += 1
    total = sum(v[0] for v in agg.values()) or 1.0
    rows = sorted(
        ({'kernel': k, 'ms': round(v[0] / 1e3, 2), 'calls': v[1], 'share': round(v[0] / total, 4)} for k, v in agg.items()), key=lambda r: -r['ms']
    )
    return [{'total_kernel_ms': round(total / 1e3, 1)}] + rows[:top]
