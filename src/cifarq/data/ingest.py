"""Ingestion: the source in ``configs/data.toml`` copied to ``data/raw/``, and ``data/manifests/raw.json`` naming
the source (its resolved revision) and each raw file's sha256. Running it again with the same source and settings is a
no-op. ``synthetic`` (the tests) writes tiny parquet files in the hub's format (an ``img`` struct of PNG bytes and a
``label``), so the later stages run as they do on the real files."""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np

from .. import paths
from . import storage

RAW = paths.DATA / 'raw'


def synthetic(dest: Path, train: int, test: int, seed: int) -> list[Path]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    rng = np.random.default_rng(seed)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for split, n in (('train', train), ('test', test)):
        labels = np.arange(n) % 10
        rng.shuffle(labels)
        images = []
        for y in labels:                                  # a class's own mean colour plus noise: learnable, not trivial
            x = rng.normal(40 + 20 * y, 60, (32, 32, 3)).clip(0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(x).save(buf, format='PNG')
            images.append({'bytes': buf.getvalue(), 'path': None})
        f = dest / f'{split}-00000-of-00001.parquet'
        pq.write_table(pa.table({'img': images, 'label': labels.tolist()}), f)
        out.append(f)
    return out


def fetch(force: bool = False) -> dict:
    """Ingest the configured source into ``data/raw/``; the manifest (unchanged when the source and settings are)."""
    cfg = paths.config('data')['source']
    inputs = paths.digest(cfg)
    pinned = not cfg['uri'].startswith('hf://') or '@' in cfg['uri']
    if not force and pinned and paths.up_to_date('raw', inputs, RAW):
        return paths.read_manifest('raw')
    if RAW.exists():
        for f in RAW.rglob('*'):
            if f.is_file():
                f.unlink()
    if cfg['uri'] == 'synthetic':
        files, resolved = synthetic(RAW, **cfg['synthetic']), {'uri': 'synthetic'}
    else:
        files, resolved = storage.fetch(cfg['uri'], RAW, cfg.get('pattern', '*'))
    if not files:
        raise FileNotFoundError(f'no files matching {cfg.get("pattern")!r} at {cfg["uri"]}')
    manifest = {'stage': 'raw', 'inputs': inputs, 'source': resolved, 'settings': cfg, 'fetched': time.strftime('%Y-%m-%d %H:%M:%S'),
                'files': {f.relative_to(RAW).as_posix(): paths.sha256(f) for f in sorted(files)}}
    paths.write_manifest('raw', manifest)
    return manifest


def shards() -> list[Path]:
    """The raw files the manifest names."""
    m = paths.read_manifest('raw')
    if m is None:
        raise FileNotFoundError('no data/manifests/raw.json: run pipeline/10_ingest.py first')
    return [RAW / f for f in m['files']]
