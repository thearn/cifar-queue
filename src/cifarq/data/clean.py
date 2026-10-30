"""Decoding, one parquet file at a time: ``clean_shard`` reads a hub parquet file (``img`` PNG bytes, ``label``) and
writes its images as uint8 RGB arrays in the current directory (a queue job's directory, which comes back with the
job), refusing images of the wrong size and unknown labels and saying how many each rule dropped. Pure: the same file
and rules give the same arrays. The split a file holds is its name's prefix (``train-``, ``test-``)."""
from __future__ import annotations

import io
import shutil
from pathlib import Path

import numpy as np

from .. import paths
from . import ingest, validate

CLEAN = paths.DATA / 'clean'


def split_of(name: str) -> str:
    return Path(name).name.split('-', 1)[0]


def clean_shard(raw: str, rules: dict, out: str | None = None) -> dict:
    """Decode ``raw`` by ``rules`` ({classes, size}) into ``out`` (default ``<split>.npz`` here): arrays ``ids`` (the
    row's index in the file), ``x`` (uint8, n x size x size x 3) and ``y`` (uint8 class index).
    {"file", "split", "rows", "kept", "dropped": {rule: rows}}."""
    import pyarrow.parquet as pq
    from PIL import Image
    raw = Path(raw)
    t = pq.read_table(raw).to_pydict()
    images = t[next(k for k in t if k in ('img', 'image'))]
    size, n_classes = rules['size'], len(rules['classes'])
    dropped = dict.fromkeys(('wrong_size', 'unknown_label'), 0)
    ids, xs, ys = [], [], []
    for i, (img, label) in enumerate(zip(images, t['label'])):
        x = np.asarray(Image.open(io.BytesIO(img['bytes'])).convert('RGB'))
        if x.shape != (size, size, 3):
            dropped['wrong_size'] += 1
            continue
        if not 0 <= int(label) < n_classes:
            dropped['unknown_label'] += 1
            continue
        ids.append(i), xs.append(x), ys.append(label)
    out = Path(out or f'{split_of(raw.name)}.npz')
    np.savez(out, ids=np.asarray(ids, dtype=np.int64), x=np.stack(xs) if xs else np.zeros((0, size, size, 3), np.uint8),
             y=np.asarray(ys, dtype=np.uint8))
    return {'file': str(out), 'split': split_of(raw.name), 'rows': len(images), 'kept': len(ids), 'dropped': dropped}


def run(executor=None, force: bool = False) -> dict:
    """Decode every raw file into ``data/clean/<split>.npz``, check the result (``validate.cleaned``) and write
    ``data/manifests/clean.json``; the manifest. ``executor``: a ``concurrent.futures`` executor taking
    ``'module:function'`` (a solve-queue ``QueueExecutor``: one job per file; the pipeline's in-process one), each
    future's ``job`` the directory the file was written in; none: here, one file after another."""
    cfg = paths.config('data')
    raw = paths.read_manifest('raw') or {}
    inputs = paths.digest([raw.get('files'), cfg['clean']])
    if not force and paths.up_to_date('clean', inputs, CLEAN):
        return paths.read_manifest('clean')
    shutil.rmtree(CLEAN, ignore_errors=True)
    CLEAN.mkdir(parents=True)
    reports = []
    if executor is None:
        for shard in ingest.shards():
            reports.append(clean_shard(str(shard), cfg['clean'], out=str(CLEAN / f'{split_of(shard.name)}.npz')))
    else:
        futures = [executor.submit(f'{__name__}:clean_shard', raw=str(shard), rules=cfg['clean']) for shard in ingest.shards()]
        for f in futures:
            r = f.result()
            target = CLEAN / Path(r['file']).name
            shutil.copy2(Path(f.job) / r['file'], target)
            reports.append({**r, 'file': str(target)})
    validate.cleaned(reports, cfg['validate'])
    manifest = {'stage': 'clean', 'inputs': inputs, 'settings': cfg['clean'],
                'rows': {r['split']: r['rows'] for r in reports}, 'kept': {r['split']: r['kept'] for r in reports},
                'dropped': {k: sum(r['dropped'][k] for r in reports) for k in reports[0]['dropped']},
                'files': {Path(r['file']).name: paths.sha256(Path(r['file'])) for r in reports}}
    paths.write_manifest('clean', manifest)
    return manifest
