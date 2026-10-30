"""CIFAR-10.1 v6 (Recht, Roelofs, Schmidt & Shankar, "Do CIFAR-10 classifiers generalize to CIFAR-10?", 2019;
github.com/modestyachts/CIFAR-10.1 at d9982ab): 2000 new test images drawn like CIFAR-10's, 200 a class, the same
class order. The autoresearch session's untouched hold-out: never trained, searched or selected on. ``fetch`` stores
the two arrays under ``data/holdout/`` and their hashes in ``data/manifests/holdout.json``; ``look`` evaluates a
saved model on them once and appends the look to ``data/holdout/looks.jsonl``, so every exposure is on record.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from .. import paths

HOLDOUT = paths.DATA / 'holdout'
REVISION = 'd9982abb0bfc4846b8d13a11e66b887d946205d0'
URL = f'https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/{REVISION}/datasets/'
FILES = {'cifar10.1_v6_data.npy': '2997188e5816f5bd545dc77771b6227828c28146049fcecf3fa10775474cacc6',
         'cifar10.1_v6_labels.npy': 'ae40beda001693674edc94d925ee8268cfe68905f8f9aff800c8dcdfcd6c9448'}


def fetch() -> dict:
    """Download the arrays at the pinned revision (unless present with the right hashes); the manifest."""
    HOLDOUT.mkdir(parents=True, exist_ok=True)
    for name, sha in FILES.items():
        f = HOLDOUT / name
        if not (f.exists() and paths.sha256(f) == sha):
            urllib.request.urlretrieve(URL + name, f)
        if paths.sha256(f) != sha:
            raise ValueError(f'{name}: sha256 {paths.sha256(f)}, expected {sha}')
    x, y = load()
    if x.shape != (2000, 32, 32, 3) or x.dtype != np.uint8 or np.bincount(y).tolist() != [200] * 10:
        raise ValueError('CIFAR-10.1 v6: unexpected shape, type or class counts')
    manifest = {'stage': 'holdout', 'source': URL, 'revision': REVISION, 'files': FILES, 'rows': int(len(y))}
    paths.write_manifest('holdout', manifest)
    return manifest


def load() -> tuple[np.ndarray, np.ndarray]:
    """(images uint8 (2000, 32, 32, 3), labels int64 (2000,)); never pickled data."""
    x = np.load(HOLDOUT / 'cifar10.1_v6_data.npy', allow_pickle=False)
    y = np.load(HOLDOUT / 'cifar10.1_v6_labels.npy', allow_pickle=False).astype(np.int64)
    return x, y


def look(directory, why: str, device: str = 'cpu') -> dict:
    """One evaluation of a saved model on the hold-out, logged with its reason: accuracy and the per-image correctness
    (for paired comparisons between models looked at)."""
    from ..footprint import predictor
    x, y = load()
    correct = predictor(directory, device).predict(x).argmax(1) == y
    entry = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'model': str(directory), 'why': why,
             'acc': round(float(correct.mean()), 4), 'n': int(len(y))}
    with open(HOLDOUT / 'looks.jsonl', 'a') as f:
        f.write(json.dumps(entry) + '\n')
    np.save(Path(directory) / 'holdout_correct.npy', correct)
    return entry
