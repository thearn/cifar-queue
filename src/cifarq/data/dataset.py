"""The dataset: CIFAR-10's train, validation and test images and labels (uint8, 32x32x3) in one array file
``data/dataset/dataset-<sha12>.npz``, with each image's id (``train:<index>`` or ``test:<index>`` in the hub's files)
and the class names. ``data/manifests/dataset.json`` names it, its hash and the images in each split; the manifest is
committed, the array file is not. Jobs name the file as their ``data`` (solve-queue sends it to each VM once, by its
content).

The split is not the template's hash split: the validation images are the first 5000 of
``numpy.random.default_rng(1234).permutation(50000)``, the rest training, both kept in that permutation's order,
which is exactly what ``train()`` drew before this stage existed (configs/data.toml, [split]). Every study so far was
run on it. The test split is only ever reported: searches optimise validation accuracy, and the champion is chosen on it.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .. import paths
from . import validate

DATASET = paths.DATA / 'dataset'


def build(force: bool = False) -> dict:
    """Build the dataset from the decoded files; its manifest (unchanged when they and the split are)."""
    from .clean import CLEAN
    cfg = paths.config('data')
    clean = paths.read_manifest('clean')
    if clean is None:
        raise FileNotFoundError('no data/manifests/clean.json: run pipeline/20_clean.py first')
    inputs = paths.digest([clean['files'], cfg['split'], cfg['clean']['classes']])
    if not force and paths.up_to_date('dataset', inputs, DATASET):
        return paths.read_manifest('dataset')
    train, test = np.load(CLEAN / 'train.npz'), np.load(CLEAN / 'test.npz')
    order = np.random.default_rng(cfg['split']['seed']).permutation(len(train['y']))
    val_idx, tr_idx = order[:cfg['split']['validation']], order[cfg['split']['validation']:]
    ids = lambda split, a: np.asarray([f'{split}:{i}' for i in a])
    arrays = {'classes': np.asarray(cfg['clean']['classes']),
              'x_train': train['x'][tr_idx], 'y_train': train['y'][tr_idx], 'ids_train': ids('train', train['ids'][tr_idx]),
              'x_val': train['x'][val_idx], 'y_val': train['y'][val_idx], 'ids_val': ids('train', train['ids'][val_idx]),
              'x_test': test['x'], 'y_test': test['y'], 'ids_test': ids('test', test['ids'])}
    validate.dataset(arrays)
    shutil.rmtree(DATASET, ignore_errors=True)
    DATASET.mkdir(parents=True)
    tmp = DATASET / 'dataset.tmp.npz'
    np.savez(tmp, **arrays)
    sha = paths.sha256(tmp)
    out = DATASET / f'dataset-{sha[:12]}.npz'
    tmp.replace(out)
    manifest = {'stage': 'dataset', 'inputs': inputs, 'file': out.name, 'sha256': sha, 'settings': cfg['split'],
                'rows': {s: int(arrays[f'y_{s}'].size) for s in ('train', 'val', 'test')},
                'classes': arrays['classes'].tolist(), 'files': {out.name: sha}}
    paths.write_manifest('dataset', manifest)
    return manifest


def path() -> Path:
    """The array file the manifest names (``build()`` first); without the manifest (a VM is sent the file alone, as
    the job's data) the one array file there."""
    m = paths.read_manifest('dataset')
    if m is not None:
        return DATASET / m['file']
    found = sorted(DATASET.glob('dataset-*.npz'))
    if len(found) != 1:
        raise FileNotFoundError(f'no data/manifests/dataset.json and {len(found)} array files in {DATASET}')
    return found[0]


_LOADED: dict = {}


def load(file: Path | str | None = None) -> dict:
    """The arrays, kept in this process after the first load: on a VM, solve-queue's persistent worker loads them once
    (``warm``) and every trial it forks reads them without a copy."""
    key = str(Path(file or path()).resolve())
    if key not in _LOADED:
        _LOADED[key] = dict(np.load(key))
    return _LOADED[key]


def warm():
    """Load the arrays now (the worker's preload: ``cifarq.data.dataset:warm``); nothing if there is none here."""
    try:
        load()
    except FileNotFoundError:
        pass
