"""The dataset: CIFAR-10 from the Hugging Face hub (uoft-cs/cifar10, its parquet files at a pinned revision), decoded
once into one uint8 array file, ``data/cifar10-<sha12>.npz`` (train 50000 and test 10000 images, 32x32x3, and labels).

``prepare()`` writes it and ``data/MANIFEST.json`` (the hub revision, the parquet files' and the array file's sha256);
the manifest is committed, the array file is not: jobs name it as their ``data`` (solve-queue's job.json), so it goes
to the bucket once and to each VM once, by its content. ``path()`` is the prepared file named by the manifest.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

REPO = 'uoft-cs/cifar10'
DATA = Path(__file__).resolve().parents[2] / 'data'
MANIFEST = DATA / 'MANIFEST.json'
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def prepare(revision: str | None = None) -> Path:
    """Download the parquet files (at ``revision``, default the hub's current one, recorded), decode them and write the
    array file and the manifest. The array file's path."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image
    revision = revision or HfApi().dataset_info(REPO).sha
    arrays, sources = {}, {}
    for split in ('train', 'test'):
        name = f'plain_text/{split}-00000-of-00001.parquet'
        f = Path(hf_hub_download(REPO, name, repo_type='dataset', revision=revision))
        sources[name] = _sha256(f)
        t = pq.read_table(f).to_pydict()
        img_col = next(k for k in t if k in ('img', 'image'))
        arrays[f'x_{split}'] = np.stack([np.asarray(Image.open(io.BytesIO(r['bytes'])).convert('RGB')) for r in t[img_col]])
        arrays[f'y_{split}'] = np.asarray(t['label'], dtype=np.uint8)
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = DATA / 'cifar10.tmp.npz'
    np.savez(tmp, **arrays)
    sha = _sha256(tmp)
    out = DATA / f'cifar10-{sha[:12]}.npz'
    tmp.replace(out)
    MANIFEST.write_text(json.dumps({'source': f'https://huggingface.co/datasets/{REPO}', 'revision': revision,
                                    'parquet_sha256': sources, 'file': out.name, 'sha256': sha,
                                    'shapes': {k: list(v.shape) for k, v in arrays.items()}}, indent=1) + '\n')
    return out


def path() -> Path:
    """The prepared array file the manifest names (``prepare()`` first); without the manifest (a VM is sent the array
    file alone, as the job's data) the one array file there."""
    if MANIFEST.exists():
        return DATA / json.loads(MANIFEST.read_text())['file']
    found = sorted(DATA.glob('cifar10-*.npz'))
    if len(found) != 1:
        raise FileNotFoundError(f'{MANIFEST} is missing and {DATA} holds {len(found)} array files')
    return found[0]


def load(file: Path | str | None = None) -> dict:
    return dict(np.load(file or path()))
