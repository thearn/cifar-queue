"""The tests run on a copy of configs/ in a temporary directory (``ML_PROJECT_ROOT``, set before the package is
imported), its source switched to ``synthetic`` (tiny CIFAR-shaped parquet files made there: no network needed), so
they never touch the repository's data/ or results/. ``built`` runs the data stages there once."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# the training settings small enough for a CPU (500 training images)
SMALL = {'width': 16, 'batch_size': 64}


def copy_configs(root: Path):
    """configs/ copied into ``root``, the source synthetic and the validation split sized for it."""
    shutil.copytree(REPO / 'configs', root / 'configs')
    f = root / 'configs' / 'data.toml'
    text = f.read_text()
    uri = next(line for line in text.splitlines() if line.startswith('uri = '))
    f.write_text(text.replace(uri, 'uri = "synthetic"').replace('validation = 5000', 'validation = 100'))


TMP = Path(tempfile.mkdtemp(prefix='cifarq-test-'))
copy_configs(TMP)
os.environ['ML_PROJECT_ROOT'] = str(TMP)
os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # a CPU by default, so the suite runs anywhere and needs no card
# To exercise the GPU paths instead -- the arithmetic cifarq.precision picks, and the kernels XLA chooses -- run
# `JAX_PLATFORMS=cuda,cpu pytest`. Name the CPU too: a few tests measure or train on it deliberately, and dropping
# it from the list makes JAX refuse the backend outright rather than fall back.


@pytest.fixture(scope='session')
def built():
    """The data stages run once on the synthetic source: the dataset's arrays."""
    from cifarq.data import clean, dataset, ingest

    ingest.fetch()
    clean.run()
    dataset.build()
    return dataset.load()
