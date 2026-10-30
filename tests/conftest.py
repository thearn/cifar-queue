"""The tests run on a copy of configs/ in a temporary directory (``ML_PROJECT_ROOT``, set before the package is
imported), its source switched to ``synthetic`` (tiny CIFAR-shaped parquet files made there: no network, no GPU), so
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
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')        # the tests are CPU tests, also on a machine with a GPU


@pytest.fixture(scope='session')
def built():
    """The data stages run once on the synthetic source: the dataset's arrays."""
    from cifarq.data import clean, dataset, ingest
    ingest.fetch()
    clean.run()
    dataset.build()
    return dataset.load()
