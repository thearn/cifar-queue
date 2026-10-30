"""The pipeline's stages as scripts, in this process (``--targets direct``), on a fresh copy of configs/ (the source
synthetic) that is its own git repository: ingest to a staged model repo. The search needs solve-queue (skipped without
it); the queue test runs a training through the daemon when ``RUN_QUEUE_TESTS=1`` and Pueue is installed."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import SMALL, copy_configs

REPO = Path(__file__).resolve().parents[1]
SETS = [f'{k}={v}' for k, v in SMALL.items()]


def run(root, script, *args):
    env = {**os.environ, 'ML_PROJECT_ROOT': str(root)}
    r = subprocess.run([sys.executable, str(REPO / 'pipeline' / script), *args], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.fixture(scope='module')
def root(tmp_path_factory):
    """A project root with the data stages run, and a README for the model card to wrap."""
    root = tmp_path_factory.mktemp('project')
    copy_configs(root)
    (root / 'README.md').write_text('# cifar-queue\n\nThe introduction.\n\n## Using the model\n\nThe rest.\n')
    for args in (['init', '-q'], ['add', '-A'], ['-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'configs']):
        subprocess.run(['git', '-C', str(root), *args], check=True)
    run(root, '10_ingest.py')
    run(root, '20_clean.py', '--targets', 'direct')
    run(root, '30_dataset.py')
    return root


def test_stages_in_process(root):
    assert 'val_acc' in run(root, '40_train.py', '--targets', 'direct', '--set', 'epochs=1', *SETS)
    run(root, '60_champion.py', '--reference', '--seeds', '3', '--targets', 'direct', '--set', 'epochs=1', *SETS)
    champion = json.loads(sorted((root / 'results' / 'champion').glob('*.json'))[-1].read_text())
    assert champion['chosen']['seed'] in (0, 1, 2) and champion['provenance']['dataset']
    assert 'staged' in run(root, '70_publish.py', '--dry-run', '--no-space')
    card = (root / 'results' / 'publish' / 'model' / 'README.md').read_text()
    assert champion['provenance']['dataset'] in card and (root / 'results' / 'publish' / 'model' / 'model.safetensors').exists()
    assert card.index('The introduction') < card.index('## The published model') < card.index('## Using the model')


def test_publish_refuses_a_champion_of_another_dataset(root):
    f = sorted((root / 'results' / 'champion').glob('*.json'))[-1]
    doc = json.loads(f.read_text())
    doc['provenance']['dataset'] = 'something else'
    other = f.with_name('other.json')
    other.write_text(json.dumps(doc))
    env = {**os.environ, 'ML_PROJECT_ROOT': str(root)}
    r = subprocess.run([sys.executable, str(REPO / 'pipeline' / '70_publish.py'), '--dry-run', '--champion', str(other)],
                       env=env, capture_output=True, text=True)
    other.unlink()
    assert r.returncode != 0 and 'train it again' in r.stderr


@pytest.mark.queue
@pytest.mark.skipif(os.environ.get('RUN_QUEUE_TESTS') != '1' or not shutil.which('pueue'),
                    reason='RUN_QUEUE_TESTS=1 and a running solve-queue daemon')
def test_training_through_the_queue(root):
    pytest.importorskip('solve_queue')
    out = run(root, '40_train.py', '--targets', 'local', '--set', 'epochs=1', *SETS)
    assert 'val_acc' in out
