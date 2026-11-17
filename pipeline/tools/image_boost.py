#!/usr/bin/env python
"""Prepare the image-feature boosting follow-up; submit locally after the ResNet18 job."""

import argparse
import json
import subprocess
import tarfile
from pathlib import Path

from runq.search.function_job import make_job

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--boost-python', type=Path, default=ROOT / '.venv-boost/bin/python')
    parser.add_argument('--trials', type=int, default=36)
    args = parser.parse_args()
    if not args.boost_python.is_file() or args.trials < 1:
        parser.error('need an existing boosting interpreter and positive trial count')
    changed = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--', 'src', 'pipeline', 'pyproject.toml'], cwd=ROOT, text=True)
    if changed.strip():
        parser.error('commit implementation changes before preparing frozen source')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    destination = ROOT / 'results/autoresearch/boost-source' / revision
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / 'source.tar'
    subprocess.run(['git', 'archive', '--format=tar', f'--output={archive}', revision], cwd=ROOT, check=True)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter='data')
    archive.unlink()
    job = make_job(
        'image-feature-boost',
        'cifarq.boost_campaign',
        'run',
        {'root': str(ROOT), 'boost_python': str(args.boost_python.absolute()), 'trials': args.trials},
        path=[destination / 'src'],
        session='cifar-resnet18',
        note='After ResNet18: image features + XGBoost/LightGBM, primary-score Optuna, within unused original 24 hours',
        gpu_mb=14500,
        host_mb=12000,
        threads=4,
        minutes=240,
        stall_min=40,
        report=['status', 'completed_trials'],
    )
    provenance = {
        'revision': revision,
        'source': str(destination),
        'job': str(job),
        'boost_python': str(args.boost_python.absolute()),
        'must_follow': 47,
        'budget': 'At most min(4 hours, unused original 24-hour total); no additional compute approval.',
    }
    (job / 'source.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
