#!/usr/bin/env python
"""Prepare a frozen-source, local-only ResNet18 campaign job; use runq submit --check before submission."""

import argparse
import json
import subprocess
import tarfile
from pathlib import Path

from runq.search.function_job import make_job

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--torch-python', type=Path, required=True, help='existing torch/torchvision interpreter; never modified')
    parser.add_argument('--hours', type=float, default=23.9)
    parser.add_argument('--trials', type=int, default=50)
    args = parser.parse_args()
    if not 0 < args.hours <= 24 or args.trials < 1:
        parser.error('approval is at most 24 local T4 hours')
    if not args.torch_python.is_file():
        parser.error('PyTorch interpreter does not exist')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
        parser.error('commit tracked implementation and protocol changes before freezing the job')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    destination = ROOT / 'results/autoresearch/resnet18-source' / revision
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / 'source.tar'
    subprocess.run(['git', 'archive', '--format=tar', f'--output={archive}', revision], cwd=ROOT, check=True)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter='data')
    archive.unlink()
    job = make_job(
        'resnet18-score',
        'cifarq.resnet18_campaign',
        'run',
        {
            'root': str(ROOT),
            'data': str(ROOT / 'data/dataset/dataset-56cff0f67835.npz'),
            'torch_python': str(args.torch_python.absolute()),
            'hours': args.hours,
            'trials': args.trials,
        },
        path=[destination / 'src'],
        note='d3 reference -> ImageNet ResNet18 -> scratch -> primary-score Optuna -> fresh seeds; local T4 only',
        session='cifar-resnet18',
        gpu_mb=14500,
        host_mb=12000,
        minutes=args.hours * 60,
        stall_min=90,
        report=['fine_tuned_score', 'scratch_score', 'confirmed_recipe_score', 'completed_trials'],
    )
    provenance = {
        'revision': revision,
        'source': str(destination),
        'job': str(job),
        'torch_python': str(args.torch_python.absolute()),
    }
    (job / 'source.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
