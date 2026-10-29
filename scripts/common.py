"""What the three scripts share: the repository's paths, the two L4 executors, and a queue executor that sends this
repository's code (src/) and names the dataset as the jobs' data (sent once per VM, by its content)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'src'
RESULTS = REPO / 'results'
sys.path.insert(0, str(SRC))
os.environ.setdefault('SOLVEQ_SESSION', 'cifar')

from cifarq import data  # noqa: E402

L4A, L4B = 'cifar-l4a', 'cifar-l4b'


def ensure_executor(name: str, *, gpu: str = 'L4', idle_minutes: float = 5., parallel: int | None = None,
                    shape: str | None = None):
    """Register ``colab:name`` (no VM until a job needs one), or update its parallel slots."""
    from solve_queue import executors
    rec = executors.load().get(name)
    if rec is None or (parallel and rec.get('parallel') != parallel):
        executors.add(name, gpu=gpu, idle_stop_minutes=idle_minutes, parallel=parallel, shape=shape, log=lambda s: None)
        executors.update(name, parallel=parallel)


def queue_executor(root: Path, target: str | None = None, targets: dict | None = None, **hints):
    """A QueueExecutor for this repository: ``target`` (one) or ``targets`` ({target: slots})."""
    from solve_queue.futures import QueueExecutor
    if not data.path().exists():
        raise SystemExit('no dataset: run scripts/prepare_data.py first')
    return QueueExecutor(root, target=target or 'local', targets=targets, path=[SRC], data=[data.path()],
                         **{'gpu_mb': 3000, 'host_mb': 6000, 'minutes': 5, **hints})


def save(kind: str, out: dict) -> Path:
    d = RESULTS / kind
    d.mkdir(parents=True, exist_ok=True)
    f = d / f'{time.strftime("%Y%m%d-%H%M%S")}.json'
    f.write_text(json.dumps(out, indent=1, default=str) + '\n')
    return f
