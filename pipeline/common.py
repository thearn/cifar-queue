"""What the pipeline's stages share: an executor per stage from configs/compute.toml (solve-queue's ``QueueExecutor``
over the stage's targets, or ``Direct``: in this process, no queue), and the run record each stage writes to
results/<stage>/<time>.json (git commit and dirty flag, dataset hash, versions, settings, results)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import Executor, Future
from contextlib import chdir
from importlib import import_module
from pathlib import Path

from cifarq import paths
from cifarq.data import dataset

ROOT, RESULTS = paths.ROOT, paths.RESULTS
SRC = Path(import_module('cifarq').__file__).resolve().parents[1]
COMPUTE = paths.config('compute')
os.environ.setdefault('SOLVEQ_SESSION', COMPUTE['session'])


class Direct(Executor):
    """The queue's executor interface in this process, one call at a time: each call runs in its own directory under
    ``root`` (as a queue job does: what it writes there is its output), its future's ``job`` that directory."""

    slots = 1

    def __init__(self, root: Path, **hints):
        self.root, self._n = Path(root), 0
        self.root.mkdir(parents=True, exist_ok=True)

    def submit(self, fn, /, **kwargs) -> Future:
        kwargs.pop('job_options', None)
        module, _, function = fn.partition(':')
        self._n += 1
        job = self.root / f'{time.strftime("%Y%m%d-%H%M%S")}-{self._n:05d}-{function}'
        job.mkdir(parents=True)
        future = Future()
        future.job, future.target = job, 'direct'
        future.set_running_or_notify_cancel()
        try:
            with chdir(job):
                result = getattr(import_module(module), function)(**kwargs)
            (job / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
            future.set_result(result)
        except Exception as e:                                  # (a failed call fails its future, as a failed job does)
            future.set_exception(e)
        return future


def ensure_executors(targets: dict):
    """Register the Colab executors ``targets`` names (no VM until a job needs one), each with as many parallel slots
    as the stage gives it."""
    from solve_queue import executors
    known = executors.load()
    for t, slots in targets.items():
        if not t.startswith('colab:'):
            continue
        name, spec = t[6:], COMPUTE['executors'][t[6:]]
        rec = known.get(name)
        if rec is None:
            executors.add(name, gpu=spec['gpu'], idle_stop_minutes=spec.get('idle_minutes', 5), parallel=slots,
                          shape=spec.get('shape'), log=lambda s: None)
        elif slots > 1 and rec.get('parallel') != slots:
            executors.update(name, parallel=slots)


def stage_targets(stage: str, override: list[str] | None = None) -> dict:
    """{target: slots} for ``stage``: ``override`` (``--targets local colab:NAME=2 ...``) or its compute.toml line."""
    if override:
        return {t.partition('=')[0]: int(t.partition('=')[2] or 1) for t in override}
    return dict(COMPUTE['stages'][stage]['targets'])


def executor(stage: str, targets: dict, data=(), send=(), **hints) -> Executor:
    """The stage's executor: ``Direct`` for ``{"direct": n}``, else a ``QueueExecutor`` over ``targets`` that sends the
    package's code (src/) and ``data`` (files that never change, once per VM by content) with each job."""
    root = RESULTS / 'jobs' / stage
    hints = {**{k: v for k, v in COMPUTE['stages'].get(stage, {}).items() if k != 'targets'}, **hints}
    if 'direct' in targets:
        if len(targets) > 1:
            raise SystemExit('direct runs in this process: it cannot share a stage with queue targets')
        return Direct(root)
    from solve_queue.futures import QueueExecutor
    ensure_executors(targets)
    return QueueExecutor(root, targets=targets, path=[SRC, *send], data=[str(d) for d in data], **hints)


def targets_arg(ap: argparse.ArgumentParser):
    ap.add_argument('--targets', nargs='+', default=None, metavar='TARGET[=SLOTS]',
                    help='where the jobs run: direct, local, colab:NAME (default: configs/compute.toml)')


def sets_arg(ap: argparse.ArgumentParser):
    ap.add_argument('--set', nargs='+', default=[], metavar='NAME=VALUE', help='hyperparameters (JSON values)')


def parse_sets(sets: list[str]) -> dict:
    """``["epochs=16", "optimizer=adamw"]`` as {"epochs": 16, "optimizer": "adamw"} (JSON values, else strings)."""
    params = {}
    for kv in sets:
        k, _, v = kv.partition('=')
        try:
            params[k] = json.loads(v)
        except json.JSONDecodeError:
            params[k] = v
    return params


def provenance() -> dict:
    """The git commit (and whether the tree had uncommitted changes), the dataset's hash, and the versions."""
    def git(*args):
        r = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    import torch
    import cifarq
    ds = paths.read_manifest('dataset') or {}
    return {'commit': git('rev-parse', '--short', 'HEAD'), 'dirty': bool(git('status', '--porcelain', '--untracked-files=no')),
            'dataset': ds.get('sha256'), 'package': cifarq.__version__, 'torch': torch.__version__,
            'python': sys.version.split()[0], 'argv': sys.argv}


def record(stage: str, out: dict) -> Path:
    """Write ``out`` with the provenance as results/<stage>/<time>.json; its path."""
    d = RESULTS / stage
    d.mkdir(parents=True, exist_ok=True)
    f = d / f'{time.strftime("%Y%m%d-%H%M%S")}.json'
    f.write_text(json.dumps({'stage': stage, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'provenance': provenance(), **out},
                            indent=1, default=str) + '\n')
    return f


def latest(stage: str) -> dict:
    runs = sorted((RESULTS / stage).glob('*.json'))
    if not runs:
        raise SystemExit(f'no results/{stage}/*.json: run that stage first')
    return json.loads(runs[-1].read_text())


def dataset_file() -> Path:
    f = dataset.path()
    if not f.exists():
        raise SystemExit('no dataset: run pipeline/30_dataset.py first')
    return f
