"""What the pipeline's stages share: an executor per stage from configs/compute.toml (runq's ``QueueExecutor``
over the stage's targets, or ``Direct``: in this process, no queue), and the run record each stage writes to
results/<stage>/<time>.json (git commit and dirty flag, dataset hash, versions, settings, results)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import Executor, Future
from contextlib import chdir
from importlib import import_module
from pathlib import Path

from cifarq import candidates, paths
from cifarq.data import dataset

ROOT, RESULTS = paths.ROOT, paths.RESULTS
SRC = Path(import_module('cifarq').__file__).resolve().parents[1]
COMPUTE = paths.config('compute')
os.environ.setdefault('RUNQ_SESSION', COMPUTE['session'])


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
        except Exception as e:  # noqa: BLE001 - a failed call fails its future, as a failed job does
            future.set_exception(e)
        return future


def ensure_executors(targets: dict):
    """Register the executors ``targets`` names from configs/compute.toml's [executors] (any provider: colab, gce, or
    sky with its clouds, spot and store; ``runq clouds``), each with as many parallel slots as the stage gives it."""
    from runq.queue import executors

    executors.ensure(targets, COMPUTE['executors'], log=lambda s: None)


def stage_targets(stage: str, override: list[str] | None = None) -> dict:
    """{target: slots} for ``stage``: ``override`` (``--targets local colab:NAME=2 ...``) or its compute.toml line."""
    if override:
        return {t.partition('=')[0]: int(t.partition('=')[2] or 1) for t in override}
    return dict(COMPUTE['stages'][stage]['targets'])


def executor(stage: str, targets: dict, data=(), send=(), **hints) -> Executor:
    """The stage's executor: ``Direct`` for ``{"direct": n}``, else a ``QueueExecutor`` over ``targets`` that sends the
    package's code (src/) and ``data`` (files that never change, once per VM by content) with each job."""
    hints = {**{k: v for k, v in COMPUTE['stages'].get(stage, {}).items() if k != 'targets'}, **hints}
    if 'direct' in targets:
        if len(targets) > 1:
            raise SystemExit('direct runs in this process: it cannot share a stage with queue targets')
        return Direct(RESULTS / 'jobs' / stage)
    from runq.queue.collect import job_dir
    from runq.search.futures import QueueExecutor

    ensure_executors(targets)
    # (queue jobs live under runq's own root since 2026-11-11, which refuses others: ``job_roots`` finds both)
    return QueueExecutor(job_dir(stage), targets=targets, path=[SRC, *send], data=[str(d) for d in data], **hints)


def job_roots(stage: str) -> list[Path]:
    """Where ``stage``'s job directories are: ``results/jobs/STAGE`` (direct runs, and queue jobs before 2026-11-11) and
    runq's root for the session (queue jobs since)."""
    roots = [RESULTS / 'jobs' / stage]
    try:
        from runq.queue.collect import job_dir
    except ImportError:
        return roots
    return [*roots, job_dir(stage)]


def targets_arg(ap: argparse.ArgumentParser):
    ap.add_argument(
        '--targets',
        nargs='+',
        default=None,
        metavar='TARGET[=SLOTS]',
        help='where the jobs run: direct, local, colab:NAME (default: configs/compute.toml)',
    )


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

    import flax
    import jax
    import optax

    import cifarq

    ds = paths.read_manifest('dataset') or {}
    return {
        'commit': git('rev-parse', '--short', 'HEAD'),
        'dirty': bool(git('status', '--porcelain', '--untracked-files=no')),
        'dataset': ds.get('sha256'),
        'package': cifarq.__version__,
        'jax': jax.__version__,
        'flax': flax.__version__,
        'optax': optax.__version__,
        'python': sys.version.split()[0],
        'argv': sys.argv,
    }


def record(stage: str, out: dict) -> Path:
    """Write ``out`` with the provenance as results/<stage>/<time>.json; its path."""
    d = RESULTS / stage
    d.mkdir(parents=True, exist_ok=True)
    f = d / f'{time.strftime("%Y%m%d-%H%M%S")}.json'
    rec = {'stage': stage, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'provenance': provenance(), **out}
    f.write_text(json.dumps(rec, indent=1, default=str) + '\n')
    candidates.note(stage, json.loads(json.dumps(rec, default=str)), f)  # the result kept in candidates.jsonl (in git)
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


def start_from_arg(ap: argparse.ArgumentParser):
    ap.add_argument(
        '--start-from',
        metavar='CANDIDATE',
        help="a candidates.jsonl entry ('best': the best search result) as a trial after Page's recipe (also with --cold)",
    )


def start_point(spec: str | None, schedule_free: bool = True) -> tuple[str, dict, float] | None:
    """The candidate ``spec`` (``start_from_arg``) as (its id, its hyperparameters, its value), checked against the search space;
    a candidate that does not fit stops the run with the reasons."""
    if not spec:
        return None
    from cifarq import tune

    try:
        entry = candidates.start_point(spec)
    except KeyError as e:
        raise SystemExit(str(e)) from None
    problems = tune.misfit(entry['params'], lambda t: tune.space(t, schedule_free=schedule_free))
    if problems:
        raise SystemExit(f'--start-from {entry["id"]} does not fit the search space: ' + '; '.join(problems))
    ds = (paths.read_manifest('dataset') or {}).get('sha256')
    stale = (
        ' (stale: scored on another dataset or by other code, so its value here is unknown)'
        if candidates.stale(entry, ds, candidates.code_identity())
        else ''
    )
    print(f'starting from {entry["id"]}: val_acc {entry["value"]}{stale}', flush=True)
    return entry['id'], dict(entry['params']), entry['value']


def session_score(metric: str, **fields) -> None:
    """Update the runq session's scoreboard entry ``metric`` (the app's best-so-far line), when the session
    (RUNQ_SESSION) is registered; a record, never a reason to stop a search."""
    from runq.campaigns import sessions

    try:
        sessions.score(metric, **fields)
    except (sessions.SessionError, OSError, ValueError) as e:
        print(f'(scoreboard not updated: {e})', flush=True)


# the leaderboard's columns: what tells this family's configurations apart at a glance
BOARD_COLUMNS = {
    'b0': ['width', 'epochs', 'optimizer', 'batch_size', 'activation', 'head'],
    'eqaug': ['w1', 'w2', 'w3', 'batch_size', 'lr', 'augment', 'kd_weight'],
}


def read_study(db: Path):
    """The newest study in ``db`` (a SQLite file another process may be writing), read from a consistent copy: the
    study and the copy's directory (the caller removes it)."""
    import sqlite3
    import tempfile

    import optuna

    tmp = Path(tempfile.mkdtemp()) / 'study.db'
    src, dst = sqlite3.connect(f'file:{db}?mode=ro', uri=True), sqlite3.connect(tmp)
    src.backup(dst)
    src.close()
    dst.close()
    storage = f'sqlite:///{tmp}'
    return optuna.load_study(study_name=optuna.study.get_all_study_names(storage)[-1], storage=storage), tmp.parent


def session_board(board: str, trials: list, metric: str, family: str, job=None) -> None:
    """The runq session's leaderboard ``board`` (the app's table of the best trials and the trial counts), when the
    session is registered; a record, never a reason to stop a search."""
    from runq.campaigns import sessions

    from cifarq import tune

    rows, counts = tune.standing(trials)
    try:
        sessions.leaderboard(board, rows, metric=metric, counts=counts, columns=BOARD_COLUMNS.get(family, []), job=job)
    except (sessions.SessionError, OSError, ValueError) as e:
        print(f'(leaderboard not updated: {e})', flush=True)


def completed_trials(source: Path) -> list:
    """The completed trials of a study: its SQLite database, or its ``export_trials.py`` export (``.jsonl``, a row a
    trial with its distributions, which git keeps where ``results/`` is not kept), as Optuna's frozen trials."""
    import optuna

    if source.suffix == '.jsonl':
        out = []
        for line in source.read_text().splitlines():
            row = json.loads(line)
            dists = {k: optuna.distributions.json_to_distribution(v) for k, v in row['distributions'].items()}
            t = optuna.trial.create_trial(params=row['params'], distributions=dists, value=row['value'], user_attrs=row.get('user_attrs', {}))
            t.number = row['number']
            out.append(t)
        return out
    old, tmp = read_study(source)
    try:
        return [t for t in old.trials if t.state.name == 'COMPLETE' and t.value is not None]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def seeded_study(src_db: Path, space_fn, out: Path, study_name: str = 'cifar-on-vm') -> tuple[int, int, float | None]:
    """A new study database ``out`` (named ``study_name``, as ``tune.on_vm`` opens it) holding the completed trials of
    the study in ``src_db`` (a database or an export: ``completed_trials``) that fit ``space_fn`` (``known_point``), as
    known points with their values: a search over reworked ranges continues from what the old one measured instead of
    from a few starting points. (kept, offered, the best kept value)."""
    import optuna

    done = completed_trials(src_db)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = optuna.create_study(study_name=study_name, storage=f'sqlite:///{out}', direction='maximize')
    kept, best = 0, None
    for t in done:
        params = known_point(t.params, space_fn)
        if params is None:
            continue
        fixed = optuna.trial.FixedTrial(params)
        space_fn(fixed)
        attrs = {k: v for k, v in t.user_attrs.items() if isinstance(v, (int, float, str, bool))}
        new.add_trial(
            optuna.trial.create_trial(
                params=params, distributions=fixed.distributions, value=t.value, user_attrs={**attrs, 'source': f'{src_db.name} trial {t.number}'}
            )
        )
        kept, best = kept + 1, t.value if best is None else max(best, t.value)
    return kept, len(done), best


def known_point(params: dict, space_fn) -> dict | None:
    """``params`` as a point of ``space_fn``, or None when it is not one: a parameter the space no longer asks for is
    dropped when the space fixes it at the same value (``srf`` True in ``eqaug2``), and refuses the point otherwise;
    any other misfit (a value outside a range or a choice) refuses it."""
    import optuna

    from cifarq import tune

    problems = tune.misfit(params, space_fn)
    gone = [p.split(':')[0] for p in problems if p.endswith(': not in the space')]
    if len(gone) < len(problems):
        return None
    kept = {k: v for k, v in params.items() if k not in gone}
    out = space_fn(optuna.trial.FixedTrial(kept))
    return kept if all(out.get(k, object()) == params[k] for k in gone) else None
