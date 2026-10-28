"""Watch a search as it runs: trials done, running and failed (by device), trials an hour, the best so far and the
top five. Reads an Optuna study's SQLite file: mode 3's (results/distributed-study.db, written here as trials end) or
mode 2's (the job's study.db, brought back from the VM every two minutes while it runs: job.json ``watch``).

    python scripts/watch.py                      # the newest study of results/distributed-study.db
    python scripts/watch.py --mode2              # the running mode 2 job's study
    python scripts/watch.py --once

For charts in a browser: uvx optuna-dashboard sqlite:///results/distributed-study.db
"""
import argparse
import collections
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

import optuna

RESULTS = Path(__file__).resolve().parents[1] / 'results'
KEYS = ('width', 'epochs', 'optimizer', 'sgd_lr', 'adamw_lr', 'batch_size', 'activation', 'mix', 'ema')


def newest_mode2_db() -> Path | None:
    dbs = sorted((RESULTS / 'jobs' / 'on-vm').glob('*/study.db'), key=lambda p: p.stat().st_mtime)
    return dbs[-1] if dbs else None


def load(db: Path):
    """The newest study in ``db``, read from a copy (a file brought back mid-write is copied, not opened in place)."""
    tmp = Path(tempfile.mkdtemp()) / 'study.db'
    src = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    dst = sqlite3.connect(tmp)
    src.backup(dst)
    src.close(); dst.close()
    storage = f'sqlite:///{tmp}'
    name = optuna.study.get_all_study_names(storage)[-1]
    return optuna.load_study(study_name=name, storage=storage), tmp


def report(db: Path) -> str:
    study, tmp = load(db)
    try:
        trials = study.trials
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    states = collections.Counter(t.state.name for t in trials)
    done = [t for t in trials if t.state.name == 'COMPLETE']
    lines = [f'{study.study_name}  ({db})', f'trials: {dict(states)}']
    if done:
        start = min(t.datetime_start for t in trials if t.datetime_start)
        end = max(t.datetime_complete for t in done)
        hours = max((end - start).total_seconds() / 3600, 1e-6)
        by = collections.Counter(t.user_attrs.get('device', '?') for t in done)
        lines.append(f'{len(done) / hours:.0f} trials an hour over {hours * 60:.0f} min; by device: {dict(by)}')
        top = sorted(done, key=lambda t: t.value, reverse=True)[:5]
        best = top[0]
        lines.append(f'best: trial {best.number}  val {best.value:.4f}  test {best.user_attrs.get("test_acc")}  '
                     f'{best.user_attrs.get("train_seconds")} s on {best.user_attrs.get("device")}')
        lines.append('      ' + ', '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}' for k, v in best.params.items()))
        lines.append('top 5:')
        for t in top:
            lines.append(f'  #{t.number:3d}  {t.value:.4f}  ' + '  '.join(f'{k}={t.params[k]:.3g}' if isinstance(t.params[k], float)
                                                                          else f'{k}={t.params[k]}' for k in KEYS if k in t.params))
        running = [t for t in trials if t.state.name == 'RUNNING']
        if running:
            lines.append(f'running: {len(running)} (trials {", ".join(str(t.number) for t in running)})')
    return '\n'.join(lines)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('db', nargs='?', type=Path)
    ap.add_argument('--mode2', action='store_true', help="the newest mode 2 job's study.db")
    ap.add_argument('--every', type=float, default=30.)
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    while True:
        db = a.db or (newest_mode2_db() if a.mode2 else RESULTS / 'distributed-study.db')
        text = report(db) if db and db.exists() else f'no study yet ({db})'
        if a.once:
            print(text)
            break
        os.system('clear')
        print(time.strftime('%H:%M:%S'), text, sep='\n')
        time.sleep(a.every)
