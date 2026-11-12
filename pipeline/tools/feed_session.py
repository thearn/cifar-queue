"""Keep the runq session's leaderboard current for a search job whose driver predates the leaderboard (a driver of
pipeline/55_tune_on_vm.py from this commit on writes it itself): reads the job's study.db every ``--every`` seconds
and stops when the job writes summary.json, or after ``--hours``.

    python pipeline/tools/feed_session.py ~/.cache/runq/jobs/cifar/tune_on_vm/00003-on_vm --family eqaug --job 4457
"""

import argparse
import os
import shutil
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'pipeline'))  # (common: the stages' helpers)


def feed(job_dir: Path, family: str, job: int | None, every: float, hours: float) -> None:
    from common import read_study, session_board

    metric = 'val_acc' if family == 'b0' else 'val_1view'
    seen, end = None, time.time() + 3600 * hours
    while time.time() < end:
        db = job_dir / 'study.db'
        if db.exists() and db.stat().st_mtime != seen:
            seen = db.stat().st_mtime
            study, tmp = read_study(db)
            try:
                session_board(f'{family} {job_dir.name}', study.trials, metric, family, job=job)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        if (job_dir / 'summary.json').exists():
            return
        time.sleep(every)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('job_dir', type=Path)
    ap.add_argument('--family', default='eqaug')
    ap.add_argument('--job', type=int, help='the queue job id the leaderboard names')
    ap.add_argument('--every', type=float, default=120.0)
    ap.add_argument('--hours', type=float, default=30.0)
    a = ap.parse_args()
    os.environ.setdefault('RUNQ_SESSION', tomllib.loads((ROOT / 'configs' / 'compute.toml').read_text())['session'])
    feed(a.job_dir.expanduser(), a.family, a.job, a.every, a.hours)
