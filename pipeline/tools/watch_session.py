"""Watch a session's queue jobs and end when one needs attention (run in the background, so its end wakes whoever
started it): the session's active jobs (queue labels starting ``SESSION/``) handed to ``solveq wait --until
first-failure --stall-min M``, again as new ones are submitted. Ends with ``solveq wait``'s code on a failure (1) or a
stall (2), and with 0 after ``--idle-min`` minutes with no active job (the queue ran dry: time for the next step).

    python pipeline/tools/watch_session.py fparch [--stall-min 15] [--idle-min 20]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time


def active(session: str) -> list[str]:
    """The ids of the session's queued and running jobs (pueue's task list: discovery only, the waiting is solveq's)."""
    out = subprocess.run(['pueue', 'status', '--json'], capture_output=True, text=True)
    if out.returncode:
        return []
    ids = []
    for i, t in json.loads(out.stdout)['tasks'].items():
        st = t['status'] if isinstance(t['status'], str) else next(iter(t['status']), '')
        if (t.get('label') or '').startswith(f'{session}/') and st in ('Queued', 'Running', 'Stashed', 'Locked'):
            ids.append(i)
    return sorted(ids, key=int)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('session')
    ap.add_argument('--stall-min', type=float, default=15)
    ap.add_argument('--idle-min', type=float, default=20)
    a = ap.parse_args()
    idle_since = None
    while True:
        ids = active(a.session)
        if not ids:
            idle_since = idle_since or time.time()
            if time.time() - idle_since > 60 * a.idle_min:
                print(f'{time.strftime("%H:%M")} no active {a.session} jobs for {a.idle_min:g} min: the queue ran dry', flush=True)
                raise SystemExit(0)
            time.sleep(60)
            continue
        idle_since = None
        print(f'{time.strftime("%H:%M")} waiting on {" ".join(ids)}', flush=True)
        r = subprocess.run(['solveq', 'wait', *ids, '--until', 'first-failure', '--stall-min', str(a.stall_min)])
        if r.returncode in (1, 2):
            print(f'{time.strftime("%H:%M")} {"a failure" if r.returncode == 1 else "a stall"} among {" ".join(ids)}', flush=True)
            raise SystemExit(r.returncode)
        time.sleep(5)


if __name__ == '__main__':
    main()
