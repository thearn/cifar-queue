"""One configuration over several seeds, as queue jobs (the autoresearch session's replications, screens and
confirmations): the per-seed results, their mean and spread, under results/autoresearch/<experiment>/, with a snapshot
of the source tree as a git ref (refs/autoresearch/<experiment>: the exact code, without committing or touching the
working tree).

    python pipeline/tools/seeds.py E1.2 --preset airbench94 --seeds 0 1 2 --targets local
    python pipeline/tools/seeds.py E0.3 --fn cifarq.train:train --params results/g4-best.json --seeds 0 1 2 3 4 --save
    python pipeline/tools/seeds.py E2.1 --preset airbench94 --set epochs=40 widths=[64,256,256] --seeds 0 --targets local
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ROOT, RESULTS, dataset_file, executor, parse_sets, provenance, sets_arg, targets_arg  # noqa: E402

PRESETS = {'airbench94': ('cifarq.airbench:train', 'AIRBENCH94'), 'airbench96': ('cifarq.airbench:train', 'AIRBENCH96')}


def snapshot(name: str) -> str | None:
    """The working tree as a commit object (``git stash create``: nothing stashed or changed), kept by a ref."""
    git = lambda *a: subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True, text=True)
    sha = git('stash', 'create', f'autoresearch {name}').stdout.strip() or git('rev-parse', 'HEAD').stdout.strip()
    if sha:
        git('update-ref', f'refs/autoresearch/{name}', sha)
    return sha or None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('experiment')
    ap.add_argument('--fn', default='cifarq.airbench:train')
    ap.add_argument('--preset', choices=sorted(PRESETS))
    ap.add_argument('--params', help="a JSON file of the function's arguments")
    sets_arg(ap)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--save', action='store_true', help="keep each seed's network (in its job's directory)")
    ap.add_argument('--stage', default='train', help="configs/compute.toml's stage for the hints (default: train)")
    targets_arg(ap)
    a = ap.parse_args()
    params, fn = {}, a.fn
    if a.preset:
        from cifarq import airbench
        fn, name = PRESETS[a.preset]
        params = dict(getattr(airbench, name))
    if a.params:
        params.update(json.loads(Path(a.params).read_text()))
    params.update(parse_sets(a.set))
    out = RESULTS / 'autoresearch' / a.experiment
    out.mkdir(parents=True, exist_ok=True)
    snap = snapshot(a.experiment)
    from common import stage_targets
    targets = stage_targets(a.stage, a.targets)
    t0 = time.time()
    with executor(a.stage, targets, data=[dataset_file()], minutes=30) as ex:
        futures = {s: ex.submit(fn, data=str(dataset_file()), **({'save': 'model'} if a.save else {}), **{**params, 'seed': s})
                   for s in a.seeds}
        for s, f in futures.items():
            print(f'seed {s}: job {f.job}', flush=True)
        runs = []
        for s, f in futures.items():
            r = f.result()
            r = {**r, 'seed': s, 'job': str(f.job)}
            if a.save and r.get('saved'):
                r['saved'] = str(Path(f.job) / r['saved'])
            runs.append(r)
            print(f"seed {s}: val {r.get('val_acc')} test {r.get('test_acc')} (no TTA {r.get('test_acc_notta')}) "
                  f"{r.get('train_seconds')} s on {r.get('device')}", flush=True)
    stat = lambda k: {'mean': round(statistics.mean(v), 5), 'sd': round(statistics.stdev(v), 5) if len(v) > 1 else None,
                      'min': min(v), 'max': max(v)} if (v := [r[k] for r in runs if r.get(k) is not None]) else None
    summary = {'experiment': a.experiment, 'fn': fn, 'params': params, 'seeds': a.seeds, 'targets': targets,
               'snapshot': snap, 'provenance': provenance(), 'wall_seconds': round(time.time() - t0, 1),
               **{k: stat(k) for k in ('val_acc', 'test_acc', 'val_acc_notta', 'test_acc_notta', 'train_seconds')},
               'runs': [{k: v for k, v in r.items() if k != 'hyperparameters'} for r in runs]}
    f = out / f'{time.strftime("%Y%m%d-%H%M%S")}.json'
    f.write_text(json.dumps(summary, indent=1, default=str) + '\n')
    print(json.dumps({k: summary[k] for k in ('val_acc', 'test_acc', 'test_acc_notta', 'train_seconds')}, indent=1))
    print('saved', f)


if __name__ == '__main__':
    main()
