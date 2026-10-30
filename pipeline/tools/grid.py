"""Many configurations over several seeds, submitted together from one process (the autoresearch sessions' curves and
screens): a JSON list of {"name": ..., "params": {...}} (each over a base configuration), every (name, seed) a queue
job, so a VM's slots stay full and no two submitting processes race for job directories. Per name, the per-seed
results and their mean and spread under results/autoresearch/<experiment>/<name>.json, as seeds.py writes them; the
source tree kept as refs/autoresearch/<experiment>.

    python pipeline/tools/grid.py M3-h1-curves --base results/autoresearch/eqsm/finalist-d1d1.json \
        --grid results/autoresearch/eqsm/m3-grid.json --seeds 1 2 3 --targets colab:cifar-g4
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS, dataset_file, executor, parse_sets, provenance, sets_arg, stage_targets, targets_arg  # noqa: E402
from seeds import snapshot  # noqa: E402


def summarise(runs: list[dict]) -> dict:
    keys = sorted({k for r in runs for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    out = {}
    for k in keys:
        v = [r[k] for r in runs if isinstance(r.get(k), (int, float))]
        out[k] = {'mean': round(statistics.mean(v), 5), 'sd': round(statistics.stdev(v), 5) if len(v) > 1 else None}
    views = [r['val_acc_by_views'] for r in runs if r.get('val_acc_by_views')]
    if views:
        out['val_acc_by_views'] = {v: round(statistics.mean(x[v] for x in views), 5) for v in views[0]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('experiment')
    ap.add_argument('--fn', default='cifarq.airbench:train')
    ap.add_argument('--base', help="a JSON file of the function's arguments every configuration starts from")
    ap.add_argument('--grid', required=True, help='a JSON list of {"name", "params"}')
    sets_arg(ap)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--stage', default='train')
    ap.add_argument('--gpu-mb', type=int)
    targets_arg(ap)
    a = ap.parse_args()
    base = json.loads(Path(a.base).read_text()) if a.base else {}
    base.update(parse_sets(a.set))
    grid = json.loads(Path(a.grid).read_text())
    out = RESULTS / 'autoresearch' / a.experiment
    out.mkdir(parents=True, exist_ok=True)
    snap, prov = snapshot(a.experiment), provenance()
    teachers = sorted({str(Path(t.strip()).resolve()) for g in grid for t in
                       str({**base, **g['params']}.get('kd_teacher') or '').split(',') if t.strip()})
    hints = {'minutes': 30, 'stall_min': 10, **({'gpu_mb': a.gpu_mb} if a.gpu_mb else {})}
    t0 = time.time()
    with executor(a.stage, stage_targets(a.stage, a.targets), data=[dataset_file(), *teachers], **hints) as ex:
        futures = {}
        for g in grid:
            params = {**base, **g['params']}
            if params.get('kd_teacher'):
                params['kd_teacher'] = ','.join(str(Path(t.strip()).resolve()) for t in params['kd_teacher'].split(','))
            for s in a.seeds:
                futures[(g['name'], s)] = (params, ex.submit(a.fn, data=str(dataset_file()), **{**params, 'seed': s}))
        print(f'{len(futures)} jobs submitted', flush=True)
        for g in grid:
            runs = []
            for s in a.seeds:
                params, f = futures[(g['name'], s)]
                try:
                    r = {k: v for k, v in f.result().items() if k != 'hyperparameters'}
                except Exception as error:                   # noqa: BLE001 - one failed run is recorded, not fatal
                    r = {'failed': repr(error)}
                    print(f"{g['name']} seed {s}: FAILED {error!r}", flush=True)
                runs.append({**r, 'seed': s, 'job': str(f.job)})
                print(f"{g['name']} seed {s}: val {r.get('val_acc')} views {r.get('val_acc_by_views')} "
                      f"{r.get('train_seconds')} s", flush=True)
            doc = {'experiment': a.experiment, 'name': g['name'], 'fn': a.fn, 'params': params, 'seeds': a.seeds,
                   'snapshot': snap, 'provenance': prov, 'summary': summarise(runs), 'runs': runs}
            (out / f"{g['name']}.json").write_text(json.dumps(doc, indent=1, default=str) + '\n')
            print(f"{g['name']}: {json.dumps(doc['summary'].get('val_acc_by_views'))}", flush=True)
    print(f'done in {time.time() - t0:.0f} s; results in {out}')


if __name__ == '__main__':
    main()
