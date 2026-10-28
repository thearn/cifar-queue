"""The champion: a study's best hyperparameters trained again over several seeds through the queue, each run saving its
network (``train(save='model')``, in its job's directory, which comes back with the job). The seed with the median
validation accuracy is the one published (chosen on validation, never on test), and the spread over seeds says how
far the search's lead over Page's recipe is seed noise. results/champion/<time>.json names the runs and the choice.

    python scripts/4_champion.py --seeds 5                  # the best mode 3 trial, here and on two L4s
    python scripts/4_champion.py --page --seeds 5           # Page's recipe the same way, for the comparison
    python scripts/4_champion.py --targets local            # this GPU only (the L4s busy, or Colab's two VMs in use)
"""
import argparse
import json
import statistics

import optuna

from common import L4A, L4B, RESULTS, data, ensure_executor, queue_executor, save
from cifarq import train, tune


def best_params(storage: str, name: str | None) -> tuple[str, dict, dict]:
    if name is None:                                         # the study with the best trial (the latest may be a short one)
        name = max((s for s in optuna.get_all_study_summaries(storage) if s.best_trial),
                   key=lambda s: s.best_trial.value).study_name
    best = optuna.load_study(study_name=name, storage=storage).best_trial
    return name, best.params, {'trial': best.number, 'val_acc': best.value, 'test_acc': best.user_attrs.get('test_acc'),
                               'device': best.user_attrs.get('device')}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--study', default=None, help='a mode 3 study (default: the one with the best trial)')
    ap.add_argument('--page', action='store_true', help="Page's recipe instead of the study's best")
    ap.add_argument('--targets', nargs='+', default=['local', f'colab:{L4A}', f'colab:{L4B}'],
                    help='where the seeds run, one at a time on each (default: this GPU and the two L4s)')
    a = ap.parse_args()
    search = None                                            # the search's own run of these parameters (seed 0)
    if a.page:
        source, params = 'tune.PAGE', tune.PAGE
    else:
        source, params, search = best_params(f'sqlite:///{RESULTS / "distributed-study.db"}', a.study)
    params = tune.train_params(params)
    targets = dict.fromkeys(a.targets, 1)
    for name in (L4A, L4B):
        if f'colab:{name}' in targets:
            ensure_executor(name)
    print(f'{source}: {a.seeds} seeds over {targets}', flush=True)
    with queue_executor(RESULTS / 'jobs' / 'champion', targets=targets, **tune.job_options(params)) as ex:
        futures = [ex.submit('cifarq.train:train', data=str(data.path()), save='model', **{**params, 'seed': s})
                   for s in range(a.seeds)]
        runs = []
        for s, f in enumerate(futures):
            r = f.result()
            runs.append({'seed': s, 'job': str(f.job), 'model': str(f.job / r['saved']), 'val_acc': r['val_acc'],
                         'test_acc': r['test_acc'], 'train_seconds': r['train_seconds'], 'device': r['device']})
            print(f"seed {s}: val {r['val_acc']:.4f} test {r['test_acc']:.4f} ({r['device']})", flush=True)
    vals = sorted(runs, key=lambda r: r['val_acc'])
    chosen = vals[(len(vals) - 1) // 2]
    tests = [r['test_acc'] for r in runs]
    out = {'source': source, 'search_trial': search, 'params': params, 'runs': runs, 'chosen': chosen,
           'test_mean': round(statistics.mean(tests), 4), 'test_stdev': round(statistics.stdev(tests), 4) if len(tests) > 1 else 0.0,
           'hyperparameters': {**train.DEFAULTS, **params, 'seed': chosen['seed']}}
    print(json.dumps({k: out[k] for k in ('chosen', 'test_mean', 'test_stdev')}, indent=1))
    print('saved', save('champion', out))
