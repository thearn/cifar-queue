"""Stage 6: the champion. A study's best parameters (or Page's recipe, --reference) trained again over several seeds
as jobs, each saving its network (train(save='model'), in its job's directory, which comes back with the job); the
seed with the median validation accuracy is the champion (chosen on validation, never on test), copied to
results/models/<time>/. The spread over seeds says how far the search's lead over Page's recipe is seed noise: report
it with the result.

    python pipeline/60_champion.py --seeds 5                             # the best trial of results/tune-study.db
    python pipeline/60_champion.py --storage results/distributed-study.db --seeds 5     # mode 3's study
    python pipeline/60_champion.py --reference --seeds 5                 # Page's recipe the same way, for the comparison
    python pipeline/60_champion.py --targets local colab:cifar-l4a colab:cifar-l4b      # the 4060 and two L4s
"""
import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

from common import RESULTS, ROOT, dataset_file, executor, parse_sets, record, sets_arg, stage_targets, targets_arg
from cifarq import train, tune


def best_params(storage: str, name: str | None) -> tuple[str, dict, dict]:
    import optuna
    if name is None:                                             # the study with the best trial (the latest may be a short one)
        name = max((s for s in optuna.get_all_study_summaries(storage) if s.best_trial),
                   key=lambda s: s.best_trial.value).study_name
    best = optuna.load_study(study_name=name, storage=storage).best_trial
    return name, best.params, {'trial': best.number, 'val_acc': best.value, 'test_acc': best.user_attrs.get('test_acc'),
                               'device': best.user_attrs.get('device')}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--storage', type=Path, default=RESULTS / 'tune-study.db', help='the study database (SQLite file)')
    ap.add_argument('--study', default=None, help='a study in it (default: the one with the best trial)')
    ap.add_argument('--reference', '--page', action='store_true', help="Page's recipe instead of the study's best")
    sets_arg(ap)                                                 # (overrides: --set epochs=30)
    targets_arg(ap)
    a = ap.parse_args()
    search = None                                                # the search's own run of these parameters (seed 0)
    if a.reference:
        source, params = 'tune.REFERENCE', tune.REFERENCE
    else:
        db = a.storage if a.storage.is_absolute() else (ROOT / a.storage).resolve()
        source, params, search = best_params(f'sqlite:///{db}', a.study)
    params = {**tune.train_params(params), **parse_sets(a.set)}
    targets = stage_targets('champion', a.targets)
    print(f'{source}: {a.seeds} seeds over {targets}', flush=True)
    with executor('champion', targets, data=[dataset_file()], **tune.job_options(params)) as ex:
        futures = [ex.submit(tune.TRAIN, data=str(dataset_file()), save='model', **{**params, 'seed': s})
                   for s in range(a.seeds)]
        runs = []
        for s, f in enumerate(futures):
            r = f.result()
            runs.append({'seed': s, 'job': str(f.job), 'model': str(Path(f.job) / r['saved']), 'val_acc': r['val_acc'],
                         'test_acc': r['test_acc'], 'train_seconds': r['train_seconds'], 'device': r['device']})
            print(f"seed {s}: val {r['val_acc']:.4f} test {r['test_acc']:.4f} ({r['device']})", flush=True)
    ranked = sorted(runs, key=lambda r: r['val_acc'])
    chosen = ranked[(len(ranked) - 1) // 2]
    model_dir = RESULTS / 'models' / time.strftime('%Y%m%d-%H%M%S')
    shutil.copytree(chosen['model'], model_dir)
    tests = [r['test_acc'] for r in runs]
    out = {'source': source, 'search_trial': search, 'params': params, 'runs': runs,
           'chosen': {**chosen, 'model': str(model_dir)},
           'test_mean': round(statistics.mean(tests), 4), 'test_stdev': round(statistics.stdev(tests), 4) if len(tests) > 1 else 0.0,
           'hyperparameters': {**train.DEFAULTS, **params, 'seed': chosen['seed']}, 'targets': targets}
    print(json.dumps({k: out[k] for k in ('chosen', 'test_mean', 'test_stdev')}, indent=1))
    print('saved', record('champion', out))
