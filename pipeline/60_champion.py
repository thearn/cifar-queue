"""Stage 6: the champion. A study's best parameters (or Page's recipe, --reference) trained again over several seeds
as jobs, each saving its network (train(save='model'), in its job's directory, which comes back with the job); the
seed with the median validation accuracy is the champion (chosen on validation, never on test), copied to
results/models/<time>/. The spread over seeds says how far the search's lead over Page's recipe is seed noise: report
it with the result. The seeds train together as one job (``train.train_seeds``: one program vmapped over the seeds,
on one card); ``--separate`` makes a job of each seed instead (to spread them over several targets).

    python pipeline/60_champion.py --seeds 5                             # the best trial of results/tune-study.db
    python pipeline/60_champion.py --storage results/distributed-study.db --seeds 5     # mode 3's study
    python pipeline/60_champion.py --reference --seeds 5                 # Page's recipe the same way, for the comparison
    python pipeline/60_champion.py --params results/g4-best.json --seeds 3 --targets colab:cifar-g4   # a file's parameters
    python pipeline/60_champion.py --separate --targets local colab:cifar-l4a colab:cifar-l4b      # a job a seed
"""

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

from common import RESULTS, ROOT, dataset_file, executor, parse_sets, record, sets_arg, stage_targets, targets_arg

from cifarq import candidates, train, tune


def best_params(storage: str, name: str | None) -> tuple[str, dict, dict]:
    import optuna

    if name is None:  # the study with the best trial (the latest may be a short one)
        name = max((s for s in optuna.get_all_study_summaries(storage) if s.best_trial), key=lambda s: s.best_trial.value).study_name
    best = optuna.load_study(study_name=name, storage=storage).best_trial
    return (
        name,
        best.params,
        {'trial': best.number, 'val_acc': best.value, 'test_acc': best.user_attrs.get('test_acc'), 'device': best.user_attrs.get('device')},
    )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--storage', type=Path, default=RESULTS / 'tune-study.db', help='the study database (SQLite file)')
    ap.add_argument('--study', default=None, help='a study in it (default: the one with the best trial)')
    ap.add_argument('--reference', '--page', action='store_true', help="Page's recipe instead of the study's best")
    ap.add_argument('--params', type=Path, help="a JSON file of train's arguments instead of the study's best")
    ap.add_argument('--candidate', default=None, help="a recorded candidate's hyperparameters (candidates.jsonl), verified by this run")
    ap.add_argument('--separate', action='store_true', help='a job a seed, instead of one job training them all together')
    sets_arg(ap)  # (overrides: --set epochs=30)
    targets_arg(ap)
    a = ap.parse_args()
    search = None  # the search's own run of these parameters (seed 0)
    if a.reference:
        source, params = 'tune.REFERENCE', tune.REFERENCE
    elif a.params:  # (train's arguments already, not a trial's)
        source, params = str(a.params), json.loads(a.params.read_text())
    elif a.candidate:  # a search's trial (to translate) or a trained run's arguments
        source, params = f'candidates.jsonl#{a.candidate}', candidates.params_of(a.candidate)
    else:
        db = a.storage if a.storage.is_absolute() else (ROOT / a.storage).resolve()
        source, params, search = best_params(f'sqlite:///{db}', a.study)
    resolved = a.params or (a.candidate and not a.candidate.startswith(candidates.SEARCH_STAGES))
    params = {**(params if resolved else tune.train_params(params)), **parse_sets(a.set)}
    targets = stage_targets('champion', a.targets)
    print(f'{source}: {a.seeds} seeds over {targets}', flush=True)
    data = str(dataset_file())
    with executor('champion', targets, data=[dataset_file()], **tune.job_options(params)) as ex:
        if a.separate:
            futures = [(f, None) for f in (ex.submit(tune.TRAIN, data=data, save='model', **{**params, 'seed': s}) for s in range(a.seeds))]
        else:
            one = ex.submit(tune.TRAIN_SEEDS, seeds=list(range(a.seeds)), data=data, save='model', **{k: v for k, v in params.items() if k != 'seed'})
            futures = [(one, s) for s in range(a.seeds)]
        runs = []
        for s, (f, i) in enumerate(futures):
            r = f.result() if i is None else f.result()[i]
            runs.append(
                {
                    'seed': s,
                    'job': str(f.job),
                    'model': str(Path(f.job) / r['saved']),
                    'val_acc': r['val_acc'],
                    'test_acc': r['test_acc'],
                    'train_seconds': r['train_seconds'],
                    'device': r['device'],
                }
            )
            print(f'seed {s}: val {r["val_acc"]:.4f} test {r["test_acc"]:.4f} ({r["device"]}, {r["train_seconds"]} s)', flush=True)
    ranked = sorted(runs, key=lambda r: r['val_acc'])
    chosen = ranked[(len(ranked) - 1) // 2]
    model_dir = RESULTS / 'models' / time.strftime('%Y%m%d-%H%M%S')
    shutil.copytree(chosen['model'], model_dir)
    tests = [r['test_acc'] for r in runs]
    out = {
        'source': source,
        'candidate': a.candidate,
        'search_trial': search,
        'params': params,
        'runs': runs,
        'chosen': {**chosen, 'model': str(model_dir)},
        'test_mean': round(statistics.mean(tests), 4),
        'test_stdev': round(statistics.stdev(tests), 4) if len(tests) > 1 else 0.0,
        'hyperparameters': {**train.DEFAULTS, **params, 'seed': chosen['seed']},
        'targets': targets,
    }
    print(json.dumps({k: out[k] for k in ('chosen', 'test_mean', 'test_stdev')}, indent=1))
    print('saved', record('champion', out))
