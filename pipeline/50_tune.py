"""Stage 5 (mode 3): the hyperparameter search. Optuna runs here; each trial is a job on the stage's targets (the 4060
and the Colab L4s at once: configs/compute.toml), placed where there is a free slot. The study is
results/tune-study.db (--storage: another, such as mode 3's results/distributed-study.db); Page's recipe
(tune.REFERENCE) is every new study's first trial unless --cold. A driver that dies loses nothing: --resume reattaches
the trials it left in flight (finished ones told, running ones followed, lost ones queued again) and goes on.

Early stopping (--prune) needs the trials to reach the study's SQLite file: targets on this machine only (direct,
local). For a pruned search on a Colab card, run the whole search inside one job there (pipeline/55_tune_on_vm.py).

    python pipeline/50_tune.py --trials 80 --targets local colab:cifar-l4a colab:cifar-l4b     # mode 3 as run
    python pipeline/50_tune.py --resume                                  # the latest study, after its driver died
    python pipeline/50_tune.py --storage results/distributed-study.db --resume --no-schedule-free
    python pipeline/50_tune.py --trials 20 --targets local --prune rungs --cold --seed 1
"""
import argparse
import collections
import json
import time
from pathlib import Path

import optuna

from common import RESULTS, ROOT, dataset_file, executor, record, stage_targets, targets_arg
from cifarq import tune
from solve_queue.hpo import make_pruner, optuna_optimize

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=80)
    targets_arg(ap)
    ap.add_argument('--storage', type=Path, default=RESULTS / 'tune-study.db', help='the study database (SQLite file)')
    ap.add_argument('--resume', nargs='?', const='latest', default=None, help='continue a study (its name, or the latest)')
    ap.add_argument('--prune', default=None, help='a solve_queue.hpo.PRUNERS name (rungs: tune.PRUNE settings)')
    ap.add_argument('--minutes', type=float, default=None, help='no new trial after this long')
    ap.add_argument('--seed', type=int, default=0, help="the sampler's seed (repeat cold runs with several)")
    ap.add_argument('--startup', type=int, default=16, help='random trials before TPE models')
    ap.add_argument('--cold', action='store_true', help="start without Page's recipe (for comparing methods and cards)")
    ap.add_argument('--no-schedule-free', action='store_true', help='the space before schedule-free training (studies made '
                    'before it, such as distributed-study.db)')
    a = ap.parse_args()
    targets = stage_targets('tune', a.targets)
    if a.prune and set(targets) - {'direct', 'local'}:
        raise SystemExit('--prune: the trials must reach the study database, so direct or local targets only')
    storage = f'sqlite:///{(ROOT / a.storage).resolve() if not a.storage.is_absolute() else a.storage}'
    RESULTS.mkdir(parents=True, exist_ok=True)
    pruner = make_pruner(a.prune, **(tune.PRUNE if a.prune == 'rungs' else {})) if a.prune else None
    if a.resume:
        name = optuna.study.get_all_study_names(storage)[-1] if a.resume == 'latest' else a.resume
        study = optuna.load_study(study_name=name, storage=storage, sampler=tune.sampler(a.seed, a.startup), pruner=pruner)
        print(f'resuming {name}: {len(study.trials)} trials so far', flush=True)
    else:
        study = optuna.create_study(direction=tune.DIRECTION, storage=storage, sampler=tune.sampler(a.seed, a.startup),
                                    pruner=pruner, study_name=f'tune-{time.strftime("%Y%m%d-%H%M%S")}')
        if not a.cold:
            study.enqueue_trial(tune.REFERENCE)
    print(f'trials in flight: {targets}', flush=True)
    space = (lambda t: tune.space(t, schedule_free=False)) if a.no_schedule_free else tune.space
    t0 = time.time()
    with executor('tune', targets, data=[dataset_file()]) as ex:
        optuna_optimize(study, tune.TRAIN, space, n_trials=a.trials, executor=ex, objective=tune.OBJECTIVE,
                        fixed={'data': str(dataset_file())}, job_options=tune.job_options, resume=True,
                        report_to=storage if a.prune else None, smooth=bool(a.prune), minutes=a.minutes)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = study.best_trial
    out = {'study': study.study_name, 'storage': storage, 'trials': len(study.trials), 'completed': len(done),
           'pruned': sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
           'best': {'trial': best.number, 'value': best.value, 'params': best.params,
                    'test_acc': best.user_attrs.get('test_acc'), 'train_seconds': best.user_attrs.get('train_seconds'),
                    'device': best.user_attrs.get('device')},
           'by_device': dict(collections.Counter(t.user_attrs.get('device', '?') for t in done)),
           'minutes_to': tune._minutes_to(study, (0.92, 0.93)), 'start': 'cold' if a.cold else 'Page', 'seed': a.seed,
           'targets': targets, 'search_seconds': round(time.time() - t0, 1)}
    print(json.dumps({k: out[k] for k in ('best', 'by_device', 'minutes_to')}, indent=1))
    print('saved', record('tune', out))
