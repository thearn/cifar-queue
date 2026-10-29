"""Mode 3: Optuna here, its trials spread by the queue over this machine's GPU and two Colab L4s at once. Each GPU takes
as many trials at once as pays (``solve_queue.parallel.calibrate``): on each L4 what mode 2's calibration found (1
without it), on the 4060 ``--local-slots`` (1: two at once ran 5 % faster in total, so each trial at half speed). The
study is results/distributed-study.db. Colab lets this account hold two GPU VMs, so run it when mode 2 is not running
(a trial whose VM cannot be had goes to another target: ``QueueExecutor``)."""
import argparse
import collections
import json
import time

import optuna

from common import L4A, L4B, RESULTS, data, ensure_executor, queue_executor, save
from cifarq import tune
from solve_queue.hpo import optuna_optimize


def l4_slots() -> int:
    runs = sorted((RESULTS / 'on-vm').glob('*.json'))
    return json.loads(runs[-1].read_text())['concurrent'] if runs else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=80)
    ap.add_argument('--local-slots', type=int, default=1)
    ap.add_argument('--l4-slots', type=int, default=None)
    ap.add_argument('--l4s', type=int, default=2, choices=(1, 2), help='how many L4 executors (the account holds 2 GPU VMs)')
    ap.add_argument('--resume', nargs='?', const='latest', default=None,
                    help='continue a study (its name, or the latest) after its driver died: trials it left in flight '
                         'are reattached to their jobs, lost ones run again')
    a = ap.parse_args()
    k = a.l4_slots or l4_slots()
    l4s = (L4A, L4B)[:a.l4s]
    targets = {'local': a.local_slots, **{f'colab:{n}': k for n in l4s}}
    for name in l4s:
        ensure_executor(name, parallel=max(k, 1))
    print(f'trials in flight: {targets}', flush=True)
    storage = f'sqlite:///{RESULTS / "distributed-study.db"}'
    if a.resume:
        name = optuna.study.get_all_study_names(storage)[-1] if a.resume == 'latest' else a.resume
        study = optuna.load_study(study_name=name, storage=storage, sampler=tune.sampler())
        print(f'resuming {name}: {len(study.trials)} trials so far', flush=True)
    else:
        study = optuna.create_study(direction='maximize', storage=storage, sampler=tune.sampler(),
                                    study_name=f'distributed-{time.strftime("%Y%m%d-%H%M%S")}')
        study.enqueue_trial(tune.PAGE)                       # Page's recipe first: the search starts from the reference
    t0 = time.time()
    with queue_executor(RESULTS / 'jobs' / 'distributed', targets=targets) as ex:
        optuna_optimize(study, tune.TRAIN, tune.space, n_trials=a.trials, executor=ex,
                        fixed={'data': str(data.path())}, job_options=tune.job_options, resume=True)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    by_device = collections.Counter(t.user_attrs.get('device', '?') for t in done)
    best = study.best_trial
    out = {'best_value': best.value, 'best_params': best.params, 'best_test_acc': best.user_attrs.get('test_acc'),
           'best_train_seconds': best.user_attrs.get('train_seconds'), 'trials': len(study.trials), 'completed': len(done),
           'by_device': dict(by_device), 'targets': targets, 'search_seconds': round(time.time() - t0, 1),
           'study': study.study_name}
    print(json.dumps(out, indent=1))
    print('saved', save('distributed', out))
