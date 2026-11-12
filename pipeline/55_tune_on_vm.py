"""Stage 5, mode 2: a whole Optuna search inside one queue job on one card (cifarq.tune.on_vm): it measures how many
trainings the card runs at once with gain (or takes the card's stored count), then keeps that many trials in flight as
processes on its GPU, early stopping reaching the study there. The study (study.db) and its summary come back in the
job's directory; progress.json and study.db come back every two minutes while it runs (pipeline/tools/watch.py
--mode2), and the study's consistent copy is checkpointed to the bucket, so a lost VM's rerun continues from it.

    python pipeline/55_tune_on_vm.py --trials 80                         # configs/compute.toml's executor (an L4)
    python pipeline/55_tune_on_vm.py --target colab:cifar-g4 --cold --pruner rungs --minutes 45
    python pipeline/55_tune_on_vm.py --target local --concurrent 1 --trials 4 --minutes 5      # the 4060, as a test
    python pipeline/55_tune_on_vm.py --resume                            # the latest run's study, on a new job
    python pipeline/55_tune_on_vm.py --target local --family eqaug2 --seeds 2 --kd results/models/ab96-45k-s0 --seed-from results/eqaug-24h-study.db
    python pipeline/55_tune_on_vm.py --target colab:cifar-g4 --pruner rungs --minutes 45 --start-from best
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from common import (
    COMPUTE,
    RESULTS,
    dataset_file,
    executor,
    job_roots,
    read_study,
    record,
    seeded_study,
    session_board,
    session_score,
    stage_targets,
    start_from_arg,
    start_point,
)


def show_on_scoreboard(doc: dict, scored: dict, family: str) -> dict:
    """The runq session's scoreboard (the app's best-so-far line) from the job's progress.json: the first starting
    point's value as the baseline, then each new best; what is shown so far."""
    metric = 'val_acc' if family == 'b0' else 'val_1view'
    warm = doc.get('warm_values') or []
    if warm and 'baseline' not in scored:
        session_score(metric, baseline=warm[0], baseline_label="Page's recipe" if family == 'b0' else 'first starting point', direction='max')
        scored['baseline'] = warm[0]
    if doc.get('best_value') is not None and doc['best_value'] != scored.get('best'):
        session_score(metric, best=doc['best_value'], job=f'trial {doc["best_trial"]}')
        scored['best'] = doc['best_value']
    return scored


def show_on_leaderboard(future, family: str) -> None:
    """The runq session's leaderboard for this search (named for its family and job) from the job's study.db, which a
    local job writes in place and a VM's brings back every two minutes (``watch``)."""
    db = future.job / 'study.db'
    if not db.exists():
        return
    study, tmp = read_study(db)
    try:
        session_board(
            f'{family} {future.job.name}', study.trials, 'val_acc' if family == 'b0' else 'val_1view', family, job=getattr(future, 'job_id', None)
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=80)
    ap.add_argument('--target', default=None, help='local or colab:NAME (default: configs/compute.toml [stages.tune_on_vm])')
    ap.add_argument(
        '--driver',
        default='optuna',
        choices=('optuna', 'ray'),
        help="optuna (its pruners) or ray (Ray Tune's schedulers, Optuna's sampler through OptunaSearch)",
    )
    ap.add_argument(
        '--pruner',
        default='none',
        choices=('none', 'rungs', 'asha', 'hyperband', 'median'),
        help='early stopping by the study (each trial reports its validation accuracy every epoch)',
    )
    ap.add_argument(
        '--pruner-settings',
        type=json.loads,
        help='JSON: the pruner\'s settings (asha: {"min_resource": 30, '
        '"reduction_factor": 2}; rungs: {"rungs": [10, 30, 60], "keep": 0.67, "min_trials": 6})',
    )
    ap.add_argument('--no-schedule-free', action='store_true', help='the space without schedule-free training (to compare with runs made before it)')
    ap.add_argument('--cold', action='store_true', help="start without Page's recipe (for comparing methods and cards)")
    ap.add_argument('--seed', type=int, default=0, help="the sampler's seed (repeat cold runs with several)")
    ap.add_argument('--startup', type=int, default=16, help='random trials before TPE models (a 30-minute L4 search fits ~30)')
    ap.add_argument('--minutes', type=float, help='stop asking for trials after this long (a fixed budget)')
    ap.add_argument(
        '--concurrent',
        type=int,
        help="trainings at once instead of measuring it (default: the card's stored "
        "count, runq cards.slots(..., 'cifar-trial'), else measured and stored)",
    )
    ap.add_argument('--gpu-mb', type=int, default=None, help="the job's GPU memory (default: the stage's; 7000 on local)")
    ap.add_argument(
        '--resume',
        action='store_true',
        help="continue the latest mode 2 run's study (the last study.db its job brought back) on a new job, after the old one died",
    )
    ap.add_argument('--family', default='b0', help='b0 (the ResNet-9 space) or a cifarq.families space: airbench, resnet9, convmixer')
    ap.add_argument('--cap', type=int, default=2_000_000, help='stored parameters at most (families; over it a trial is pruned at once)')
    ap.add_argument('--max-epochs', type=float, default=40, help='the epochs range ends here (families)')
    ap.add_argument('--kd', default='', help="a teacher's saved directory (families: distillation enters the space)")
    ap.add_argument('--resume-from', help="a study database to continue (instead of --resume's latest)")
    ap.add_argument('--warm-from', help="an earlier study's database: its best --warm-top trials are the starting points")
    ap.add_argument('--warm-top', type=int, default=4)
    ap.add_argument('--seed-from', type=Path, help="an earlier study's database: its completed trials that fit this space enter as known points")
    ap.add_argument(
        '--seeds',
        type=int,
        default=1,
        choices=(1, 2),
        help="2: each trial on seeds 0 and 1 (seed 1 for survivors of seed 0's early stopping), the mean as its value, test sealed",
    )
    ap.add_argument(
        '--warm-set',
        type=json.loads,
        default={},
        help='JSON: values set in every starting point (with --kd: {"kd": true, "kd_weight": 0.5, "kd_temp": 4.0})',
    )
    start_from_arg(ap)
    a = ap.parse_args()
    if a.start_from and (a.family != 'b0' or a.warm_from or a.driver == 'ray'):
        raise SystemExit('--start-from: the b0 space with the Optuna driver, without --warm-from')
    start = start_point(a.start_from, schedule_free=not a.no_schedule_free)
    slot_key = 'cifar-trial' if a.family == 'b0' else f'cifar-{a.family}'
    target = a.target or next(iter(stage_targets('tune_on_vm')))
    gpu = shape = None
    if target.startswith('colab:'):
        spec = COMPUTE['executors'][target[6:]]
        gpu, shape = spec['gpu'], spec.get('shape')
        if a.concurrent is None:
            from runq.cloud import cards

            stored = (cards.card(gpu, shape).get('slots') or {}).get(slot_key)
            if stored:
                a.concurrent = stored
                print(f'{gpu}: {stored} trainings at once (stored)', flush=True)
    elif a.concurrent is None:
        a.concurrent = 1  # (the 4060: two at once gave 5 % more throughput in total)
    resume, seeded_best = {}, None
    if a.seed_from:
        if a.resume or a.resume_from or a.warm_from or a.start_from:
            raise SystemExit('--seed-from starts a new study: not with --resume, --resume-from, --warm-from or --start-from')
        import functools

        from cifarq import families

        seeded = RESULTS / 'resume' / f'seeded-{a.family}-{time.strftime("%Y%m%d-%H%M%S")}.db'
        space_fn = functools.partial(families.SPACES[a.family], max_epochs=a.max_epochs, kd=a.kd and 'teacher')
        kept, total, seeded_best = seeded_study(a.seed_from.expanduser(), space_fn, seeded)
        print(f'seeded from {a.seed_from}: {kept} of {total} completed trials fit the {a.family} space', flush=True)
        resume = {'resume_from': str(seeded)}
    elif a.resume_from:
        frozen = RESULTS / 'resume' / f'{Path(a.resume_from).parent.name}-{time.strftime("%Y%m%d-%H%M%S")}.db'
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(a.resume_from, frozen)
        resume = {'resume_from': str(frozen)}
    elif a.resume:
        dbs = sorted(
            [db for stage in ('on-vm', 'tune_on_vm') for root in job_roots(stage) for db in root.glob('*/study.db')], key=lambda p: p.stat().st_mtime
        )
        if not dbs:
            raise SystemExit('no earlier mode 2 study to resume')
        # a frozen copy, the new job's data (data never changes): the old job's file may still be written to
        frozen = RESULTS / 'resume' / f'{dbs[-1].parent.name}-{time.strftime("%Y%m%d-%H%M%S")}.db'
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(dbs[-1], frozen)
        resume = {'resume_from': str(frozen)}
        print(f'resuming from {dbs[-1]} (as {frozen})', flush=True)
    warm_points = None
    if a.warm_from:
        import optuna

        storage = f'sqlite:///{Path(a.warm_from).resolve()}'
        old = optuna.load_study(study_name=optuna.study.get_all_study_names(storage)[-1], storage=storage)
        best = sorted((t for t in old.trials if t.state.name == 'COMPLETE'), key=lambda t: -t.value)[: a.warm_top]
        warm_points = [{**t.params, **a.warm_set} for t in best]
        print(f'starting from {a.warm_from} trials {[t.number for t in best]}', flush=True)
    elif start:
        from cifarq import tune

        warm_points = [*([] if a.cold else [tune.PAGE]), {**start[1], **a.warm_set}]
    t0 = time.time()
    data = str(dataset_file())
    hints = {
        'minutes': (a.minutes or 240) + 15,
        'watch': ['progress.json', 'study.db'],
        'report': ['best_value', 'completed', 'pruned'],
        'checkpoint': {'include': ['study-ckpt.db', 'progress.json'], 'every_s': 60},
    }
    if a.gpu_mb or target == 'local':
        hints['gpu_mb'] = a.gpu_mb or 7000
    # the whole GPU for the search: its own trials share it, nothing else should
    # (a teacher's directory is sent like the data, once per VM, at the same absolute path)
    with executor('tune_on_vm', {target: 1}, data=[dataset_file(), *resume.values(), *([str(Path(a.kd).resolve())] if a.kd else [])], **hints) as ex:
        if a.driver == 'ray':
            if a.pruner in ('median', 'rungs'):
                raise SystemExit('the Ray driver takes asha, hyperband or none')
            future = ex.submit(
                'cifarq.tune:on_vm_ray',
                n_trials=a.trials,
                data=data,
                scheduler=a.pruner,
                minutes=a.minutes,
                concurrent=a.concurrent or 1,
                warm=not a.cold,
                seed=a.seed,
                startup=a.startup,
            )
        else:
            future = ex.submit(
                'cifarq.tune:on_vm',
                n_trials=a.trials,
                data=data,
                pruner=a.pruner,
                minutes=a.minutes,
                concurrent=a.concurrent,
                pruner_settings=a.pruner_settings,
                schedule_free=not a.no_schedule_free,
                warm=not a.cold or bool(start),
                seed=a.seed,
                startup=a.startup,
                family=a.family,
                cap=a.cap,
                max_epochs=a.max_epochs,
                kd=a.kd and str(Path(a.kd).resolve()),
                warm_points=warm_points,
                seeds=a.seeds,
                **resume,
            )
        print(f'job {future.job}', flush=True)
        seen, scored = None, {}
        if seeded_best is not None:  # (the seeded study's first trials are known points, not starting points)
            session_score('val_1view' if a.family != 'b0' else 'val_acc', baseline=seeded_best, baseline_label='best known when seeded')
            scored['baseline'] = seeded_best
        while not future.done():
            time.sleep(20)
            p = future.job / 'progress.json'
            if p.exists() and p.stat().st_mtime != seen:
                seen, doc = p.stat().st_mtime, json.loads(p.read_text())
                print(
                    f'{time.strftime("%H:%M:%S")}  {doc["complete"]}/{doc["n_trials"]} trials ({doc.get("pruned", 0)} pruned), '
                    f'{doc["concurrent"]} at once; '
                    f'best {doc["best_value"]} (trial {doc["best_trial"]}, test {doc["best_test_acc"]})',
                    flush=True,
                )
                scored = show_on_scoreboard(doc, scored, a.family)
                show_on_leaderboard(future, a.family)
        result = future.result()
    result['wall_seconds_from_submit'] = round(time.time() - t0, 1)
    if gpu and not a.concurrent and result.get('calibration') and not result['calibration'][0].get('given'):
        # the card's measured trainings at once, kept for the next run (runq cards.json: --concurrent unneeded)
        from runq.cloud import cards

        cards.set_slots(gpu, shape, slot_key, int(result['concurrent']))
        print(f'recorded: {gpu} runs {result["concurrent"]} cifar trainings at once', flush=True)
    # (the study stays in the job's directory, which the record names: no shared copy for a later run to overwrite;
    # results/on-vm-study.db is the scripts/ layout's last mode 2 study, kept as it was)
    print(json.dumps(result, indent=1))
    print('saved', record('tune_on_vm', {'target': target, 'job': str(future.job), 'study': str(future.job / 'study.db'), 'result': result}))
