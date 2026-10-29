"""Mode 2: a whole Optuna search inside one queue job on a Colab L4 (cifarq.tune.on_vm): it measures how many trainings
the L4 runs at once with gain, then keeps that many trials in flight on the VM's GPU. The study (study.db) and its
summary come back in the job's directory."""
import argparse
import json
import shutil
import time

from common import L4A, RESULTS, data, ensure_executor, queue_executor, save

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=80)
    ap.add_argument('--driver', default='optuna', choices=('optuna', 'ray'),
                    help="optuna (its pruners) or ray (Ray Tune's schedulers, Optuna's sampler through OptunaSearch)")
    ap.add_argument('--pruner', default='none', choices=('none', 'rungs', 'asha', 'hyperband', 'median'),
                    help='early stopping by the study (each trial reports its validation accuracy every epoch)')
    ap.add_argument('--minutes', type=float, help='stop asking for trials after this long (a fixed budget)')
    ap.add_argument('--concurrent', type=int, help="trainings at once instead of measuring it (default: the card's stored "
                    "count, solve-queue cards.slots(..., 'cifar-trial'), else measured and stored)")
    ap.add_argument('--executor', default=L4A, help='the executor (colab:NAME without colab:)')
    ap.add_argument('--gpu', default='L4', help="the executor's card when it is registered here (L4, G4, A100, T4)")
    ap.add_argument('--high-mem', action='store_true', help="the card's high-RAM shape (the A100 80 GB)")
    ap.add_argument('--resume', action='store_true', help="continue the latest mode 2 run's study (the last study.db its "
                                                            'job brought back) on a new job, after the old one died')
    a = ap.parse_args()
    ensure_executor(a.executor, gpu=a.gpu, shape='high-mem' if a.high_mem else None)
    if a.concurrent is None:
        from solve_queue import cards
        stored = (cards.card(a.gpu, 'high-mem' if a.high_mem else None).get('slots') or {}).get('cifar-trial')
        if stored:
            a.concurrent = stored
            print(f'{a.gpu}: {stored} trainings at once (stored)', flush=True)
    resume = {}
    if a.resume:
        dbs = sorted((RESULTS / 'jobs' / 'on-vm').glob('*/study.db'), key=lambda p: p.stat().st_mtime)
        if not dbs:
            raise SystemExit('no earlier mode 2 study to resume')
        # a frozen copy, the new job's data (data never changes): the old job's file may still be written to
        frozen = RESULTS / 'resume' / f'{dbs[-1].parent.name}-{time.strftime("%Y%m%d-%H%M%S")}.db'
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(dbs[-1], frozen)
        resume = {'resume_from': str(frozen)}
        print(f'resuming from {dbs[-1]} (as {frozen})', flush=True)
    t0 = time.time()
    # the whole GPU for the search: its own trials share it, nothing else should
    # (progress.json and study.db come back every two minutes while it runs: scripts/watch.py --mode2 reads the study)
    # the study's consistent copy is checkpointed to the bucket as it changes: a lost VM's rerun continues from it
    with queue_executor(RESULTS / 'jobs' / 'on-vm', f'colab:{a.executor}', minutes=(a.minutes or 240) + 15, gpu_mb=22000,
                        watch=['progress.json', 'study.db'], report=['best_value', 'completed', 'pruned'],
                        checkpoint={'include': ['study-ckpt.db', 'progress.json'], 'every_s': 60}) as ex:
        if resume:
            ex.data = ex.data + [resume['resume_from']]
        if a.driver == 'ray':
            if a.pruner in ('median', 'rungs'):
                raise SystemExit('the Ray driver takes asha, hyperband or none')
            future = ex.submit('cifarq.tune:on_vm_ray', n_trials=a.trials, data=str(data.path()), scheduler=a.pruner,
                               minutes=a.minutes, concurrent=a.concurrent or 1)
        else:
            future = ex.submit('cifarq.tune:on_vm', n_trials=a.trials, data=str(data.path()), pruner=a.pruner,
                               minutes=a.minutes, concurrent=a.concurrent, **resume)
        print(f'job {future.job_id}: {future.job}', flush=True)
        seen = None
        while not future.done():
            time.sleep(20)
            p = future.job / 'progress.json'
            if p.exists() and p.stat().st_mtime != seen:
                seen, doc = p.stat().st_mtime, json.loads(p.read_text())
                print(f"{time.strftime('%H:%M:%S')}  {doc['complete']}/{doc['n_trials']} trials ({doc.get('pruned', 0)} pruned), {doc['concurrent']} at once; "
                      f"best {doc['best_value']} (trial {doc['best_trial']}, test {doc['best_test_acc']})", flush=True)
        result = future.result()
    result['wall_seconds_from_submit'] = round(time.time() - t0, 1)
    if not a.concurrent and result.get('calibration') and not result['calibration'][0].get('given'):
        # the card's measured trainings at once, kept for the next run (solve-queue cards.json: --concurrent unneeded)
        from solve_queue import cards
        cards.set_slots(a.gpu, 'high-mem' if a.high_mem else None, 'cifar-trial', int(result['concurrent']))
        print(f'recorded: {a.gpu} runs {result["concurrent"]} cifar trainings at once', flush=True)
    if (future.job / 'study.db').exists():
        shutil.copy(future.job / 'study.db', RESULTS / 'on-vm-study.db')
    print(json.dumps(result, indent=1))
    print('saved', save('on-vm', result), 'and', RESULTS / 'on-vm-study.db')
