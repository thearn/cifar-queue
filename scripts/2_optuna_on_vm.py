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
    ap.add_argument('--resume', action='store_true', help="continue the latest mode 2 run's study (the last study.db its "
                                                            'job brought back) on a new job, after the old one died')
    a = ap.parse_args()
    ensure_executor(L4A)
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
    with queue_executor(RESULTS / 'jobs' / 'on-vm', f'colab:{L4A}', minutes=240, gpu_mb=22000,
                        watch=['progress.json', 'study.db']) as ex:
        if resume:
            ex.data = ex.data + [resume['resume_from']]
        future = ex.submit('cifarq.tune:on_vm', n_trials=a.trials, data=str(data.path()), **resume)
        print(f'job {future.job_id}: {future.job}', flush=True)
        seen = None
        while not future.done():
            time.sleep(20)
            p = future.job / 'progress.json'
            if p.exists() and p.stat().st_mtime != seen:
                seen, doc = p.stat().st_mtime, json.loads(p.read_text())
                print(f"{time.strftime('%H:%M:%S')}  {doc['complete']}/{doc['n_trials']} trials, {doc['concurrent']} at once; "
                      f"best {doc['best_value']} (trial {doc['best_trial']}, test {doc['best_test_acc']})", flush=True)
        result = future.result()
    result['wall_seconds_from_submit'] = round(time.time() - t0, 1)
    shutil.copy(future.job / 'study.db', RESULTS / 'on-vm-study.db')
    print(json.dumps(result, indent=1))
    print('saved', save('on-vm', result), 'and', RESULTS / 'on-vm-study.db')
