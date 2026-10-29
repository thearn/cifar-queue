"""PBT as a second stage on the G4 (docs/PLAN-mode2.md step 7), after a probe of the same configuration: its accuracy
over 3 seeds (the baseline), batch sizes 256-4096, and one training against two at once (cifarq.pbt). The
configuration is the best of the G4 card-matrix runs (results/g4-best.json, train's names).

    python scripts/5_pbt_g4.py [--population 8] [--concurrent 4] [--interval 3]
"""
import argparse
import json
import time

from common import RESULTS, data, ensure_executor, queue_executor, save

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--executor', default='cifar-g4'); ap.add_argument('--gpu', default='G4')
    ap.add_argument('--population', type=int, default=8); ap.add_argument('--concurrent', type=int, default=4)
    ap.add_argument('--interval', type=int, default=3)
    ap.add_argument('--skip-probe', action='store_true')
    a = ap.parse_args()
    base = json.loads((RESULTS / 'g4-best.json').read_text())
    ensure_executor(a.executor, gpu=a.gpu, idle_minutes=1)
    out = {'base': base}
    with queue_executor(RESULTS / 'jobs' / 'pbt', f'colab:{a.executor}', minutes=30, gpu_mb=20000, host_mb=16000,
                        report=['seconds']) as ex:
        if not a.skip_probe:
            f = ex.submit('cifarq.pbt:probe', base=base, data=str(data.path()))
            print(f'probe: job {f.job_id}', flush=True)
            out['probe'] = f.result()
            print(json.dumps(out['probe'], indent=1), flush=True)
        f = ex.submit('cifarq.pbt:run_pbt', base=base, data=str(data.path()), population=a.population,
                      concurrent=a.concurrent, interval=a.interval)
        print(f'pbt: job {f.job_id}', flush=True)
        out['pbt'] = f.result()
        print(json.dumps(out['pbt']['best'], indent=1), flush=True)
    print('saved', save('pbt', out))
