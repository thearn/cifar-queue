"""Mode 1: one training at fixed hyperparameters (cifarq.train.DEFAULTS, 16 epochs) on a Colab L4 through the queue."""
import json
import time

from common import L4A, RESULTS, data, ensure_executor, queue_executor, save
from cifarq.train import DEFAULTS

if __name__ == '__main__':
    ensure_executor(L4A)
    params = {**DEFAULTS, 'epochs': 16}
    t0 = time.time()
    with queue_executor(RESULTS / 'jobs' / 'single', f'colab:{L4A}', minutes=5) as ex:
        future = ex.submit('cifarq.train:train', data=str(data.path()), **params)
        print(f'job {future.job_id}: {future.job}', flush=True)
        result = future.result()
    result['wall_seconds_from_submit'] = round(time.time() - t0, 1)
    print(json.dumps({k: v for k, v in result.items() if k != 'history'}, indent=1))
    print('saved', save('single', result))
