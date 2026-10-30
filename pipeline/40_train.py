"""Stage 4: one training at train.DEFAULTS (with --set overrides), as one job: the round trip through the queue to
the stage's target, and the baseline every later result is compared with (mode 1 was the defaults at 16 epochs on an
L4: 88.9 % validation in 37 s).

    python pipeline/40_train.py --set epochs=16                         # configs/compute.toml's target (the 4060)
    python pipeline/40_train.py --targets colab:cifar-l4a --set epochs=16   # mode 1
    python pipeline/40_train.py --targets direct --set epochs=2         # in this process, no queue
    python pipeline/40_train.py --save                                  # keeping the network (in the job's directory)
"""
import argparse
import json

from common import dataset_file, executor, parse_sets, record, sets_arg, stage_targets, targets_arg
from cifarq import tune

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    targets_arg(ap)
    sets_arg(ap)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save', action='store_true', help="keep the network (in the job's directory)")
    a = ap.parse_args()
    params = parse_sets(a.set)
    targets = stage_targets('train', a.targets)
    with executor('train', targets, data=[dataset_file()]) as ex:
        f = ex.submit(tune.TRAIN, data=str(dataset_file()), **({'save': 'model'} if a.save else {}), **{'seed': a.seed, **params})
        r = f.result()
    summary = {k: v for k, v in r.items() if isinstance(v, (int, float, str))}
    print(json.dumps(summary, indent=1))
    print('saved', record('train', {'targets': targets, 'job': str(f.job), 'params': params, 'result': r}))
