"""The sealed test set opened once for a session's final models (docs/autoresearch/2026-10-30-equivariant-small,
PLAN §2.1): each saved model's test accuracy at 1, 2 and 6 views (one look covers all views of the same network), the
mean, standard deviation and one-sided 95 % lower bound over seeds, and the targets' thresholds checked. Every look is
appended to data/holdout/test_looks.jsonl with its reason.

    python pipeline/tools/open_test.py --why "eqsm M6 finalist" results/jobs/train/00200-train/model ...
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT  # noqa: E402
from compare_b0 import t_ppf  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('models', nargs='+')
    ap.add_argument('--why', required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    from cifarq import airbench
    d = np.load(sorted(glob.glob(str(ROOT / 'data/dataset/dataset-*.npz')))[-1])
    x, y = d['x_test'], d['y_test']
    x = x.transpose(0, 2, 3, 1) if x.shape[1] == 3 else x
    rows = []
    for m in a.models:
        acc = {v: float((airbench.Predictor(m, a.device, tta_level=lv).predict(x).argmax(1) == y).mean())
               for lv, v in ((0, 1), (1, 2), (2, 6))}
        rows.append({'model': m, 'test_acc_by_views': acc})
        print(m, acc, flush=True)
    out = {'why': a.why, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'models': rows}
    for v in (1, 2, 6):
        xs = [r['test_acc_by_views'][v] for r in rows]
        mean = statistics.mean(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        lower = mean - t_ppf(.95, len(xs) - 1) * sd / len(xs) ** .5 if len(xs) > 1 else mean
        out[f'{v}_views'] = {'mean': mean, 'sd': sd, 'lower95': lower, 'n': len(xs),
                             'meets_96': bool(mean >= .96 and lower >= .958)}
    log = ROOT / 'data/holdout/test_looks.jsonl'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as f:
        f.write(json.dumps(out) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'models'}, indent=1))


if __name__ == '__main__':
    main()
