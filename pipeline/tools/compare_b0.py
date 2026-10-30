"""A candidate against B0 by the session's win rule (docs/autoresearch/2026-10-29-footprint-arch/PLAN.md §2.1): its
seeds' test and validation accuracies against B0's (E0.3), Welch's one-sided test, the one-sided 95 % lower bound of
the difference, and a bootstrap over the 10,000 test images of one saved model each (the candidate's median-validation
seed against results/models/B0), paired image by image.

    python pipeline/tools/compare_b0.py results/autoresearch/M4-fe2-t158/<time>.json [--b0 results/autoresearch/E0.3/...]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ROOT  # noqa: E402


def t_sf(t: float, dof: float) -> float:
    """Student's t upper tail, P(T > t), by integrating the density (no scipy in the project's environment)."""
    pdf = lambda x: math.exp(math.lgamma((dof + 1) / 2) - math.lgamma(dof / 2)) / math.sqrt(dof * math.pi) \
        * (1 + x * x / dof) ** (-(dof + 1) / 2)
    xs = np.linspace(t, t + 200, 4001)
    return float(np.trapezoid([pdf(x) for x in xs], xs)) if t < 200 else 0.


def t_ppf(q: float, dof: float) -> float:
    """The t quantile by bisection on ``t_sf``."""
    lo, hi = 0., 50.
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_sf(mid, dof) > 1 - q else (lo, mid)
    return (lo + hi) / 2


def welch(a, b) -> dict:
    """Mean difference a - b, Welch's one-sided p (a > b), and the one-sided 95 % lower bound of the difference."""
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = statistics.mean(a) - statistics.mean(b)
    return {'delta': d, 't': d / (va + vb) ** .5, 'dof': dof, 'p': t_sf(d / (va + vb) ** .5, dof),
            'lower95': d - t_ppf(.95, dof) * (va + vb) ** .5}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('candidate', help="seeds.py's result for the candidate (run with --save)")
    ap.add_argument('--b0', default=sorted(glob.glob(str(ROOT / 'results/autoresearch/E0.3/*.json')))[-1])
    ap.add_argument('--b0-model', default=str(ROOT / 'results/models/B0'))
    ap.add_argument('--boot', type=int, default=10000)
    a = ap.parse_args()
    cand, b0 = json.loads(Path(a.candidate).read_text()), json.loads(Path(a.b0).read_text())
    out = {}
    for k in ('test_acc', 'val_acc'):
        x, y = [r[k] for r in cand['runs']], [r[k] for r in b0['runs']]
        out[k] = {'candidate': [statistics.mean(x), statistics.stdev(x), len(x)], 'b0': [statistics.mean(y), statistics.stdev(y), len(y)],
                  **welch(x, y)}
    # the bootstrap over test images: the candidate's median-validation seed, B0's saved model
    from cifarq import footprint
    runs = sorted(cand['runs'], key=lambda r: r['val_acc'])
    model = Path(runs[len(runs) // 2]['job']) / 'model'
    d = np.load(sorted(glob.glob(str(ROOT / 'data/dataset/dataset-*.npz')))[-1])
    x, y = d['x_test'], d['y_test']
    x = x.transpose(0, 2, 3, 1) if x.shape[1] == 3 else x
    c1 = footprint.predictor(model, 'cuda').predict(x).argmax(1) == y
    c0 = footprint.predictor(a.b0_model, 'cuda').predict(x).argmax(1) == y
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(y), size=(a.boot, len(y)))
    diffs = (c1[idx].mean(1) - c0[idx].mean(1))
    out['paired_bootstrap'] = {'candidate_model': str(model), 'b0_model': a.b0_model, 'candidate_acc': float(c1.mean()),
                               'b0_acc': float(c0.mean()), 'delta': float(c1.mean() - c0.mean()),
                               'ci95': [float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))],
                               'p_one_sided': float((diffs <= 0).mean())}
    t = out['test_acc']
    out['win'] = bool(t['delta'] >= .002 and t['p'] < .05 and out['paired_bootstrap']['ci95'][0] > 0
                      and out['val_acc']['delta'] > 0 and len(cand['runs']) >= 5)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
