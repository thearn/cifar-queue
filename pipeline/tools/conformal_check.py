"""Run with an environment holding surestop and scipy (not cifar-queue's: ~/.venvs/hpo-eval).

The Conformal Pruner's rule (surestop.KillRule) against our rung settings, replayed on cifar-queue's complete curves.
Curves resampled onto a 5 % grid of training (KillRule needs one step grid; our trials run 6-30 epochs). Repeated random
calibration/test splits: KillRule fitted on the calibration half, both rules scored on the test half by compute used
and the share of the test half's top 10 % they cut (the conditional false-kill rate)."""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/mnt/c/Users/trist/Documents/code/solve-queue/src')
from solve_queue.hpo import simulate_rungs
from surestop import KillRule

curves = []
for f in Path('/mnt/c/Users/trist/Documents/code/cifar-queue/results/jobs/distributed').glob('*/result.json'):
    try:
        r = json.loads(f.read_text())
    except ValueError:
        continue
    h = r.get('history') or []
    if h and not r.get('pruned') and math.isfinite(r['val_acc']) and len(h) == r['hyperparameters']['epochs']:
        curves.append(np.array([x['val_acc'] for x in h]))
GRID = list(range(5, 101, 5))
def on_grid(c, smooth):
    s = np.maximum.accumulate(c) if smooth else c
    return np.array([s[max(1, math.ceil(p / 100 * len(c))) - 1] for p in GRID])
rng = np.random.default_rng(0)
print(f'{len(curves)} curves; 200 random splits, calibration 100 / test {len(curves) - 100}')
rows = {}
for split in range(200):
    idx = rng.permutation(len(curves)); cal, test = idx[:100], idx[100:]
    tc = [curves[i] for i in test]
    finals = np.array([c[-1] for c in tc]); good = finals >= np.quantile(finals, 0.9)
    for smooth in (False, True):
        Ycal = np.array([on_grid(curves[i], smooth) for i in cal]); Yt = np.array([on_grid(c, smooth) for c in tc])
        for alpha in (0.0125, 0.02, 0.03):
            for hold in (None, 0.9):
                try:
                    rule = KillRule(alpha=alpha, good_quantile=0.10, maximize=True, hold_until_resolved=hold).fit(Ycal)
                except Exception:
                    continue
                cut, cost = 0, 0.
                for c, y, g in zip(tc, Yt, good):
                    k = rule.kill_step(y)
                    stop = GRID[k] if k is not None else 100
                    cost += len(c) * stop / 100; cut += int(g and k is not None)
                key = f'conformal alpha {alpha}' + (' hold 0.9' if hold else '') + (' best-so-far' if smooth else '')
                rows.setdefault(key, []).append((cost / sum(len(c) for c in tc), cut / good.sum()))
    for rungs, keep in [((10, 30, 60), 2 / 3), ((20, 50), 1 / 2), ((10, 30, 90), 1 / 3)]:
        s = simulate_rungs(tc, rungs, keep)
        rows.setdefault(f'rungs {list(rungs)} keep {keep:.2f} best-so-far', []).append((s['compute'], 1 - s['top_kept'] / s['top']))
print(f'{"rule":48s} {"compute":>8s} {"top-10 % cut":>13s}')
for k, v in sorted(rows.items(), key=lambda kv: np.mean([x[0] for x in kv[1]])):
    a = np.array(v)
    print(f'{k:48s} {100 * a[:, 0].mean():7.0f} % {100 * a[:, 1].mean():11.0f} %')
