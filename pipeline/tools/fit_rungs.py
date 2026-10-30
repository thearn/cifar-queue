"""Early stopping fitted to one family's own learning curves (the autoresearch session's M3): from a mode 2 study run
with a report-only pruner (``--pruner rungs --pruner-settings '{"rungs": [10], "keep": 1.0, "min_trials": 1}'``: every
epoch reported to the study, nothing cut), the complete trials' curves on a 5 % grid of training, the rank
correlation of each point with the final value, and successive halving replayed for candidate rung settings
(solve_queue.hpo.rung_check, simulate_rungs). Recommends the cheapest setting that keeps the single best trial and at
least ``--keep-top`` of the final top 10 %, as JSON for ``55_tune_on_vm.py --pruner rungs --pruner-settings``.

    python pipeline/tools/fit_rungs.py results/jobs/tune_on_vm/00012-on_vm/study.db [--keep-top 0.9]
"""
from __future__ import annotations

import argparse
import itertools
import json

import optuna
from solve_queue.hpo import rung_check, simulate_rungs

GRID = list(range(5, 101, 5))
CANDIDATES = [list(r) for n in (1, 2, 3) for r in itertools.combinations((10, 20, 30, 40, 50, 60, 70, 80), n)]


def curves_of(db: str, min_points: int = 4) -> list[list[float]]:
    """Each complete trial's reported values (step = percent of training) as its best value so far at each point of
    ``GRID`` (the last report at or before it), ending with its final value."""
    storage = f'sqlite:///{db}'
    out = []
    for name in optuna.study.get_all_study_names(storage):
        for t in optuna.load_study(study_name=name, storage=storage).trials:
            if t.state.name != 'COMPLETE' or len(t.intermediate_values) < min_points:
                continue
            steps = sorted(t.intermediate_values)
            curve, best = [], float('-inf')
            for p in GRID:
                seen = [t.intermediate_values[s] for s in steps if s <= p]
                best = max([best, *seen]) if seen else best
                curve.append(best if seen else float('nan'))
            curve[-1] = max(curve[-1], t.value)
            first = next((v for v in curve if v == v), t.value)
            out.append([first if v != v else v for v in curve])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('db')
    ap.add_argument('--keep-top', type=float, default=0.9, help='the share of the final top 10 %% a setting must keep')
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    curves = curves_of(a.db)
    print(f'{len(curves)} complete curves (a 5 % grid)')
    if len(curves) < 20:
        raise SystemExit('fewer than 20 complete curves: run the report-only phase longer')
    for row in rung_check(curves, points=tuple(range(10, 91, 10))):
        print(f"  {row['percent']:3d} %: rho {row['rho']:.2f}; keeping the top third cuts {row['top_cut']} of {row['top']}")
    rows = []
    for rungs, keep in itertools.product(CANDIDATES, (1 / 3, 1 / 2, 2 / 3)):
        s = simulate_rungs(curves, rungs, keep)
        rows.append({'rungs': rungs, 'keep': round(keep, 3), **s})
    ok = [r for r in rows if r['best_kept'] and r['top_kept'] >= a.keep_top * r['top']]
    for r in sorted(ok, key=lambda r: r['compute'])[:8]:
        print(f"rungs {r['rungs']} keep {r['keep']}: compute {100 * r['compute']:.0f} %, top kept {r['top_kept']}/{r['top']}")
    if not ok:
        raise SystemExit('no setting keeps the best trial and the top: run without pruning')
    best = min(ok, key=lambda r: r['compute'])
    print(json.dumps({'rungs': best['rungs'], 'keep': best['keep'], 'min_trials': 6}))


if __name__ == '__main__':
    main()
