"""Where early stopping is safe for this model: from complete learning curves (every trial run to its end: the mode 3
job directories' result.json histories, or a no-pruning mode 2 study's), the rank correlation of validation accuracy
part-way through with its final value, how many of the final top 10 % a cut there would lose, and successive halving
replayed for a few rung settings (runq.hpo.rung_check, simulate_rungs). tune.PRUNE was chosen from this.

    python pipeline/tools/rung_check.py [JOB_DIRS_ROOT ...]
"""

import json
import math
import sys
from pathlib import Path

from runq.queue.collect import job_dir
from runq.search.hpo import rung_check, simulate_rungs

roots = [Path(a) for a in sys.argv[1:]] or [
    r for d in ('distributed', 'tune') for r in (Path(__file__).resolve().parents[2] / 'results' / 'jobs' / d, job_dir(d))
]
curves = []
for root in roots:
    for f in root.glob('*/result.json'):
        try:
            r = json.loads(f.read_text())
        except ValueError:
            continue
        h = r.get('history') or []
        if h and not r.get('pruned') and math.isfinite(r['val_acc']) and len(h) == r['hyperparameters']['epochs']:
            curves.append([x['val_acc'] for x in h])
print(f'{len(curves)} complete curves')
for smooth in (False, True):
    print('best so far' if smooth else 'latest epoch')
    for row in rung_check(curves, smooth=smooth):
        print(f'  {row["percent"]:3d} %: rho {row["rho"]:.2f}; keeping the top third cuts {row["top_cut"]} of the {row["top"]} final top-10 %')
for rungs, keep in [((10, 30, 90), 1 / 3), ((10, 20, 40, 80), 1 / 2), ((20, 50), 1 / 2), ((30, 60), 1 / 2), ((10, 30, 60), 2 / 3)]:
    s = simulate_rungs(curves, rungs, keep)
    print(
        f'rungs {list(rungs)} keep {keep:.2f}: compute {100 * s["compute"]:.0f} %, top-10 % kept {s["top_kept"]}/{s["top"]}, '
        f'best kept {s["best_kept"]}'
    )
