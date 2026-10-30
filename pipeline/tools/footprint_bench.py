"""The footprint and runtime table of saved models (``cifarq.footprint.bench``), each against a reference: parameters,
bytes, MACs, peak GPU and CPU memory, latency and throughput, and its footprint class (S, P or out). Local machine only.

    python pipeline/tools/footprint_bench.py --ref results/models/B0 results/models/candidate ... [--cap 2000000] [--no-gpu]
    python pipeline/tools/footprint_bench.py --repeat 5 results/models/B0        # the evaluator's repeatability check
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cifarq import footprint  # noqa: E402
from cifarq.paths import RESULTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('models', nargs='+')
    ap.add_argument('--ref', help='the reference model directory (its class bounds)')
    ap.add_argument('--cap', type=int, help='a parameter cap replacing the reference parameters (Goal A: 2000000)')
    ap.add_argument('--no-gpu', action='store_true')
    ap.add_argument('--repeat', type=int, default=1, help='measure each model this many times (coefficients of variation)')
    ap.add_argument('--out', default=str(RESULTS / 'footprint'))
    a = ap.parse_args()
    ref = footprint.bench(a.ref, gpu=not a.no_gpu) if a.ref else None
    rows = []
    for m in a.models:
        reps = [footprint.bench(m, gpu=not a.no_gpu) for _ in range(a.repeat)]
        row = reps[0]
        if a.repeat > 1:
            row['cv'] = {name: {k: round(statistics.pstdev(v) / statistics.mean(v), 4) for k in ('p50_ms', 'rss_mb', 'peak_rss_mb', 'peak_gpu_mb')
                                if k in reps[0]['runs'][name] and statistics.mean(v := [r['runs'][name][k] for r in reps])}
                         for name in reps[0]['runs']}
            row['repeats'] = [r['runs'] for r in reps]
        if ref:
            row['class'] = footprint.classify(row, ref, a.cap)
        rows.append(row)
        print(json.dumps({k: row[k] for k in row if k not in ('repeats',)}, indent=1), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f'{time.strftime("%Y%m%d-%H%M%S")}.json'
    f.write_text(json.dumps({'ref': ref, 'rows': rows, 'cap': a.cap}, indent=1) + '\n')
    print(f)


if __name__ == '__main__':
    main()
