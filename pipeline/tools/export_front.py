"""The accuracy / compute front of the front study (pipeline/57_front.py), saved where it is committed: docs/front/
pareto.json (each front point: MACs a view, parameters, two-seed one-view validation, its seeds, the full
hyperparameters to train it again, where it came from), trials.csv (every completed trial, dominated ones too) and
front.svg (validation against MACs, the front drawn). Run it again after more trials.

    ~/.venvs/cifarq/bin/python pipeline/tools/export_front.py
"""
import csv
import json
import sys
import time
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from cifarq import families  # noqa: E402

OUT = ROOT / 'docs' / 'front'
STORAGE = f'sqlite:///{ROOT / "results" / "front-study.db"}'


def point(t) -> dict:
    """A front point: its numbers, where it came from, and the complete arguments of the training call
    (``train_kwargs``: the searched hyperparameters with the space's fixed ones, the teacher by its results/ path),
    so it can be trained again exactly (``families.train_two_seeds(**train_kwargs)``)."""
    a = t.user_attrs
    kwargs = families.front_space(optuna.trial.FixedTrial(t.params), kd=f'results/{families.TEACHER_DIR}')
    kwargs['widths'] = list(kwargs['widths'])
    new = 'source' not in a
    return {'trial': t.number, 'macs_per_view': int(t.values[1]), 'val_acc_1view': round(t.values[0], 5),
            'seeds': [0, 1] if new else a.get('seeds'), 'val_acc_1view_by_seed': a.get('val_acc_notta_seeds'),
            'params': a.get('params') or kwargs_params(kwargs), 'source': a.get('source', 'front study'),
            'device': a.get('device'), 'train_seconds': a.get('train_seconds'), 'hyperparameters': t.params,
            'train_kwargs': kwargs}


def kwargs_params(kwargs) -> int:
    from cifarq.airbench import DEFAULTS, build, param_counts
    return param_counts(build({**DEFAULTS, **kwargs})[0])['stored']


def svg(trials, front, path: Path):
    w, h, m = 720, 440, 60
    xs = [t.values[1] / 1e6 for t in trials]
    ys = [100 * t.values[0] for t in trials]
    x0, x1, y0, y1 = 0, max(xs) * 1.08, min(ys) - 0.2, max(ys) + 0.2
    X = lambda v: m + (v - x0) / (x1 - x0) * (w - 2 * m)
    Y = lambda v: h - m - (v - y0) / (y1 - y0) * (h - 2 * m)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" font-size="12">',
           f'<rect width="{w}" height="{h}" fill="white"/>',
           f'<line x1="{m}" y1="{h - m}" x2="{w - m}" y2="{h - m}" stroke="#444"/>',
           f'<line x1="{m}" y1="{m}" x2="{m}" y2="{h - m}" stroke="#444"/>',
           f'<text x="{w / 2}" y="{h - 18}" text-anchor="middle">MACs a view (millions)</text>',
           f'<text x="18" y="{h / 2}" text-anchor="middle" transform="rotate(-90 18 {h / 2})">validation accuracy, one view, '
           f'two seeds (%)</text>',
           f'<text x="{w / 2}" y="24" text-anchor="middle" font-size="14">Flip-equivariant airbench, 100 epochs: '
           f'accuracy / compute front</text>']
    for v in range(0, int(x1) + 1, 50):
        out.append(f'<text x="{X(v):.0f}" y="{h - m + 16}" text-anchor="middle" fill="#555">{v}</text>')
    yv = round(y0, 1)
    while yv <= y1:
        out.append(f'<text x="{m - 6}" y="{Y(yv) + 4:.0f}" text-anchor="end" fill="#555">{yv:.1f}</text>')
        out.append(f'<line x1="{m}" y1="{Y(yv):.0f}" x2="{w - m}" y2="{Y(yv):.0f}" stroke="#eee"/>')
        yv = round(yv + 0.2, 1)
    for t in trials:
        new = 'source' not in t.user_attrs
        out.append(f'<circle cx="{X(t.values[1] / 1e6):.1f}" cy="{Y(100 * t.values[0]):.1f}" r="4" '
                   f'fill="{"#9bb7d4" if new else "#d4b59b"}" opacity="0.8"/>')
    pts = sorted(front, key=lambda t: t.values[1])
    out.append('<polyline fill="none" stroke="#c0392b" stroke-width="2" points="'
               + ' '.join(f'{X(t.values[1] / 1e6):.1f},{Y(100 * t.values[0]):.1f}' for t in pts) + '"/>')
    for t in pts:
        out.append(f'<circle cx="{X(t.values[1] / 1e6):.1f}" cy="{Y(100 * t.values[0]):.1f}" r="5" fill="#c0392b"/>')
        out.append(f'<text x="{X(t.values[1] / 1e6) + 7:.1f}" y="{Y(100 * t.values[0]) - 7:.1f}" fill="#c0392b">'
                   f'{t.values[1] / 1e6:.0f}M {100 * t.values[0]:.2f}%</text>')
    out.append(f'<text x="{w - m}" y="{m - 8}" text-anchor="end" fill="#555">blue: new trials; tan: the 2026-10-30 '
               f'session\'s runs; red: the front</text></svg>')
    path.write_text('\n'.join(out) + '\n')


if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name='eqairbench-front', storage=STORAGE)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    front = sorted(study.best_trials, key=lambda t: t.values[1])
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {'study': 'eqairbench-front', 'exported': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'objectives': ['val_acc_1view (two seeds, up)', 'macs_per_view (down)'],
           'model': 'flip-equivariant airbench, depth 3, 100 epochs, distilled from airbench96 (45k split, seed 0), '
                    'parameter cap 2,000,000', 'teacher_recipe': families.AB96_45K,
           'trials': {'completed': len(done), 'pruned_or_refused': sum(t.state.name == 'PRUNED' for t in study.trials),
                      'failed': sum(t.state.name == 'FAIL' for t in study.trials)},
           'note': 'Validation on the 5000 held-out training images; the test set was not opened. Identical runs '
                   'differ by about 0.2 points (cuDNN), the floor for any difference between points.',
           'front': [point(t) for t in front]}
    (OUT / 'pareto.json').write_text(json.dumps(doc, indent=1) + '\n')
    with open(OUT / 'trials.csv', 'w', newline='') as f:
        keys = sorted({k for t in done for k in t.params})
        wr = csv.writer(f)
        wr.writerow(['trial', 'on_front', 'val_acc_1view', 'macs_per_view', 'params', 'source', 'device', 'train_seconds', *keys])
        ids = {t.number for t in front}
        for t in sorted(done, key=lambda t: t.values[1]):
            a = t.user_attrs
            wr.writerow([t.number, t.number in ids, round(t.values[0], 5), int(t.values[1]), a.get('params'),
                         a.get('source', 'front study'), a.get('device'), a.get('train_seconds'), *[t.params.get(k) for k in keys]])
    svg(done, front, OUT / 'front.svg')
    for p in doc['front']:
        print(f'{p["macs_per_view"] / 1e6:6.0f} M  {100 * p["val_acc_1view"]:.2f} %  trial {p["trial"]} ({p["source"]})')
    print('wrote', ', '.join(str(OUT / n) for n in ('pareto.json', 'trials.csv', 'front.svg')))
