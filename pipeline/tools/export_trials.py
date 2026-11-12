"""A study's completed and pruned trials as JSON lines (number, state, value, curve, params, distributions, scalar user
attributes), the form git keeps: ``results/`` is not committed, so a campaign folder holds its studies' exports, and
``--seed-from`` in ``55_tune_on_vm.py`` and ``57_front.py`` reads them as it reads a database.

    python pipeline/tools/export_trials.py results/eqaug2-study.db docs/autoresearch/2026-11-11-idle-search/eqaug2-trials.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import completed_trials  # noqa: E402


def rows(trials: list) -> list[dict]:
    import optuna

    return [
        {
            'number': t.number,
            'state': t.state.name,
            'value': t.value if t.state.name == 'COMPLETE' else None,
            'intermediate': {str(k): v for k, v in t.intermediate_values.items()},
            'params': t.params,
            'distributions': {k: optuna.distributions.distribution_to_json(d) for k, d in t.distributions.items()},
            'user_attrs': {k: v for k, v in t.user_attrs.items() if isinstance(v, (int, float, str, bool))},
        }
        for t in trials
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('study', type=Path)
    ap.add_argument('out', type=Path)
    a = ap.parse_args()
    out = rows(completed_trials(a.study, pruned=True))
    a.out.write_text(''.join(json.dumps(r) + '\n' for r in out))
    print(f'{len(out)} completed and pruned trials -> {a.out}')


if __name__ == '__main__':
    main()
