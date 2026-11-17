"""NetScore (A. Wong, arXiv:1806.05512), a secondary accuracy / complexity comparison, not the repo objective.

The primary metric is ``cifarq.efficiency``'s relative error-cost score.

``Omega = 20 log10(a**alpha / (p**beta * m**gamma))`` with ``alpha=2``, ``beta=gamma=0.5``, where the paper's units are
percent accuracy, millions of parameters and billions of multiply-accumulates. This module takes the repository's own
units instead (a fraction, a parameter count, a MAC count) and converts, so that call sites cannot mix the two.

NetScore needs no reference model, so it ranks every network measured under one protocol on one scale, where the
error-cost score ranks a cohort against its own reference. It prices parameters and MACs rather than measured
latency and bytes. Use the paper's default exponents for published comparisons; the API also permits alternatives.
Do not invent neural-network parameter or MAC equivalents for pipelines where they are not meaningful.

Run ``python -m cifarq.netscore records.json`` to rank recorded measurements that carry parameters and MACs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ALPHA, BETA, GAMMA = 2.0, 0.5, 0.5
MILLION, BILLION = 1e6, 1e9


def _positive(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def netscore(accuracy: float, params: float, macs: float, *, alpha: float = ALPHA, beta: float = BETA, gamma: float = GAMMA) -> float:
    """The paper's ``Omega`` in decibels from an accuracy fraction in (0, 1], a parameter count and MACs per image.

    Higher is better. Zero accuracy, parameters or MACs have no logarithm and are rejected rather than floored.
    Perfect accuracy is allowed here, unlike the error-cost score, which inverts the error.
    """
    if not math.isfinite(accuracy) or not 0 < accuracy <= 1:
        raise ValueError('accuracy must be finite and in (0, 1]; NetScore takes a fraction, not a percentage')
    for name, exponent in (('alpha', alpha), ('beta', beta), ('gamma', gamma)):
        if not math.isfinite(exponent) or exponent < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    value = (
        alpha * math.log10(100 * accuracy)
        - beta * math.log10(_positive('params', params) / MILLION)
        - gamma * math.log10(_positive('macs', macs) / BILLION)
    )
    return 20 * value


def information_density(accuracy: float, params: float) -> float:
    """Canziani's accuracy per million parameters, the metric NetScore extends; the paper reports it alongside."""
    if not math.isfinite(accuracy) or not 0 < accuracy <= 1:
        raise ValueError('accuracy must be finite and in (0, 1]')
    return 100 * accuracy / (_positive('params', params) / MILLION)


def assess(document: dict[str, Any], *, alpha: float = ALPHA, beta: float = BETA, gamma: float = GAMMA) -> dict[str, Any]:
    """Score every row of every cohort, each row needing ``accuracy``, ``params`` and ``macs``.

    No reference is used, so rows are comparable across cohorts that share a protocol's dataset and split; the ranking
    stays per cohort because nothing else here establishes that they do.
    """
    cohorts = document['cohorts']
    if not cohorts:
        raise ValueError('assessment needs at least one cohort')
    ranked = []
    for cohort in cohorts:
        protocol = cohort['protocol']
        for key in ('dataset', 'split', 'size_metric'):
            if not protocol.get(key):
                raise ValueError(f'cohort {cohort["id"]}: missing protocol field {key}')
        if protocol['split'] != 'validation':
            raise ValueError('selection scores require validation accuracy, not test accuracy')
        seen = set()
        scored = []
        for row in cohort['rows']:
            if row['id'] in seen:
                raise ValueError(f'duplicate measurement id {row["id"]!r}')
            seen.add(row['id'])
            for key in ('params', 'macs'):
                if row.get(key) is None:
                    raise ValueError(f'{row["id"]}: NetScore needs {key}; it prices complexity, not latency')
            scored.append(
                {
                    **row,
                    'netscore': netscore(row['accuracy'], row['params'], row['macs'], alpha=alpha, beta=beta, gamma=gamma),
                    'information_density': information_density(row['accuracy'], row['params']),
                }
            )
        ranked.append({**cohort, 'rows': sorted(scored, key=lambda row: row['netscore'], reverse=True)})
    return {**document, 'metric': 'netscore-wong-2018', 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'cohorts': ranked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records', type=Path)
    parser.add_argument('--alpha', type=float, default=ALPHA, help='accuracy exponent (default: 2.0)')
    parser.add_argument('--beta', type=float, default=BETA, help='parameter-count exponent (default: 0.5)')
    parser.add_argument('--gamma', type=float, default=GAMMA, help='MAC-count exponent (default: 0.5)')
    args = parser.parse_args()
    result = assess(json.loads(args.records.read_text(encoding='utf-8')), alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
