"""The primary repo metric: relative validation error penalized by inference latency and deployed bytes.

Run ``python -m cifarq.efficiency records.json`` to rank each independent cohort in a recorded assessment.
The API and CLI keep the reference at 1; multiply by 100 for the reporting scale. Historical cohorts may use
explicitly labelled size proxies; primary comparisons require deployed bytes and matched complete-pipeline
measurements. Never compare different benchmark protocols. See ``docs/metrics/README.md``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Measurement:
    """Accuracy is a fraction; latency and size use the same positive units as the reference."""

    accuracy: float
    latency: float | None = None
    size: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.accuracy) or not 0 <= self.accuracy < 1:
            raise ValueError('accuracy must be finite and in [0, 1); perfect accuracy has undefined inverse error')
        for name, value in (('latency', self.latency), ('size', self.size)):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be finite and positive when supplied')


def score(candidate: Measurement, reference: Measurement, *, alpha: float = 0.25, beta: float = 0.10) -> float:
    """Higher is better: (reference error / error) * latency ratio**alpha * size ratio**beta.

    Missing cost measurements are allowed only when their exponent is zero. Reference scores exactly one.
    Perfect observed accuracy is rejected, not smoothed with an arbitrary error floor.
    """
    value = math.log1p(-reference.accuracy) - math.log1p(-candidate.accuracy)
    for name, exponent, cost, base in (
        ('latency', alpha, candidate.latency, reference.latency),
        ('size', beta, candidate.size, reference.size),
    ):
        if not math.isfinite(exponent) or exponent < 0:
            raise ValueError(f'{name} exponent must be finite and nonnegative')
        if exponent == 0:
            continue
        if cost is None or base is None:
            raise ValueError(f'{name} is required for candidate and reference when its exponent is positive')
        value += exponent * (math.log(base) - math.log(cost))
    result = math.exp(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError('score is outside the positive finite floating-point range')
    return result


def assess(document: dict[str, Any], *, alpha: float = 0.25, beta: float = 0.10) -> dict[str, Any]:
    """Rank explicitly compatible cohorts; retain input evidence and never pool their scores."""
    cohorts = document['cohorts']
    if not cohorts:
        raise ValueError('assessment needs at least one cohort')
    ranked = []
    for cohort in cohorts:
        protocol = cohort['protocol']
        for key in ('dataset', 'split', 'hardware', 'precision', 'batch', 'timing_scope', 'size_metric'):
            if not protocol.get(key):
                raise ValueError(f'cohort {cohort["id"]}: missing protocol field {key}')
        if protocol['split'] != 'validation':
            raise ValueError('selection scores require validation accuracy, not test accuracy')
        rows = cohort['rows']
        by_id = {}
        for row in rows:
            if row['id'] in by_id:
                raise ValueError(f'duplicate measurement id {row["id"]!r}')
            if not row['accuracy_policy'] or row['accuracy_policy'] != row['latency_policy']:
                raise ValueError(f'{row["id"]}: accuracy and latency policies must match')
            by_id[row['id']] = Measurement(row['accuracy'], row.get('latency'), row.get('size'))
        if cohort['reference'] not in by_id:
            raise ValueError(f'unknown reference {cohort["reference"]!r}')
        reference = by_id[cohort['reference']]
        scored = [{**row, 'score': score(by_id[row['id']], reference, alpha=alpha, beta=beta)} for row in rows]
        ranked.append({**cohort, 'rows': sorted(scored, key=lambda row: row['score'], reverse=True)})
    return {**document, 'metric': 'relative-error-cost-v1', 'alpha': alpha, 'beta': beta, 'cohorts': ranked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records', type=Path)
    parser.add_argument('--alpha', type=float, default=0.25, help='latency penalty (default: 0.25)')
    parser.add_argument('--beta', type=float, default=0.10, help='size penalty (default: 0.10)')
    args = parser.parse_args()
    result = assess(json.loads(args.records.read_text(encoding='utf-8')), alpha=args.alpha, beta=args.beta)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
