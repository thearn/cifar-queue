"""Error-cost tradeoffs and the evidence boundary of retrospective scoring."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cifarq.efficiency import Measurement, assess, score


def test_reference_and_meaningful_tradeoffs():
    ref = Measurement(0.95, 10, 100)
    assert score(ref, ref) == 1
    assert score(Measurement(0.96, 10, 100), ref) == pytest.approx(1.25)
    assert score(Measurement(0.95, 5, 100), ref) == pytest.approx(2**0.25)
    assert score(Measurement(0.95, 10, 50), ref) == pytest.approx(2**0.1)
    assert score(Measurement(0.96, 20, 100), ref) == pytest.approx(1.25 / 2**0.25)
    assert score(Measurement(0.94, 20, 200), ref) < 1
    assert score(Measurement(0.96, 0.01, 0.1), Measurement(0.95, 0.01, 0.1)) == pytest.approx(1.25)


@pytest.mark.parametrize(
    'kwargs',
    [
        {'accuracy': -0.01},
        {'accuracy': 1},
        {'accuracy': 95},
        {'accuracy': float('nan')},
        {'accuracy': float('inf')},
        {'accuracy': 0.95, 'latency': 0},
        {'accuracy': 0.95, 'size': -1},
        {'accuracy': 0.95, 'size': float('inf')},
    ],
)
def test_invalid_measurements_fail(kwargs):
    with pytest.raises(ValueError):
        Measurement(**kwargs)


def test_missing_costs_are_not_silently_imputed():
    ref = Measurement(0.95, 10, 100)
    with pytest.raises(ValueError, match='size is required'):
        score(Measurement(0.96, 10), ref)
    with pytest.raises(ValueError, match='latency is required'):
        score(Measurement(0.96), ref, beta=0)
    assert score(Measurement(0.96), ref, alpha=0, beta=0) == pytest.approx(1.25)
    assert score(Measurement(0.96, 10), ref, beta=0) == pytest.approx(1.25)
    assert score(Measurement(0), Measurement(0), alpha=0, beta=0) == 1


@pytest.mark.parametrize('weight', [-1, float('nan'), float('inf')])
def test_invalid_weights_fail(weight):
    ref = Measurement(0.95, 10, 100)
    with pytest.raises(ValueError):
        score(ref, ref, alpha=weight)
    with pytest.raises(ValueError):
        score(ref, ref, beta=weight)


def recorded():
    path = Path(__file__).resolve().parents[1] / 'docs' / 'metrics' / 'recorded.json'
    return json.loads(path.read_text())


def test_assessment_keeps_cohorts_and_provenance_separate():
    document = recorded()
    original = copy.deepcopy(document)
    result = assess(document)
    assert document == original
    assert len(result['cohorts']) == len(document['cohorts'])
    for source, cohort in zip(document['cohorts'], result['cohorts'], strict=True):
        assert cohort['protocol'] == source['protocol']
        assert [r['score'] for r in cohort['rows']] == sorted((r['score'] for r in cohort['rows']), reverse=True)
        assert next(r['score'] for r in cohort['rows'] if r['id'] == cohort['reference']) == 1
        assert cohort['sources'] == source['sources']


@pytest.mark.parametrize('defect', ['policy', 'split', 'protocol', 'duplicate', 'reference'])
def test_assessment_rejects_incompatible_evidence(defect):
    document = recorded()
    cohort = document['cohorts'][0]
    if defect == 'policy':
        cohort['rows'][0]['latency_policy'] = 'six-view'
    elif defect == 'split':
        cohort['protocol']['split'] = 'test'
    elif defect == 'protocol':
        del cohort['protocol']['hardware']
    elif defect == 'duplicate':
        cohort['rows'].append(cohort['rows'][0])
    else:
        cohort['reference'] = 'missing'
    with pytest.raises(ValueError):
        assess(document)


def test_cli_scores_saved_records_without_training():
    path = Path(__file__).resolve().parents[1] / 'docs' / 'metrics' / 'recorded.json'
    run = subprocess.run(
        [sys.executable, '-m', 'cifarq.efficiency', str(path), '--beta', '0'],
        check=True,
        text=True,
        capture_output=True,
    )
    result = json.loads(run.stdout)
    assert result['beta'] == 0
    assert result['metric'] == 'relative-error-cost-v1'
    assert all(c['rows'] for c in result['cohorts'])
