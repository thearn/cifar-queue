"""NetScore against published values, its units, and where it disagrees with the relative error-cost score."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from cifarq.efficiency import Measurement, score
from cifarq.netscore import assess, information_density, netscore


def test_matches_published_values():
    # The paper's own units, recomputed from its reported figures: ResNet-50 (76.13 %, 25.6 M, 3.9 GMACs) against
    # MobileNetv1 (70.6 %, 4.2 M, 0.569 GMACs). The efficient network scores far above the more accurate one.
    assert netscore(0.7613, 25.6e6, 3.9e9) == pytest.approx(55.27, abs=0.01)
    assert netscore(0.706, 4.2e6, 0.569e9) == pytest.approx(70.17, abs=0.01)
    assert information_density(0.7613, 25.6e6) == pytest.approx(2.974, abs=0.001)


def test_units_and_exponents():
    base = netscore(0.95, 1e6, 1e9)
    assert base == pytest.approx(20 * 2 * math.log10(95))  # a is a percentage, p is millions, m is billions
    assert netscore(0.95, 4e6, 1e9) == pytest.approx(base - 20 * 0.5 * 2 * 0.30103)  # four times the parameters
    assert netscore(0.95, 1e6, 4e9) == pytest.approx(base - 20 * 0.5 * 2 * 0.30103)  # four times the MACs
    assert netscore(0.95, 1e6, 1e9, beta=0, gamma=0) == pytest.approx(20 * 2 * math.log10(95))
    assert netscore(1.0, 1e6, 1e9) > base  # perfect accuracy has a NetScore, unlike an inverted error


@pytest.mark.parametrize(
    'args',
    [
        (0.0, 1e6, 1e9),
        (1.1, 1e6, 1e9),
        (float('nan'), 1e6, 1e9),
        (95, 1e6, 1e9),
        (0.95, 0, 1e9),
        (0.95, 1e6, 0),
        (0.95, -1, 1e9),
        (0.95, 1e6, float('inf')),
    ],
)
def test_invalid_inputs_fail(args):
    with pytest.raises(ValueError):
        netscore(*args)


def test_disagrees_with_error_cost_when_size_is_cheap():
    """The two metrics rank the same pair of measured networks differently, which is why both are kept.

    The numbers are the measured ``d3`` and ``baseline`` arms at batch 256. Ignoring size, the error-cost score buys
    d3's extra accuracy with its extra latency and calls it the better network; NetScore, whose size and compute
    exponents are fixed at 0.5, calls the smaller one better by three decibels.
    """
    baseline = Measurement(0.9448, 2.701, 1_113_336)
    d3 = Measurement(0.9493, 3.554, 1_798_488)
    assert score(d3, baseline, beta=0) > 1
    assert netscore(0.9493, 1_798_488, 92_978_976) < netscore(0.9448, 1_113_336, 69_070_368)


def measured():
    path = Path(__file__).resolve().parents[1] / 'docs' / 'metrics' / 'measured.json'
    return json.loads(path.read_text())


def test_assessment_ranks_and_keeps_provenance():
    document = measured()
    result = assess(document)
    assert result['metric'] == 'netscore-wong-2018'
    for source, cohort in zip(document['cohorts'], result['cohorts'], strict=True):
        assert cohort['protocol'] == source['protocol']
        assert [r['netscore'] for r in cohort['rows']] == sorted((r['netscore'] for r in cohort['rows']), reverse=True)


@pytest.mark.parametrize('defect', ['missing_macs', 'split', 'duplicate'])
def test_assessment_rejects_incompatible_evidence(defect):
    document = measured()
    cohort = document['cohorts'][0]
    if defect == 'missing_macs':
        cohort['rows'][0].pop('macs')
    elif defect == 'split':
        cohort['protocol']['split'] = 'test'
    else:
        cohort['rows'].append(dict(cohort['rows'][0]))
    with pytest.raises(ValueError):
        assess(document)


def test_cli_ranks_saved_records():
    path = Path(__file__).resolve().parents[1] / 'docs' / 'metrics' / 'measured.json'
    run = subprocess.run([sys.executable, '-m', 'cifarq.netscore', str(path)], check=True, text=True, capture_output=True)
    result = json.loads(run.stdout)
    assert result['cohorts'] and all(c['rows'] for c in result['cohorts'])
    assert all('netscore' in r and 'information_density' in r for c in result['cohorts'] for r in c['rows'])
