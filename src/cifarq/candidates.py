"""Candidates: the best results the stages found, kept in the repository with the code (``candidates.jsonl``).

``results/`` stays out of git (it holds studies, models and logs); this file is the part of it that must not be lost:
one JSON line per result a stage reports (a search's best trial, a front's most accurate point, a trained run, a
champion over seeds), with its hyperparameters, the dataset it was scored on (the manifest's hash) and the code that
scored it (the git tree of ``src/cifarq`` and whether it had changes). ``note`` adds the line a stage record implies
(``pipeline/common.record`` calls it); a champion also marks the candidate it retrained as verified, and publishing
marks the champion published. An entry scored on another dataset or by other training code than the current is stale:
still the best known, worth running again. ``python -m cifarq.candidates`` lists the best per stage and metric.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from cifarq import paths

FILE = paths.ROOT / 'candidates.jsonl'
SEARCH_STAGES = ('tune', 'on-vm', 'distributed', 'single')  # (the search stages' record names, old layouts included)


def code_identity() -> dict[str, Any]:
    """The training code: the git tree hash of ``src/cifarq`` at HEAD and whether it has uncommitted changes."""

    def git(*args: str) -> str:
        r = subprocess.run(['git', '-C', str(paths.ROOT), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ''

    return {
        'tree': git('rev-parse', 'HEAD:src/cifarq'),
        'commit': git('rev-parse', '--short', 'HEAD'),
        'dirty': bool(git('status', '--porcelain', '--', 'src/cifarq')),
    }


def entries(path: Path = FILE) -> list[dict[str, Any]]:
    """Every candidate, verifications applied (a 'verify' line updates the entry it names)."""
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding='utf-8').splitlines() if path.is_file() else ():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get('kind') == 'verify':
            if e['id'] in out:
                out[e['id']].setdefault('verified', {})[e['check']] = {k: e[k] for k in ('passed', 'detail', 'when') if k in e}
        else:
            out[e['id']] = e
    return list(out.values())


def _append(line: dict[str, Any], path: Path = FILE) -> None:
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(line, default=str) + '\n')


def best(stage_group: str, metric: str, dataset: str | None = None, path: Path = FILE) -> dict[str, Any] | None:
    """The highest ``metric`` among a group's entries ('search' covers every search stage), on ``dataset`` if given."""
    pool = [
        e
        for e in entries(path)
        if e.get('metric') == metric
        and (e['stage'] in SEARCH_STAGES if stage_group == 'search' else e['stage'] == stage_group)
        and (dataset is None or e.get('dataset') == dataset)
    ]
    return max(pool, key=lambda e: e['value'], default=None)


def stale(entry: dict[str, Any], dataset: str | None, code: dict[str, Any]) -> bool:
    """Scored on another dataset or by other (or uncommitted) training code than ``dataset`` and ``code``."""
    return entry.get('dataset') != dataset or entry.get('code', {}).get('tree') != code.get('tree') or bool(entry.get('code', {}).get('dirty'))


def record(entry: dict[str, Any], path: Path = FILE) -> bool:
    """Append ``entry`` (its id, stage, metric, value and params; when, dataset and code filled if missing); True when it is
    the best of its stage group and metric on its dataset."""
    if any(e['id'] == entry['id'] for e in entries(path)):
        raise ValueError(f'{path}: candidate {entry["id"]!r} is already recorded')
    group = 'search' if entry['stage'] in SEARCH_STAGES else entry['stage']
    prior = best(group, entry['metric'], entry.get('dataset'), path)
    full = {'when': time.strftime('%Y-%m-%d %H:%M:%S'), 'code': code_identity(), **entry}
    _append(full, path)
    return prior is None or full['value'] > prior['value']


def verify(entry_id: str, check: str, passed: bool, detail: str = '', path: Path = FILE) -> None:
    """Record a verification (``check``: 'champion', 'published') on candidate ``entry_id``."""
    if not any(e['id'] == entry_id for e in entries(path)):
        raise KeyError(f'{path}: no candidate {entry_id!r}')
    _append(
        {'kind': 'verify', 'id': entry_id, 'check': check, 'passed': bool(passed), 'detail': detail, 'when': time.strftime('%Y-%m-%d %H:%M:%S')}, path
    )


def from_record(stage: str, rec: dict[str, Any], source: str) -> dict[str, Any] | None:
    """The candidate a stage record implies (None when it reports no result): ``source`` its results path."""
    prov = rec.get('provenance', {})
    base = {'id': f'{stage}-{Path(source).stem}', 'stage': stage, 'source': source, 'dataset': prov.get('dataset')}
    if prov.get('commit'):
        base['commit'] = prov['commit'] + ('+' if prov.get('dirty') else '')
    if stage in SEARCH_STAGES and 'best_value' in rec:  # the 2026-10-27/29 layout (no provenance; the PyTorch code)
        return {
            **base,
            'metric': 'val_acc',
            'value': rec['best_value'],
            'test': rec.get('best_test_acc'),
            'params': rec.get('best_params', {}),
            'device': rec.get('device'),
        }
    if stage in SEARCH_STAGES and rec.get('best'):
        b = rec['best']
        return {**base, 'metric': 'val_acc', 'value': b['value'], 'params': b.get('params', {}), 'study': rec.get('study'), 'trial': b.get('trial')}
    if stage == 'front' and rec.get('front'):
        top = max(rec['front'], key=lambda p: p.get('val_1view', 0.0))
        return {
            **base,
            'metric': 'val_1view',
            'value': top['val_1view'],
            'params': top.get('params', {}),
            'study': rec.get('study'),
            'trial': top.get('trial'),
            'macs': top.get('macs'),
        }
    if stage == 'train' and isinstance(rec.get('result'), dict) and 'val_acc' in rec['result']:
        r = rec['result']
        return {
            **base,
            'metric': 'val_acc',
            'value': r['val_acc'],
            'test': r.get('test_acc'),
            'params': rec.get('params', {}),
            'device': r.get('device'),
        }
    if stage == 'champion' and rec.get('runs'):
        runs = rec['runs']
        chosen = rec.get('chosen') or {}
        val = chosen.get('val_acc') if isinstance(chosen, dict) else None
        return {
            **base,
            'metric': 'val_acc',
            'value': val if val is not None else max(r.get('val_acc', 0.0) for r in runs),
            'test': rec.get('test_mean'),
            'test_stdev': rec.get('test_stdev'),
            'seeds': len(runs),
            'params': rec.get('params', {}),
            'from': rec.get('candidate') or rec.get('source') or rec.get('search_trial'),
        }
    return None


def note(stage: str, rec: dict[str, Any], source: Path) -> dict[str, Any] | None:
    """Record the candidate ``rec`` implies (``from_record``); a champion also verifies the candidate it retrained: the
    one its record names (``60_champion.py --candidate``), else any with the same hyperparameters. The entry, or None."""
    try:
        rel = str(Path(source).resolve().relative_to(paths.ROOT))
    except ValueError:
        rel = str(source)
    entry = from_record(stage, rec, rel)
    if entry is None:
        return None
    record(entry)
    if stage == 'champion':
        named = rec.get('candidate')
        same = [
            e
            for e in entries()
            if e['id'] != entry['id'] and e['stage'] != 'champion' and (e['id'] == named if named else e.get('params') == entry['params'])
        ]
        for e in same:
            verify(
                e['id'],
                'champion',
                True,
                f'retrained over {entry.get("seeds")} seeds: test {entry.get("test")} +- {entry.get("test_stdev")} ({entry["id"]})',
            )
    return entry


def params_of(entry_id: str) -> dict[str, Any]:
    """Candidate ``entry_id``'s hyperparameters (``60_champion.py --candidate``)."""
    for e in entries():
        if e['id'] == entry_id:
            return dict(e['params'])
    raise KeyError(f'{FILE}: no candidate {entry_id!r}')


def start_point(spec: str, path: Path = FILE) -> dict[str, Any]:
    """The candidate a search starts from: ``best`` (the best search result by validation accuracy) or a candidate id;
    its entry, hyperparameters included."""
    entry = best('search', 'val_acc', path=path) if spec == 'best' else next((e for e in entries(path) if e['id'] == spec), None)
    if entry is None:
        raise KeyError(f'{path}: no candidate {spec!r}')
    return entry


def main() -> None:
    """Print the best candidate per stage group and metric, and whether it is stale."""
    ds = (paths.read_manifest('dataset') or {}).get('sha256')
    code = code_identity()
    rows = sorted({('search' if e['stage'] in SEARCH_STAGES else e['stage'], e['metric']) for e in entries()})
    for group, metric in rows:
        b = best(group, metric)
        assert b is not None
        flags = ', '.join(k for k, v in b.get('verified', {}).items() if v.get('passed')) or '-'
        state = 'STALE' if stale(b, ds, code) else 'current'
        print(f'{group:9} {metric:10} {b["value"]:.4f}  {b["id"]:28} test {b.get("test", "-")}  verified {flags}  {state}')


if __name__ == '__main__':
    main()
