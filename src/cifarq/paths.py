"""The repository's directories and its TOML configs. ``ML_PROJECT_ROOT`` moves them all (the tests run the pipeline
in a temporary copy); on a VM a job is given its files by path, so nothing there depends on these."""
from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path

ROOT = Path(os.environ.get('ML_PROJECT_ROOT') or Path(__file__).resolve().parents[2])
CONFIGS, DATA, RESULTS = ROOT / 'configs', ROOT / 'data', ROOT / 'results'
MANIFESTS = DATA / 'manifests'


def config(name: str) -> dict:
    """``configs/<name>.toml``."""
    return tomllib.loads((CONFIGS / f'{name}.toml').read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj) -> str:
    """A short hash of a JSON-able value (a stage's inputs and settings)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def read_manifest(name: str) -> dict | None:
    f = MANIFESTS / f'{name}.json'
    return json.loads(f.read_text()) if f.exists() else None


def write_manifest(name: str, manifest: dict) -> Path:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    f = MANIFESTS / f'{name}.json'
    f.write_text(json.dumps(manifest, indent=1, default=str) + '\n')
    return f


def up_to_date(name: str, inputs: str, directory: Path) -> bool:
    """Whether stage ``name`` already ran on ``inputs`` (a ``digest`` of what it read and its settings) and its files
    are still in ``directory`` with the hashes it recorded."""
    m = read_manifest(name)
    return bool(m and m.get('inputs') == inputs and all(
        (directory / f).exists() and sha256(directory / f) == h for f, h in m['files'].items()))
