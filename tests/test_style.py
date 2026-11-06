"""The code's shape, held by tests (the same checks as runq's): no module over 1,000 lines, no unused code (vulture,
with the names used from outside it listed in tests/vulture_whitelist.py with their reasons), and a count of
functions without type hints that may only fall. Formatting and lint are the pre-commit hook's (ruff); these run with
the suite. vulture and ruff are in the test extra; without them the two tests that need them skip."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ['src', 'pipeline', 'app', 'scripts']
LIMIT = 1000
HINTS_BASELINE = 330  # functions lacking type hints when the rule came in; lower it as they are hinted


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def test_modules_stay_under_the_limit():
    over = {str(f.relative_to(ROOT)): n for d in CODE for f in (ROOT / d).rglob('*.py') if (n := sum(1 for _ in f.open())) > LIMIT}
    assert over == {}, f'over {LIMIT} lines (split them): {over}'


@pytest.mark.skipif(not _have('vulture'), reason='vulture not installed (the test extra)')
def test_no_unused_code():
    dirs = [d for d in CODE if (ROOT / d).exists()]
    r = subprocess.run(
        [sys.executable, '-m', 'vulture', *dirs, 'tests/vulture_whitelist.py', '--min-confidence', '60'], cwd=ROOT, capture_output=True, text=True
    )
    assert r.returncode == 0, 'unused code (delete it, or whitelist it with its reason):\n' + r.stdout


@pytest.mark.skipif(not _have('ruff'), reason='ruff not installed (the test extra)')
def test_missing_type_hints_only_fall():
    dirs = [d for d in CODE if (ROOT / d).exists()]
    r = subprocess.run(
        [
            sys.executable,
            '-m',
            'ruff',
            'check',
            *dirs,
            '--select',
            'ANN',
            '--ignore',
            'ANN002,ANN003,ANN401',
            '--output-format',
            'json',
            '--no-cache',
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    found = json.loads(r.stdout or '[]')
    if len(found) <= HINTS_BASELINE:
        return
    changed = subprocess.run(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.split()
    near = [x for x in found if str(Path(x['filename']).relative_to(ROOT)) in changed]
    raise AssertionError(
        f'{len(found)} functions lack type hints, {len(found) - HINTS_BASELINE} over the {HINTS_BASELINE} allowed: '
        'hint the new ones. In the files changed here:\n'
        + '\n'.join(f'{Path(x["filename"]).relative_to(ROOT)}:{x["location"]["row"]} {x["message"]}' for x in near[:40])
    )
