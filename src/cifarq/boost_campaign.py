"""Bounded image-feature boosting follow-up; no allowance beyond the original combined 24-hour ceiling."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import paths
from .pipeline_score import objective
from .resnet18_campaign import session_cmd, stop_invalid_streak, write

if TYPE_CHECKING:
    import optuna


def remaining_allowance(previous: dict, own: dict | None = None, *, now: float | None = None) -> float:
    if previous.get('active'):
        raise RuntimeError('ResNet18 is still active; boosting must wait')
    if previous.get('spent_seconds', -1) < 0:
        raise ValueError('missing ResNet18 spending evidence')
    own = own or {'spent_seconds': 0}
    spent = own['spent_seconds']
    if own.get('active'):
        elapsed = (time.time() if now is None else now) - own['wall_started']
        spent = max(spent, own['spent_at_start'] + elapsed)
    return max(0, min(4 * 3600, 24 * 3600 - previous['spent_seconds']) - spent)


def sample(trial, learned: bool) -> dict:
    hp = {
        'backend': trial.suggest_categorical('backend', ['xgboost', 'lightgbm']),
        'features': trial.suggest_categorical('features', ['pooled_rgb', 'hog_color', 'learned'] if learned else ['pooled_rgb', 'hog_color']),
        'pca_components': trial.suggest_categorical('pca_components', [0, 32, 64, 128]),
        'rounds': trial.suggest_int('rounds', 50, 500, step=50),
        'depth': trial.suggest_int('depth', 3, 10),
        'lr': trial.suggest_float('lr', 0.02, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1),
        'colsample': trial.suggest_float('colsample', 0.5, 1),
        'l1': trial.suggest_float('l1', 1e-6, 10, log=True),
        'l2': trial.suggest_float('l2', 1e-3, 30, log=True),
        'max_bin': trial.suggest_categorical('max_bin', [63, 127, 255]),
    }
    if hp['backend'] == 'xgboost':
        hp['min_child_weight'] = trial.suggest_float('min_child_weight', 0.1, 30, log=True)
    else:
        hp['leaves'] = trial.suggest_categorical('leaves', [15, 31, 63, 127])
        hp['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 10, 100)
    return hp


@dataclass
class _BoostCampaign:
    out: Path
    budget_path: Path
    budget: dict
    previous: dict
    reference: dict
    encoder: Path
    data: str
    boost_python: str
    recovered_spent: float
    started: float
    deadline: float
    reserve: float

    def progress(self, stage: str, **extra: object) -> None:
        spent = time.monotonic() - self.started
        current = {
            **self.budget,
            'spent_seconds': self.recovered_spent + spent,
            'previous_campaign_seconds': self.previous['spent_seconds'],
            'stage': stage,
            **extra,
        }
        write(self.budget_path, current)
        report = {
            'stage': stage,
            'spent_seconds': current['spent_seconds'],
            'remaining_seconds': max(0, self.deadline - time.monotonic()),
            'updated_unix_s': time.time(),
        }
        write(self.out / 'progress.json', report)
        if os.environ.get('RUNQ_JOB_DIR'):
            write(Path(os.environ['RUNQ_JOB_DIR']) / 'progress.json', report)
        print(json.dumps(report), flush=True)

    def train_one(self, name: str, config: dict, cap: float, encoder_path: Path | None = None) -> dict:
        result_path = self.out / f'{name}.json'
        if result_path.exists():
            return json.loads(result_path.read_text())
        remaining = min(cap, self.deadline - time.monotonic())
        if remaining < 90:
            raise TimeoutError('combined campaign allowance exhausted')
        request = {
            'data': self.data,
            'output': str(self.out / name),
            'reference': self.reference,
            'config': config,
            'encoder': str(encoder_path or self.encoder),
            'cache': str(self.out / 'feature-cache'),
        }
        path = self.out / f'{name}.request.json'
        write(path, request)
        self.progress(name)
        try:
            subprocess.run(
                [self.boost_python, '-m', 'cifarq.image_boost', '--request', str(path), '--output', str(result_path)],
                env={
                    **os.environ,
                    'PYTHONPATH': str(paths.ROOT / 'src'),
                    'OMP_NUM_THREADS': '4',
                    'OPENBLAS_NUM_THREADS': '4',
                    'PYTHONUNBUFFERED': '1',
                },
                timeout=max(1, remaining - 15),
                check=True,
            )
        finally:
            self.progress(name)
        measured = json.loads(result_path.read_text())
        session_cmd(
            'note',
            '--id',
            'E2',
            '--outcome',
            'promising' if measured['score'] > 100 else 'rejected',
            '--result',
            f'{name}: score={measured["score"]:.3f}, accuracy={measured["accuracy"]:.5f}, '
            f'latency={measured["latency"]:.3f} ms, bytes={measured["size"]}; {result_path}',
            '--decision',
            'Screening measurement only; no automatic promotion. Full extractor cost included.',
            timeout=15,
        )
        return measured

    def run_baselines(self, failures: list[dict]) -> list[dict]:
        self.progress('boost-baselines')
        session_cmd('milestone', 'M5', '--state', 'active', timeout=15)
        baseline_results = []
        for features in ('pooled_rgb', 'hog_color', 'learned'):
            for backend in ('xgboost', 'lightgbm'):
                if self.deadline - time.monotonic() < self.reserve + 120:
                    break
                name = f'baseline-{features}-{backend}'
                try:
                    measured = self.train_one(
                        name, {'features': features, 'backend': backend}, min(600, self.deadline - time.monotonic() - self.reserve)
                    )
                    baseline_results.append({'name': name, **measured})
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, TimeoutError) as exc:
                    failures.append({'name': name, 'error': str(exc)})
                    write(self.out / 'failures.json', {'failures': failures})
        if not baseline_results:
            raise RuntimeError('No valid boosting baseline; inspect preserved logs rather than inventing scores.')
        session_cmd('milestone', 'M5', '--state', 'done', timeout=15)
        return baseline_results

    def _trial_objective(self, trial: optuna.Trial) -> float:
        config = {**sample(trial, True), 'seed': 40}
        trial.set_user_attr('config', config)
        name = f'trial-{trial.number:04d}'
        trial.set_user_attr('name', name)
        measured = self.train_one(name, config, min(900, self.deadline - time.monotonic() - self.reserve))
        for key in ('accuracy', 'latency', 'size', 'saved'):
            trial.set_user_attr(key, measured[key])
        return measured['score']

    def search(self, baseline_results: list[dict], trials: int) -> tuple[optuna.trial.FrozenTrial, int]:
        import optuna

        session_cmd('milestone', 'M6', '--state', 'active', timeout=15)
        study = optuna.create_study(
            study_name='image-boost-primary-score-v1',
            direction='maximize',
            storage=f'sqlite:///{self.out / "optuna.db"}',
            sampler=optuna.samplers.TPESampler(seed=20261117),
            load_if_exists=True,
        )
        if not study.trials:
            for measured in baseline_results:
                study.add_trial(
                    optuna.trial.create_trial(value=measured['score'], user_attrs={'config': measured['config'], 'name': measured['name']})
                )
        for interrupted in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
            study.tell(interrupted.number, state=optuna.trial.TrialState.FAIL)
        if len(study.trials) < trials and self.deadline - time.monotonic() > self.reserve + 90:
            study.optimize(
                self._trial_objective,
                n_trials=trials - len(study.trials),
                timeout=max(0, self.deadline - time.monotonic() - self.reserve),
                catch=(subprocess.CalledProcessError, subprocess.TimeoutExpired, TimeoutError),
                callbacks=[stop_invalid_streak],
            )
        completed = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
        return study.best_trial, completed

    def confirm(self, best: optuna.trial.FrozenTrial, previous_dir: Path, failures: list[dict]) -> tuple[list[dict], dict | None]:
        confirmations = []
        for seed in (71, 72, 73):
            if self.deadline - time.monotonic() < 120:
                break
            paired_encoder = previous_dir / f'confirmation-{seed}'
            if not (paired_encoder / 'weights.pt').exists():
                paired_encoder = self.encoder
            try:
                measured = self.train_one(
                    f'confirmation-{seed}',
                    {**best.user_attrs['config'], 'seed': seed},
                    min(600, self.deadline - time.monotonic()),
                    encoder_path=paired_encoder,
                )
                confirmations.append(measured)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, TimeoutError) as exc:
                failures.append({'name': f'confirmation-{seed}', 'error': str(exc)})
        if not confirmations:
            return confirmations, None
        average = {
            'accuracy': statistics.mean(r['accuracy'] for r in confirmations),
            'latency': statistics.median(t for r in confirmations for t in r['latency_ms_samples']),
            'size': statistics.mean(r['size'] for r in confirmations),
        }
        return confirmations, {**average, 'score': objective(average, self.reference), 'n_seeds': len(confirmations)}


def run(root: str, boost_python: str, trials: int = 36) -> dict:
    root_path = Path(root)
    previous_dir = root_path / 'results/autoresearch/resnet18-score'
    previous = json.loads((previous_dir / 'budget.json').read_text())
    resnet = json.loads((previous_dir / 'summary.json').read_text())
    reference = json.loads((previous_dir / 'reference.json').read_text())
    out = root_path / 'results/autoresearch/image-boost'
    out.mkdir(parents=True, exist_ok=True)
    budget_path = out / 'budget.json'
    budget = json.loads(budget_path.read_text()) if budget_path.exists() else {'spent_seconds': 0}
    allowance = remaining_allowance(previous, budget)
    if allowance < 120:
        result = {'status': 'budget-stopped', 'reason': 'No material allowance remains within the original 24 hours.', 'remaining_seconds': allowance}
        write(out / 'summary.json', result)
        return result
    recovered_spent = min(4 * 3600, 24 * 3600 - previous['spent_seconds']) - allowance
    started = time.monotonic()
    budget.update(active=True, spent_seconds=recovered_spent, spent_at_start=recovered_spent, wall_started=time.time())
    encoder = previous_dir / resnet['best_search']['name']
    if not (encoder / 'weights.pt').exists():
        raise ValueError('best ResNet18 encoder is not available')
    ctx = _BoostCampaign(
        out=out,
        budget_path=budget_path,
        budget=budget,
        previous=previous,
        reference=reference,
        encoder=encoder,
        data=str(root_path / 'data/dataset/dataset-56cff0f67835.npz'),
        boost_python=boost_python,
        recovered_spent=recovered_spent,
        started=started,
        deadline=started + allowance - 60,
        reserve=min(1800, allowance * 0.2),
    )
    try:
        failures: list[dict] = []
        baseline_results = ctx.run_baselines(failures)
        best, completed_trials = ctx.search(baseline_results, trials)
        write(out / 'best.json', {'name': best.user_attrs['name'], 'config': best.user_attrs['config'], 'score': best.value})
        confirmations, confirmed = ctx.confirm(best, previous_dir, failures)
        result = {
            'status': 'measured' if len(confirmations) == 3 else 'confirmation-needed',
            'reference': reference,
            'best': {'name': best.user_attrs['name'], 'config': best.user_attrs['config'], 'score': best.value},
            'confirmed': confirmed,
            'baselines': baseline_results,
            'failures': failures,
            'completed_trials': completed_trials,
            'total_previous_seconds': previous['spent_seconds'],
            'budget_limit': 'min(4 hours, unused original 24-hour allowance); no additional compute approval',
        }
        write(out / 'summary.json', result)
        session_cmd('milestone', 'M6', '--state', 'done' if len(confirmations) == 3 else 'blocked', '--note', result['status'], timeout=15)
        return result
    finally:
        ctx.progress('stopped', active=False)
