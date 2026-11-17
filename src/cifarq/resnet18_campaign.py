"""Bounded local campaign driver. GPU work runs in separate interpreters; the queue imports only this driver."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from . import paths

if TYPE_CHECKING:
    import optuna


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def session_cmd(*arguments: str, timeout: int = 30) -> None:
    if os.environ.get('RUNQ_JOB_DIR'):
        try:
            subprocess.run(
                [sys.executable, '-m', 'runq', 'session', *arguments, '--session', 'cifar-resnet18'],
                check=False,
                stdout=subprocess.DEVNULL,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f'bookkeeping warning: {exc}', flush=True)


def stop_invalid_streak(study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
    import optuna

    if len(study.trials) >= 5 and all(t.state == optuna.trial.TrialState.FAIL for t in study.trials[-5:]):
        study.stop()


def d3_reference(data: str, root: str, output: str) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from .airbench import Predictor
    from .pipeline_score import DATASET_SHA256, evaluate, load_development

    if not any('T4' in device.device_kind for device in jax.devices()):
        raise RuntimeError('reference must run on the T4')
    arrays = load_development(data)
    saved = json.loads((Path(root) / 'docs/metrics/measured.json').read_text())
    cohort = next(c for c in saved['cohorts'] if c['id'] == 'measured-checkpoint-one-view-b256')
    rows = [r for r in cohort['rows'] if r['campaign'] == 'readout-recipe-spectral' and r['arm'] == 'd3']
    if len(rows) != 16 or len({r['seed'] for r in rows}) != 16:
        raise ValueError('frozen d3 reference needs the 16 distinct declared seeds')
    runs = []
    for row in rows:
        directory = Path(root) / 'results/autoresearch' / Path(*row['id'].split('::')) / 'model'
        predictor = Predictor(directory, device='cuda', dtype=jnp.float16, tta_level=0)

        def predict(images, model=predictor):
            return model.predict(images).argmax(1).astype(np.int64)

        measured = evaluate(predict, arrays['x_val'], arrays['y_val'])
        measured.update(
            seed=row['seed'],
            id=row['id'],
            size=sum((directory / f).stat().st_size for f in ('model.safetensors', 'config.json')),
            artifact_sha256=paths.sha256(directory / 'model.safetensors'),
        )
        runs.append(measured)
        print(f'd3 seed {row["seed"]}: {measured["accuracy"]:.4f}, {measured["latency"]:.3f} ms', flush=True)
        del predictor
        gc.collect()
    result = {
        'accuracy': statistics.mean(r['accuracy'] for r in runs),
        'accuracy_sd': statistics.stdev(r['accuracy'] for r in runs),
        'latency': statistics.median(t for r in runs for t in r['latency_ms_samples']),
        'size': statistics.mean(r['size'] for r in runs),
        'runs': runs,
        'dataset_sha256': DATASET_SHA256,
        'precision': 'float16',
        'batch': 256,
        'score': 100.0,
        'device': str(jax.devices()[0]),
        'jax': jax.__version__,
        'timing_scope': runs[0]['timing_scope'],
    }
    write(Path(output), result)
    return result


def sample(trial, max_epochs: int = 100) -> dict:
    initialization = trial.suggest_categorical('initialization', ['imagenet', 'scratch'])
    optimizer = trial.suggest_categorical('optimizer', ['sgd', 'adamw'])
    lr_key = f'lr_{initialization}_{optimizer}'
    lr_bounds = (1e-4, 0.03) if initialization == 'imagenet' else (0.01, 0.3)
    if optimizer == 'adamw':
        lr_bounds = (1e-5, 0.003)
    hp = {
        'initialization': initialization,
        'stem': trial.suggest_categorical('stem', ['imagenet', 'cifar']),
        'resolution': trial.suggest_categorical('resolution', [32, 48, 64, 96]),
        'optimizer': optimizer,
        'epochs': trial.suggest_int('epochs', 20, max_epochs, step=10),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
        'lr': trial.suggest_float(lr_key, *lr_bounds, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 0.01, log=True),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.2),
        'warmup': trial.suggest_int('warmup', 0, 5),
        'min_lr_ratio': trial.suggest_float('min_lr_ratio', 0.0, 0.1),
        'augment': trial.suggest_categorical('augment', ['basic', 'randaugment', 'trivialaugment']),
        'mixup': trial.suggest_categorical('mixup', [0.0, 0.2, 0.8]),
        'cutmix': trial.suggest_categorical('cutmix', [0.0, 1.0]),
        'erase': trial.suggest_categorical('erase', [0.0, 0.1, 0.25]),
        'precision': 'float16',
    }
    if optimizer == 'sgd':
        hp['momentum'] = trial.suggest_float('momentum', 0.85, 0.95)
    if initialization == 'imagenet':
        hp['backbone_lr_ratio'] = trial.suggest_float('backbone_lr_ratio', 0.05, 1.0, log=True)
        hp['freeze_epochs'] = trial.suggest_categorical('freeze_epochs', [0, 2, 5])
    return hp


def child(python: str, payload: dict, output: Path, seconds: float) -> dict:
    if seconds < 1:
        raise TimeoutError('no campaign allowance remains')
    request = output.with_suffix('.request.json')
    write(request, payload)
    env = {**os.environ, 'PYTHONPATH': str(paths.ROOT / 'src'), 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4'}
    subprocess.run(
        [python, '-m', 'cifarq.resnet18_campaign', '--request', str(request), '--output', str(output)],
        check=True,
        timeout=seconds,
        env=env,
    )
    return json.loads(output.read_text())


@dataclass
class _Campaign:
    root: str
    data: str
    torch_python: str
    out: Path
    budget_file: Path
    budget: dict
    started: float
    deadline: float
    reserve: float
    reference: dict = field(default_factory=dict)

    def update(self, stage: str, **extra: object) -> None:
        spent = time.monotonic() - self.started
        write(self.budget_file, {**self.budget, 'spent_seconds': self.budget['spent_seconds'] + spent, 'stage': stage, **extra})
        progress = {'stage': stage, 'spent_seconds': spent, 'remaining_seconds': max(0, self.deadline - time.monotonic()), **extra}
        write(self.out / 'progress.json', progress)
        if os.environ.get('RUNQ_JOB_DIR'):
            write(Path(os.environ['RUNQ_JOB_DIR']) / 'progress.json', progress)
            session_cmd('now', f'{stage}: {spent / 3600:.2f} hours in this execution; local T4 only.')
        print(json.dumps(progress), flush=True)

    def train_one(self, name: str, config: dict, cap: float) -> dict:
        output = self.out / f'{name}.json'
        if output.exists():
            return json.loads(output.read_text())
        remaining = min(cap, self.deadline - time.monotonic() - 30)
        if remaining < 120:
            raise TimeoutError('campaign allowance exhausted')
        self.update(name)
        try:
            result = child(
                self.torch_python,
                {
                    'action': 'train',
                    'data': self.data,
                    'output': str(self.out / name),
                    'reference': self.reference,
                    'config': config,
                    'seconds': max(1, remaining - 90),
                },
                output,
                remaining,
            )
        finally:
            self.update(name)
        session_cmd(
            'note',
            '--id',
            'E1',
            '--result',
            f'{name}: score={result["score"]:.3f}, accuracy={result["accuracy"]:.5f}, '
            f'latency={result["latency"]:.3f} ms, bytes={result["size"]}; artifacts {output}',
            '--outcome',
            'inconclusive' if name == 'correctness-smoke' else 'promising',
            '--decision',
            'Completed measurement only; continue the frozen plan without a champion or significance claim.',
        )
        return result

    def load_reference(self) -> dict:
        self.update('reference')
        reference_file = self.out / 'reference.json'
        if reference_file.exists():
            self.reference = json.loads(reference_file.read_text())
            return self.reference
        processes = []
        reference_deadline = min(self.deadline, time.monotonic() + 1800)
        for repeat in range(3):
            result = child(
                sys.executable,
                {'action': 'reference', 'data': self.data, 'root': self.root, 'output': str(self.out / f'reference-process-{repeat}.json')},
                self.out / f'reference-process-{repeat}.json',
                reference_deadline - time.monotonic(),
            )
            processes.append(result)
            self.update('reference', process=repeat + 1)
        self.reference = {
            **processes[0],
            'latency': statistics.median(p['latency'] for p in processes),
            'process_latencies': [p['latency'] for p in processes],
        }
        write(reference_file, self.reference)
        return self.reference

    def _trial_objective(self, trial: optuna.Trial) -> float:
        config = {**sample(trial), 'seed': 40}
        trial.set_user_attr('config', config)
        result = self.train_one(f'trial-{trial.number:04d}', config, min(3600, self.deadline - time.monotonic() - self.reserve))
        for key in ('accuracy', 'latency', 'size', 'score', 'saved'):
            trial.set_user_attr(key, result[key])
        return result['score']

    def search(self, fine: dict, scratch: dict, trials: int) -> tuple[dict, int]:
        import optuna

        study = optuna.create_study(
            study_name='resnet18-primary-score-v1',
            storage=f'sqlite:///{self.out / "optuna.db"}',
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=20261116, multivariate=True),
            load_if_exists=True,
        )
        if not study.trials:
            for result in (fine, scratch):
                study.add_trial(
                    optuna.trial.create_trial(
                        value=result['score'],
                        user_attrs={key: result[key] for key in ('config', 'accuracy', 'latency', 'size', 'score', 'saved')},
                    )
                )
        for interrupted in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
            study.tell(interrupted.number, state=optuna.trial.TrialState.FAIL)
        remaining_trials = max(0, trials - len(study.trials))
        if remaining_trials:
            study.optimize(
                self._trial_objective,
                n_trials=remaining_trials,
                timeout=max(0, self.deadline - time.monotonic() - self.reserve),
                catch=(subprocess.CalledProcessError, subprocess.TimeoutExpired, TimeoutError),
                callbacks=[stop_invalid_streak],
            )
        candidates = [
            {'name': 'imagenet-baseline', **fine},
            {'name': 'scratch-baseline', **scratch},
            *[
                {'name': f'trial-{t.number:04d}', 'score': t.value, 'config': t.user_attrs['config']}
                for t in study.trials
                if t.value is not None and t.params
            ],
        ]
        return max(candidates, key=lambda r: r['score']), len([t for t in study.trials if t.value is not None])


def run(root: str, data: str, torch_python: str, hours: float = 24.0, trials: int = 50) -> dict:
    """Run reference, fine-tuning, scratch, Optuna, and fresh-seed confirmation sequentially on one queued GPU."""
    if not 0 < hours <= 24 or trials < 1:
        raise ValueError('campaign exceeds approved limits')
    root_path = Path(root).resolve()
    out = root_path / 'results/autoresearch/resnet18-score'
    out.mkdir(parents=True, exist_ok=True)
    budget_file = out / 'budget.json'
    budget = json.loads(budget_file.read_text()) if budget_file.exists() else {'limit_seconds': hours * 3600, 'spent_seconds': 0}
    if budget.get('active'):
        # A killed driver cannot run finally; charge elapsed time (including downtime) on recovery.
        budget['spent_seconds'] = max(budget['spent_seconds'], budget['spent_at_start'] + time.time() - budget['wall_started'])
    budget.update(active=True, wall_started=time.time(), spent_at_start=budget['spent_seconds'])
    started = time.monotonic()
    allowance = min(hours * 3600, budget['limit_seconds']) - budget['spent_seconds']
    ctx = _Campaign(root, data, torch_python, out, budget_file, budget, started, started + allowance, min(3 * 3600, allowance * 0.15))
    try:
        reference = ctx.load_reference()
        ctx.train_one(
            'correctness-smoke',
            {'initialization': 'scratch', 'stem': 'cifar', 'resolution': 32, 'epochs': 1, 'warmup': 0},
            600,
        )
        session_cmd('milestone', 'M0', '--state', 'done', '--note', 'Reference frozen; one-epoch training smoke passed.')
        session_cmd('milestone', 'M1', '--state', 'active')
        fine = ctx.train_one('imagenet-baseline', {'initialization': 'imagenet', 'epochs': 30}, 2 * 3600)
        session_cmd('milestone', 'M1', '--state', 'done')
        session_cmd('milestone', 'M2', '--state', 'active')
        scratch = ctx.train_one('scratch-baseline', {'initialization': 'scratch', 'epochs': 100, 'lr': 0.1}, 4 * 3600)
        session_cmd('milestone', 'M2', '--state', 'done')
        session_cmd('milestone', 'M3', '--state', 'active')
        best, completed_trials = ctx.search(fine, scratch, trials)
        session_cmd('milestone', 'M3', '--state', 'done', '--note', 'Bounded search ended; failures remain in SQLite.')
        session_cmd('milestone', 'M4', '--state', 'active')
        confirmations = []
        for seed in (71, 72, 73):
            if ctx.deadline - time.monotonic() < 600:
                break
            confirmations.append(
                ctx.train_one(f'confirmation-{seed}', {**best['config'], 'seed': seed}, min(3600, ctx.deadline - time.monotonic() - 30))
            )
        summary = {
            'reference': reference,
            'fine_tuned_score': fine['score'],
            'scratch_score': scratch['score'],
            'best_search': best,
            'completed_trials': completed_trials,
            'confirmation_scores': [r['score'] for r in confirmations],
            'status': 'confirmation-needed' if len(confirmations) < 3 else 'measured-not-significance-tested',
            'confirmed_recipe_score': None,
        }
        if confirmations:
            from .pipeline_score import objective

            measured = {
                'accuracy': statistics.mean(r['accuracy'] for r in confirmations),
                'latency': statistics.median(t for r in confirmations for t in r['latency_ms_samples']),
                'size': statistics.mean(r['size'] for r in confirmations),
            }
            summary['confirmed_recipe_score'] = objective(measured, reference)
            summary['confirmed_recipe_accuracy'] = measured['accuracy']
            summary['confirmed_recipe_accuracy_sd'] = statistics.stdev(r['accuracy'] for r in confirmations) if len(confirmations) > 1 else None
        write(out / 'summary.json', summary)
        session_cmd('milestone', 'M4', '--state', 'done' if len(confirmations) == 3 else 'blocked', '--note', summary['status'])
        ctx.update('finished', best_score=best['score'])
        return summary
    finally:
        ctx.update('stopped', active=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.request.read_text())
    action = payload.pop('action')
    if action == 'reference':
        result = d3_reference(**payload)
    elif action == 'train':
        from .resnet18 import train

        result = train(**payload)
    else:
        raise ValueError(f'unknown action {action}')
    write(args.output, result)


if __name__ == '__main__':
    main()
