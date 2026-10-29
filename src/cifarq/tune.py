"""The hyperparameter search: its space, the per-trial queue options, and the driver that runs a whole Optuna search
inside one queue job on a VM (mode 2).

``space`` is the whole training space of ``train.DEFAULTS``, conditional where a choice makes others moot (an
augmentation off has no strength; each optimiser its own learning rate range and parameters). Every optional feature
has a switch (off) and, when on, a strength on a log scale reaching down to where it has almost no effect, so the
search can turn each one down by degrees or off altogether:

- length and model: epochs 6-30, width 16-64, activation (relu, gelu, silu, celu with its alpha), dropout (on/off,
  strength), the classifier head (linear, cosine, batch-normalised logits) with its logit scale, input whitening
  (on/off, its epsilon)
- optimiser: SGD (lr 0.005-0.8, momentum 0-0.97, Nesterov) or AdamW (lr 5e-5-1e-2, betas), weight decay (its own range
  for each), batch size, label smoothing, gradient clipping and weight EMA (each on/off, strength or horizon)
- schedule: warmup fraction and start, decay shape (cosine, linear, exponential, none), final fraction
- augmentation, each on or off: translation, rotation, scale, flip, brightness, contrast, saturation, cutout, and mixup
  or cutmix (alpha, probability)
- evaluation: test-time flip averaging on or off

The objective is the validation accuracy (5000 held-out training images); the test accuracy is kept as an attribute.
The sampler is multivariate TPE over the conditional groups, with a constant liar so trials in flight are not proposed
twice.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

TRAIN = 'cifarq.train:train'


# David Page's baseline DAWNBench recipe (ResNet-9, 24 epochs) as a point of this space, enqueued as each search's
# first trial: 64 channels, batch 512, SGD with Nesterov momentum 0.9, peak learning rate 0.4 (linear up over 5 of 24
# epochs from 0, then linearly down to 0), weight decay 5e-4, label smoothing 0.2, logits x 0.125, random crops from
# 4-pixel reflect padding, flips, 8x8 cutout on every image; no whitening, EMA or test-time flips (his later versions)
PAGE = {'epochs': 24, 'width': 64, 'activation': 'relu', 'batch_size': 512, 'use_dropout': False,
        'use_label_smoothing': True, 'label_smoothing': 0.2, 'head': 'linear', 'use_logit_scale': True, 'logit_scale': 0.125, 'whiten': False,
        'tta_flip': False, 'optimizer': 'sgd', 'sgd_lr': 0.4, 'momentum': 0.9, 'nesterov': True, 'sgd_weight_decay': 5e-4,
        'clip': False, 'ema': False, 'warmup_frac': 5 / 24, 'warmup_start_frac': 0.005, 'decay': 'linear',
        'final_lr_frac': 0.001, 'use_translate': True, 'aug_translate': 4.0, 'use_rotate': False, 'use_scale': False,
        'aug_flip': 0.5, 'use_brightness': False, 'use_contrast': False, 'use_saturation': False, 'use_cutout': True,
        'cutout_size': 8, 'cutout_prob': 1.0, 'mix': 'none'}


def sampler(seed: int = 0):
    import warnings
    import optuna
    with warnings.catch_warnings():                       # (multivariate, group and constant_liar are marked experimental)
        warnings.simplefilter('ignore', optuna.exceptions.ExperimentalWarning)
        return optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True, constant_liar=True, n_startup_trials=16)


def space(trial) -> dict:
    s = trial.suggest_float
    c = trial.suggest_categorical
    p = {'epochs': trial.suggest_int('epochs', 6, 30, step=2), 'width': c('width', [16, 24, 32, 48, 64]),
         'activation': c('activation', ['relu', 'gelu', 'silu', 'celu']), 'batch_size': c('batch_size', [256, 512, 1024])}
    if p['activation'] == 'celu':
        p['celu_alpha'] = s('celu_alpha', 0.05, 1.0, log=True)
    # the classifier head and the scale on its logits, each head its own range (a cosine's logits lie in [-1, 1], a
    # batch-normalised head's are unit variance, a linear head's start large)
    p['head'] = c('head', ['linear', 'cosine', 'bn'])
    if p['head'] == 'linear':
        p['logit_scale'] = s('logit_scale', 0.03, 2.0, log=True) if c('use_logit_scale', [False, True]) else 1.0
    elif p['head'] == 'cosine':
        p['logit_scale'] = s('cosine_scale', 2.0, 50.0, log=True)
    else:
        p['logit_scale'] = s('bn_scale', 0.3, 10.0, log=True)
    if c('whiten', [False, True]):
        p.update(whiten=True, whiten_eps=s('whiten_eps', 1e-4, 0.1, log=True))
    p['tta_flip'] = c('tta_flip', [False, True])
    p['dropout'] = s('dropout', 0.005, 0.5, log=True) if c('use_dropout', [False, True]) else 0.0
    p['label_smoothing'] = s('label_smoothing', 0.002, 0.3, log=True) if c('use_label_smoothing', [False, True]) else 0.0
    p['optimizer'] = c('optimizer', ['sgd', 'adamw'])
    if p['optimizer'] == 'sgd':
        p.update(lr=s('sgd_lr', 0.005, 0.8, log=True), momentum=s('momentum', 0.0, 0.97), nesterov=c('nesterov', [True, False]),
                 weight_decay=s('sgd_weight_decay', 1e-6, 5e-3, log=True))
    else:
        p.update(lr=s('adamw_lr', 5e-5, 1e-2, log=True), beta1=s('beta1', 0.7, 0.97), beta2=c('beta2', [0.98, 0.99, 0.999]),
                 weight_decay=s('adamw_weight_decay', 1e-4, 0.5, log=True))
    p['grad_clip'] = s('grad_clip', 0.25, 50.0, log=True) if c('clip', [False, True]) else 0.0
    # (the EMA's horizon, 1 / (1 - decay) steps, from 5 to 2000: short is nearly no averaging)
    p['ema_decay'] = 1 - 1 / s('ema_horizon', 5.0, 2000.0, log=True) if c('ema', [False, True]) else 0.0
    p.update(warmup_frac=s('warmup_frac', 0.0, 0.5), warmup_start_frac=s('warmup_start_frac', 0.005, 1.0, log=True),
             decay=c('decay', ['cosine', 'linear', 'exponential', 'none']))
    p['final_lr_frac'] = 1.0 if p['decay'] == 'none' else s('final_lr_frac', 1e-3, 0.3, log=True)
    for name, low, high in (('translate', 0.1, 6.0), ('rotate', 0.2, 25.0), ('scale', 0.002, 0.3),
                            ('brightness', 0.005, 0.5), ('contrast', 0.005, 0.5), ('saturation', 0.005, 0.6)):
        p[f'aug_{name}'] = s(f'aug_{name}', low, high, log=True) if c(f'use_{name}', [False, True]) else 0.0
    p['aug_flip'] = s('aug_flip', 0.0, 0.5)
    if c('use_cutout', [False, True]):
        p.update(cutout_size=trial.suggest_int('cutout_size', 2, 16, step=2), cutout_prob=s('cutout_prob', 0.05, 1.0, log=True))
    else:
        p['cutout_size'] = 0
    p['mix'] = c('mix', ['none', 'mixup', 'cutmix'])
    if p['mix'] != 'none':
        p.update(mix_alpha=s('mix_alpha', 0.05, 2.0, log=True), mix_prob=s('mix_prob', 0.05, 1.0, log=True))
    return p


def train_params(params: dict) -> dict:
    """A point of the space as the search records it (a trial's ``params``, or ``PAGE``) as ``train``'s arguments."""
    import optuna
    return space(optuna.trial.FixedTrial(params))


# on a VM a trial runs as a fork of solve-queue's persistent worker, which imported torch and the training code and
# loaded the arrays once (the local GPU's jobs run as their own processes either way)
WORKER = {'preload': ['torch', 'torch.nn.functional', 'cifarq.train', 'cifarq.data:warm']}


def job_options(params: dict) -> dict:
    """A trial's queue options: its expected minutes and GPU memory by width and length (an L4 or the 4060), and the
    persistent worker on a VM."""
    w, e = params.get('width', 32), params.get('epochs', 10)
    # (two trials on a VM take turns at its card: the next one's set-up overlaps this one's training)
    return {'minutes': round(1 + e * (w / 32) ** 2 * 0.06, 1), 'gpu_mb': 1500 + 40 * w, 'worker': WORKER,
            'vm_env': {'SOLVEQ_DEVICE_TURNS': '1'}}


# early stopping chosen from this model's own curves (scripts/rung_check.py on 161 complete trials, 2026-10-29):
# Page-style schedules gain most of their accuracy as the rate decays, so accuracy at 30 % of training ranks the final
# best poorly (rho 0.65; keeping the top third there cut 11 of the 17 trials that finished in the top 10 %). Rungs at
# 10/30/60 % keeping the best two thirds, on the best value so far, used 51 % of the compute and kept 15 of 17.
PRUNE = {'rungs': [10, 30, 60], 'keep': 2 / 3, 'min_trials': 6}


def _checkpoint(db: Path, out: Path):
    """A consistent copy of the study (SQLite's backup, not a file copy mid-write) as ``study-ckpt.db``: the job's
    checkpoint, sent to the bucket as it changes and restored by the queue before a lost VM's rerun."""
    import sqlite3
    tmp = out.with_suffix('.part')
    src, dst = sqlite3.connect(db), sqlite3.connect(tmp)
    try:
        src.backup(dst)
    finally:
        src.close(); dst.close()
    tmp.replace(out)


def _best_by_time(study) -> list:
    """[(seconds from the first trial's start, best value so far)] at each completed trial that raised it."""
    done = sorted((t for t in study.trials if t.state.name == 'COMPLETE' and t.datetime_complete), key=lambda t: t.datetime_complete)
    if not done:
        return []
    t0 = min(t.datetime_start for t in study.trials if t.datetime_start)
    out, best = [], float('-inf')
    for t in done:
        if t.value > best:
            best = t.value
            out.append((round((t.datetime_complete - t0).total_seconds()), round(best, 4)))
    return out


def on_vm(n_trials: int = 80, study_name: str = 'cifar-on-vm', candidates=(1, 2, 3, 4, 6), seed: int = 0,
          data: str | None = None, resume_from: str | None = None, pruner: str = 'none', concurrent: int | None = None,
          minutes: float | None = None, turns: bool = True) -> dict:
    """Mode 2, run as one queue job on a VM: how many trainings the GPU runs at once with gain (``calibrate``, on
    2-epoch width-32 runs), then Optuna over the whole ``space`` with that many trials in flight as processes on the GPU, the study in
    ``study.db`` in the job's directory (brought back with the job). ``resume_from``: a study database of a run that
    died (the last ``study.db`` its job brought back while it ran, sent as this job's data): copied in and continued,
    its trials in flight run again (``recover_running``). ``pruner``: early stopping by the study (``asha``,
    ``hyperband``, ``median``, ``none``: ``solve_queue.hpo.PRUNERS``), each trial reporting its validation accuracy after
    every epoch; ``rungs`` is the one chosen for this model (``PRUNE``), and every pruner sees the best value so far.
    ``concurrent``: trainings at once, given (a stored calibration) rather than measured. ``minutes``: stop
    asking for trials after this long (a fixed budget; those in flight finish). ``turns``: when one training at a time is
    best, two processes take turns at the card (``solve_queue.gpu.device_turn``), so the next trial's set-up overlaps this
    one's training. The best trial, the calibration and the timing."""
    import optuna
    import shutil
    from solve_queue.hpo import optuna_optimize, recover_running
    from solve_queue.parallel import GPUProcessExecutor, calibrate
    from solve_queue.hpo import make_pruner
    t0 = time.time()
    if concurrent:
        k, table = concurrent, [{'concurrent': concurrent, 'given': True}]
    else:
        k, table = calibrate(TRAIN, {'epochs': 2, 'width': 32, 'data': data}, candidates=candidates)
    t_cal = time.time() - t0
    storage = f'sqlite:///{Path.cwd() / "study.db"}'
    ckpt = Path.cwd() / 'study-ckpt.db'
    if not resume_from and not (Path.cwd() / 'study.db').exists() and ckpt.exists():
        # a rerun after a lost VM: the queue restored the last checkpoint of the study (job.json checkpoint)
        resume_from = str(ckpt)
    if resume_from:
        if Path(resume_from).resolve() != (Path.cwd() / 'study.db').resolve():
            shutil.copy(resume_from, Path.cwd() / 'study.db')
        study_name = optuna.study.get_all_study_names(storage)[-1]
    study = optuna.create_study(direction='maximize', storage=storage, study_name=study_name, load_if_exists=True,
                                sampler=sampler(seed), pruner=make_pruner(pruner, **(PRUNE if pruner == 'rungs' else {})))
    if resume_from:
        recover_running(study)                               # (trials ran as processes here: those in flight run again)
    if not study.trials:
        study.enqueue_trial(PAGE)                            # Page's recipe first: the search starts from the reference
    def progress(line: str):
        """Each trial told: its line in the log, and progress.json (replaced whole) for whoever watches the job."""
        print(line, flush=True)
        done = [t for t in study.trials if t.state.name == 'COMPLETE']
        best = max(done, key=lambda t: t.value) if done else None
        doc = {'when': time.time(), 'trials': len(study.trials), 'complete': len(done), 'n_trials': n_trials, 'concurrent': k,
               'pruned': sum(1 for t in study.trials if t.state.name == 'PRUNED'),
               'best_value': best.value if best else None, 'best_trial': best.number if best else None,
               'best_params': best.params if best else None,
               'best_test_acc': best.user_attrs.get('test_acc') if best else None}
        Path('progress.json.part').write_text(json.dumps(doc, indent=1) + '\n')
        Path('progress.json.part').replace('progress.json')
        _checkpoint(Path.cwd() / 'study.db', ckpt)

    pool = k
    if turns and k == 1:
        import os
        os.environ['SOLVEQ_DEVICE_TURNS'] = '1'             # (inherited by the pool's processes)
        pool = 2
    with GPUProcessExecutor(pool) as ex:
        optuna_optimize(study, TRAIN, space, n_trials=n_trials, executor=ex, concurrency=pool, fixed={'data': data},
                        report_to=storage if pruner != 'none' else None, minutes=minutes, smooth=pruner != 'none',
                        log=progress)
    best = study.best_trial
    states = [t.state.name for t in study.trials]
    out = {'best_value': best.value, 'best_params': best.params, 'best_test_acc': best.user_attrs.get('test_acc'),
           'best_train_seconds': best.user_attrs.get('train_seconds'), 'pruner': pruner,
           'completed': states.count('COMPLETE'), 'pruned': states.count('PRUNED'), 'failed': states.count('FAIL'),
           # the best value as the search went (seconds since its start, value): the yield against time and units
           'best_by_time': _best_by_time(study),
           'trials': len(study.trials), 'concurrent': k, 'pool': pool, 'turns': pool != k,
           'calibration': table, 'calibration_seconds': round(t_cal, 1),
           'search_seconds': round(time.time() - t0 - t_cal, 1), 'device': best.user_attrs.get('device')}
    Path('summary.json').write_text(json.dumps(out, indent=1) + '\n')
    return out


def _ray_trainable(config: dict):
    """A Ray Tune trial: ``train`` with its config, reporting the validation accuracy after each epoch at the percent
    of training done (Ray's scheduler stops it there when it should), and its final result."""
    import optuna
    from ray import tune as rtune
    from .train import train
    config = dict(config)
    data = config.pop('data', None)
    # Ray's OptunaSearch passes the sampled values by their Optuna names (sgd_lr, use_cutout, ...): ``space`` run on a
    # trial fixed at those values gives train's own arguments, as in on_vm
    config = space(optuna.trial.FixedTrial(config))

    best = [float('-inf')]

    def report(step, value):
        best[0] = max(best[0], float(value))                  # the best so far (high-rate epochs jump around)
        rtune.report({'val_acc': best[0], 'progress': int(step)})
        return False                                       # (Ray stops a trial itself)
    out = train(data=data, report=report, **config)
    rtune.report({'val_acc': out['val_acc'], 'progress': 100, 'test_acc': out.get('test_acc'),
                  'train_seconds': out['train_seconds'], 'device': out['device']})


def on_vm_ray(n_trials: int = 80, scheduler: str = 'asha', concurrent: int = 1, minutes: float | None = None,
              seed: int = 0, data: str | None = None) -> dict:
    """Mode 2 driven by Ray Tune on the VM: the same space and Optuna's TPE sampler (Ray's ``OptunaSearch``, define-by-run
    over ``space``), Ray's own early stopping (``asha``: ``ASHAScheduler``; ``hyperband``: ``HyperBandScheduler``;
    ``none``: FIFO) on the percent of training done, ``concurrent`` trials at once sharing the GPU (each a 1/concurrent
    GPU share), and a time budget (``minutes``). The best trial and the counts, like ``on_vm``."""
    import ray
    from ray import tune as rtune
    from ray.tune.schedulers import ASHAScheduler, FIFOScheduler, HyperBandScheduler
    from ray.tune.search.optuna import OptunaSearch
    t0 = time.time()
    ray.init(include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
    # (Ray's rungs are geometric: grace 30 and factor 2 give 30 and 60 %, keeping half: the nearest to PRUNE that
    # simulate_rungs found safe, 57 % of the compute, 12 of 17 top trials kept)
    sched = {'asha': lambda: ASHAScheduler(time_attr='progress', max_t=100, grace_period=30, reduction_factor=2),
             'hyperband': lambda: HyperBandScheduler(time_attr='progress', max_t=100, reduction_factor=3),
             'none': lambda: FIFOScheduler()}[scheduler]()
    search = OptunaSearch(space=space, sampler=sampler(seed), metric='val_acc', mode='max')   # (module-level: Ray pickles it)
    tuner = rtune.Tuner(
        rtune.with_resources(_ray_trainable, {'cpu': 1, 'gpu': 1 / concurrent}),
        param_space={'data': data},
        tune_config=rtune.TuneConfig(metric='val_acc', mode='max', search_alg=search, scheduler=sched,
                                     num_samples=n_trials, max_concurrent_trials=concurrent,
                                     time_budget_s=60 * minutes if minutes else None),
        run_config=rtune.RunConfig(storage_path=str(Path.cwd() / 'ray'), name='mode2', verbose=0))
    results = tuner.fit()
    rows = [r for r in results if r.metrics]
    finished = [r for r in rows if r.metrics.get('progress') == 100]
    best = max(finished, key=lambda r: r.metrics['val_acc']) if finished else None
    out = {'driver': 'ray', 'scheduler': scheduler, 'trials': len(rows), 'completed': len(finished),
           'pruned': len(rows) - len(finished), 'concurrent': concurrent,
           'best_value': best.metrics['val_acc'] if best else None,
           'best_test_acc': best.metrics.get('test_acc') if best else None,
           'best_params': {k: v for k, v in (best.config or {}).items() if k != 'data'} if best else None,
           'search_seconds': round(time.time() - t0, 1)}
    Path('summary.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    ray.shutdown()
    return out
