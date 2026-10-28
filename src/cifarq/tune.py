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


def job_options(params: dict) -> dict:
    """A trial's queue options: its expected minutes and GPU memory by width and length (an L4 or the 4060)."""
    w, e = params.get('width', 32), params.get('epochs', 10)
    return {'minutes': round(1 + e * (w / 32) ** 2 * 0.06, 1), 'gpu_mb': 1500 + 40 * w}


def on_vm(n_trials: int = 80, study_name: str = 'cifar-on-vm', candidates=(1, 2, 3, 4, 6), seed: int = 0,
          data: str | None = None, resume_from: str | None = None) -> dict:
    """Mode 2, run as one queue job on a VM: how many trainings the GPU runs at once with gain (``calibrate``, on
    2-epoch width-32 runs), then Optuna over the whole ``space`` with that many trials in flight as processes on the GPU, the study in
    ``study.db`` in the job's directory (brought back with the job). ``resume_from``: a study database of a run that
    died (the last ``study.db`` its job brought back while it ran, sent as this job's data): copied in and continued,
    its trials in flight run again (``recover_running``). The best trial, the calibration and the timing."""
    import optuna
    import shutil
    from solve_queue.hpo import optuna_optimize, recover_running
    from solve_queue.parallel import GPUProcessExecutor, calibrate
    t0 = time.time()
    k, table = calibrate(TRAIN, {'epochs': 2, 'width': 32, 'data': data}, candidates=candidates)
    t_cal = time.time() - t0
    storage = f'sqlite:///{Path.cwd() / "study.db"}'
    if resume_from:
        shutil.copy(resume_from, Path.cwd() / 'study.db')
        study_name = optuna.study.get_all_study_names(storage)[-1]
    study = optuna.create_study(direction='maximize', storage=storage, study_name=study_name, load_if_exists=True,
                                sampler=sampler(seed))
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
               'best_value': best.value if best else None, 'best_trial': best.number if best else None,
               'best_params': best.params if best else None,
               'best_test_acc': best.user_attrs.get('test_acc') if best else None}
        Path('progress.json.part').write_text(json.dumps(doc, indent=1) + '\n')
        Path('progress.json.part').replace('progress.json')

    with GPUProcessExecutor(k) as ex:
        optuna_optimize(study, TRAIN, space, n_trials=n_trials, executor=ex, concurrency=k, fixed={'data': data},
                        log=progress)
    best = study.best_trial
    out = {'best_value': best.value, 'best_params': best.params, 'best_test_acc': best.user_attrs.get('test_acc'),
           'best_train_seconds': best.user_attrs.get('train_seconds'),
           'trials': len(study.trials), 'concurrent': k, 'calibration': table, 'calibration_seconds': round(t_cal, 1),
           'search_seconds': round(time.time() - t0 - t_cal, 1), 'device': best.user_attrs.get('device')}
    Path('summary.json').write_text(json.dumps(out, indent=1) + '\n')
    return out
