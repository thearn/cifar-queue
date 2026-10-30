"""The autoresearch session's search spaces (docs/autoresearch/2026-10-29-footprint-arch/PLAN.md, M3): one per family
trained by ``airbench.train``, each the family's architecture (sampled freely; configurations over ``param_cap`` are
pruned at step 0 by the trainer, so the sampler learns the boundary) plus the shared recipe (``recipe``): epochs up to
``max_epochs``, batch, SGD's decoupled learning rate, momentum and weight decay, label smoothing, schedule, bias
scaler, batch-norm momentum, translation, cutout, policy augmentation, test-time views, logit scale.

``SPACES[family](trial, cap, max_epochs)`` gives ``airbench.train``'s arguments; ``WARM[family]`` the points each study
starts from (as ``trial.params``); ``CALIBRATE[family]`` a short training to time concurrency with.
"""
from __future__ import annotations

TRAIN = 'cifarq.airbench:train'


def recipe(trial, max_epochs: float = 40, kd: str = '') -> dict:
    s, c = trial.suggest_float, trial.suggest_categorical
    p = {'epochs': s('epochs', 20.0, max_epochs), 'batch_size': c('batch_size', [256, 512, 1024]),
         'lr': s('lr', 2.0, 30.0, log=True), 'momentum': s('momentum', 0.7, 0.95),
         'weight_decay': s('weight_decay', 0.003, 0.05, log=True), 'bias_scaler': s('bias_scaler', 4.0, 128.0, log=True),
         'label_smoothing': s('label_smoothing', 0.0, 0.3), 'warmup_frac': s('warmup_frac', 0.02, 0.3),
         'final_lr_frac': s('final_lr_frac', 0.0, 0.15), 'bn_momentum': s('bn_momentum', 0.05, 0.6, log=True),
         'translate': c('translate', [0, 2, 3, 4]), 'cutout': c('cutout', [0, 4, 8, 12, 16]),
         'whiten_bias_epochs': trial.suggest_int('whiten_bias_epochs', 0, 6),
         'scaling_factor': s('scaling_factor', 1 / 20, 1 / 3, log=True), 'tta_level': c('tta_level', [0, 1, 2]),
         'lookahead': c('lookahead', [True, False])}
    p['augment'] = c('augment', ['none', 'ta_wide', 'randaugment'])
    if p['augment'] == 'randaugment':
        p['augment_m'] = trial.suggest_int('augment_m', 2, 15)
    if kd:
        p['kd_teacher'] = kd
        p['kd_weight'] = s('kd_weight', 0.0, 0.95) if c('kd', [True, False]) else 0.0
        p['kd_temp'] = s('kd_temp', 1.0, 8.0, log=True)
    return p


def airbench_space(trial, cap: int = 2_000_000, max_epochs: float = 40, kd: str = '') -> dict:
    c = trial.suggest_categorical
    p = {'arch': 'airbench', 'depth': c('depth', [2, 3]),
         'widths': (c('w1', [32, 48, 64, 80, 96, 112, 128]), c('w2', [96, 128, 160, 192, 224, 256, 288, 320, 384]),
                    c('w3', [96, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512]))}
    return {**p, **recipe(trial, max_epochs, kd), 'param_cap': cap}


def resnet9_space(trial, cap: int = 2_000_000, max_epochs: float = 40, kd: str = '') -> dict:
    c = trial.suggest_categorical
    p = {'arch': 'resnet9', 'arch_kw': {'width': c('width', [16, 20, 24, 28, 32, 35]),
                                        'activation': c('activation', ['relu', 'gelu', 'celu'])}}
    return {**p, **recipe(trial, max_epochs, kd), 'param_cap': cap}


def convmixer_space(trial, cap: int = 2_000_000, max_epochs: float = 40, kd: str = '') -> dict:
    c = trial.suggest_categorical
    p = {'arch': 'convmixer', 'arch_kw': {'dim': c('dim', [128, 192, 256, 320]), 'depth': trial.suggest_int('cm_depth', 4, 20),
                                          'kernel': c('kernel', [3, 5, 7, 9]), 'patch': c('patch', [1, 2])}}
    return {**p, **recipe(trial, max_epochs, kd), 'param_cap': cap}


SPACES = {'airbench': airbench_space, 'resnet9': resnet9_space, 'convmixer': convmixer_space}

# airbench94's recipe (1.97 M, widths 64/256/256) as a point of the airbench space, at the cap's epochs
_AB94 = {'depth': 2, 'w1': 64, 'w2': 256, 'w3': 256, 'epochs': 9.9, 'batch_size': 1024, 'lr': 11.5, 'momentum': 0.85,
         'weight_decay': 0.0153, 'bias_scaler': 64.0, 'label_smoothing': 0.2, 'warmup_frac': 0.23, 'final_lr_frac': 0.07,
         'bn_momentum': 0.4, 'translate': 2, 'cutout': 0, 'whiten_bias_epochs': 3, 'scaling_factor': 1 / 9, 'tta_level': 2,
         'lookahead': True, 'augment': 'none'}
# airbench96's recipe at the cap's 40 epochs (M2: depth 3 at 64/192/192 the best screen, 95.48 % val; 64/160/224
# 95.34 %; depth 2 at 64/256/256 94.96 %)
_R96 = {**_AB94, 'epochs': 40.0, 'cutout': 12, 'translate': 4, 'lr': 9.0, 'weight_decay': 0.012, 'warmup_frac': 0.1,
        'final_lr_frac': 0.0, 'bn_momentum': 0.1}
WARM = {'airbench': [{**_R96, 'depth': 3, 'w2': 192, 'w3': 192}, {**_R96, 'depth': 3, 'w2': 160, 'w3': 224}, _R96],
        'resnet9': [{**{k: v for k, v in _AB94.items() if k not in ('depth', 'w1', 'w2', 'w3')}, 'width': 32,
                     'activation': 'gelu', 'epochs': 30.0}],
        'convmixer': [{**{k: v for k, v in _AB94.items() if k not in ('depth', 'w1', 'w2', 'w3')}, 'dim': 256, 'cm_depth': 8,
                       'kernel': 5, 'patch': 2, 'epochs': 30.0, 'augment': 'randaugment', 'augment_m': 12}]}
CALIBRATE = {'airbench': {'epochs': 2, 'widths': (64, 256, 256)},
             'resnet9': {'epochs': 2, 'arch': 'resnet9', 'arch_kw': {'width': 32}},
             'convmixer': {'epochs': 1, 'arch': 'convmixer', 'arch_kw': {'dim': 256, 'depth': 8, 'kernel': 5, 'patch': 2}}}
