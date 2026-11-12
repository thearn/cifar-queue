"""The autoresearch session's search spaces (docs/autoresearch/2026-10-29-footprint-arch/PLAN.md, M3): one per family
trained by ``airbench.train``, each the family's architecture (sampled freely; configurations over ``param_cap`` are
pruned at step 0 by the trainer, so the sampler learns the boundary) plus the shared recipe (``recipe``): epochs up to
``max_epochs``, batch, SGD's decoupled learning rate, momentum and weight decay, label smoothing, schedule, bias
scaler, batch-norm momentum, translation, cutout, policy augmentation, test-time views, logit scale.

(M3-airbench phase 1 found its best at w1 128 and batch 1024, the tops of their ranges: both widened for phase 2.)

``SPACES[family](trial, cap, max_epochs)`` gives ``airbench.train``'s arguments; ``WARM[family]`` the points each study
starts from (as ``trial.params``); ``CALIBRATE[family]`` a short training to time concurrency with.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    import optuna

TRAIN = 'cifarq.airbench:train'
TRAIN_TWO_SEEDS = 'cifarq.families:train_two_seeds'


def train_two_seeds(data: str | None = None, report: Any = None, **hp) -> dict:
    """A search trial on two seeds (session 2026-10-30-equivariant-small, PLAN §2.1): seed 0 trains with the study's
    early stopping (its curve is the one the pruner sees); a survivor trains again at seed 1, unpruned; the value is
    the mean validation accuracy. The test set stays sealed. One seed's search values ran 0.2-0.5 pp optimistic in
    the previous session."""
    from . import airbench

    hp = {**hp, 'test': 'sealed'}
    r0 = airbench.train(data, report=report, **{**hp, 'seed': 0})
    if r0.get('pruned'):
        return r0
    r1 = airbench.train(data, **{**hp, 'seed': 1})
    mean = lambda k: round((r0[k] + r1[k]) / 2, 5)
    return {
        **r0,
        'val_acc': mean('val_acc'),
        'val_acc_notta': mean('val_acc_notta'),
        'val_acc_seeds': [r0['val_acc'], r1['val_acc']],
        'val_acc_notta_seeds': [r0['val_acc_notta'], r1['val_acc_notta']],
        'train_seconds': round(r0['train_seconds'] + r1['train_seconds'], 1),
    }


def recipe(trial: optuna.Trial, max_epochs: float = 40, kd: str = '') -> dict:
    s, c = trial.suggest_float, trial.suggest_categorical
    p = {
        'epochs': s('epochs', 20.0, max_epochs),
        'batch_size': c('batch_size', [256, 512, 1024, 2048]),
        'lr': s('lr', 2.0, 30.0, log=True),
        'momentum': s('momentum', 0.7, 0.95),
        'weight_decay': s('weight_decay', 0.003, 0.05, log=True),
        'bias_scaler': s('bias_scaler', 4.0, 128.0, log=True),
        'label_smoothing': s('label_smoothing', 0.0, 0.3),
        'warmup_frac': s('warmup_frac', 0.02, 0.3),
        'final_lr_frac': s('final_lr_frac', 0.0, 0.15),
        'bn_momentum': s('bn_momentum', 0.05, 0.6, log=True),
        'translate': c('translate', [0, 2, 3, 4]),
        'cutout': c('cutout', [0, 4, 8, 12, 16]),
        'whiten_bias_epochs': trial.suggest_int('whiten_bias_epochs', 0, 6),
        'scaling_factor': s('scaling_factor', 1 / 20, 1 / 3, log=True),
        'tta_level': c('tta_level', [0, 1, 2]),
        'lookahead': c('lookahead', [True, False]),
    }
    p['augment'] = c('augment', ['none', 'ta_wide', 'randaugment'])
    if p['augment'] == 'randaugment':
        p['augment_m'] = trial.suggest_int('augment_m', 2, 15)
    if kd:
        p['kd_teacher'] = kd
        p['kd_weight'] = s('kd_weight', 0.0, 0.95) if c('kd', [True, False]) else 0.0
        p['kd_temp'] = s('kd_temp', 1.0, 8.0, log=True)
    return p


def airbench_space(trial: optuna.Trial, cap: int = 2_000_000, max_epochs: float = 40, kd: str = '') -> dict:
    c = trial.suggest_categorical
    p = {
        'arch': 'airbench',
        'depth': c('depth', [2, 3]),
        'widths': (
            c('w1', [32, 48, 64, 80, 96, 112, 128, 160, 192]),
            c('w2', [96, 128, 160, 192, 224, 256, 288, 320, 384]),
            c('w3', [96, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512]),
        ),
    }
    return {**p, **recipe(trial, max_epochs, kd), 'param_cap': cap}


def resnet9_space(trial: optuna.Trial, cap: int = 2_000_000, max_epochs: float = 40, kd: str = '') -> dict:
    c = trial.suggest_categorical
    p = {'arch': 'resnet9', 'arch_kw': {'width': c('width', [16, 20, 24, 28, 32, 35]), 'activation': c('activation', ['relu', 'gelu', 'celu'])}}
    return {**p, **recipe(trial, max_epochs, kd), 'param_cap': cap}


def convmixer_space(trial: optuna.Trial, cap: int = 2_000_000, max_epochs: float = 40, kd: str = '') -> dict:
    c = trial.suggest_categorical
    p = {
        'arch': 'convmixer',
        'arch_kw': {
            'dim': c('dim', [128, 192, 256, 320]),
            'depth': trial.suggest_int('cm_depth', 4, 20),
            'kernel': c('kernel', [3, 5, 7, 9]),
            'patch': c('patch', [1, 2]),
        },
    }
    return {**p, **recipe(trial, max_epochs, kd), 'param_cap': cap}


def pyramidnet_space(trial: optuna.Trial, cap: int = 2_000_000, max_epochs: float = 40, kd: str = '') -> dict:
    """PyramidNet (M2: depth 110, alpha 48, 1.77 M, 95.18 % val untuned): blocks a stage (depth 6n + 2, or 9n + 2
    with bottlenecks), the widening alpha."""
    c = trial.suggest_categorical
    bottleneck, n = c('bottleneck', [False, True]), trial.suggest_int('pn_n', 3, 21)
    p = {
        'arch': 'pyramidnet',
        'arch_kw': {'depth': (9 if bottleneck else 6) * n + 2, 'bottleneck': bottleneck, 'alpha': c('alpha', [24, 32, 48, 64, 84, 100, 128, 160])},
    }
    return {**p, **recipe(trial, max_epochs, kd), 'param_cap': cap}


def _eq_recipe(trial: optuna.Trial, max_epochs: float) -> dict:
    """The recipe around the flip-equivariant finalist's at a fixed ``max_epochs`` (session 2026-10-30, M5), scored at
    one view (``tta_level`` 0; ``val_acc_by_views`` still reports 2 and 6)."""
    c, s = trial.suggest_categorical, trial.suggest_float
    return {
        'epochs': float(max_epochs),
        'batch_size': c('batch_size', [512, 1024]),
        'lr': s('lr', 4.0, 14.0, log=True),
        'momentum': s('momentum', 0.7, 0.9),
        'weight_decay': s('weight_decay', 0.004, 0.015, log=True),
        'bias_scaler': s('bias_scaler', 24.0, 96.0, log=True),
        'label_smoothing': s('label_smoothing', 0.1, 0.3),
        'warmup_frac': s('warmup_frac', 0.05, 0.25),
        'final_lr_frac': s('final_lr_frac', 0.0, 0.08),
        'bn_momentum': s('bn_momentum', 0.035, 0.12, log=True),
        'translate': c('translate', [2, 3, 4]),
        'cutout': c('cutout', [8, 12, 16]),
        'whiten_bias_epochs': 3,
        'scaling_factor': s('scaling_factor', 1 / 25, 1 / 8, log=True),
        'tta_level': 0,
        'lookahead': True,
    }


def _widths(trial: optuna.Trial, lows: tuple[int, int, int]) -> tuple[int, int, int]:
    c = trial.suggest_categorical
    return (c('w1', list(range(lows[0], 184, 8))), c('w2', list(range(lows[1], 200, 8))), c('w3', list(range(lows[2], 232, 8))))


def eqairbench_space(
    trial: optuna.Trial, cap: int = 2_000_000, max_epochs: float = 100, kd: str = '', lows: tuple[int, int, int] = (96, 96, 112)
) -> dict:
    """Session 2026-10-30-equivariant-small, M5: the flip-equivariant airbench (depth 3; the last 1 or 2 groups
    equivariant, M3's winner), structured receptive fields on or off (M4's one non-negative lever), widths under the
    cap, and the recipe around the finalist's (``_eq_recipe``). Distillation from ``kd`` (one airbench96) with its
    weight free."""
    c, s = trial.suggest_categorical, trial.suggest_float
    p = {'arch': 'airbench', 'depth': 3, 'flip_eq': c('flip_eq', [1, 2]), 'srf': c('srf', [False, True]), 'widths': _widths(trial, lows)}
    p.update(_eq_recipe(trial, max_epochs), augment='none', param_cap=cap)
    if kd:
        p.update(kd_teacher=kd, kd_weight=s('kd_weight', 0.0, 0.4), kd_temp=s('kd_temp', 2.0, 8.0, log=True))
    return p


def eqaug_space(
    trial: optuna.Trial, cap: int = 2_000_000, max_epochs: float = 40, kd: str = '', lows: tuple[int, int, int] = (96, 96, 112), m_max: int = 15
) -> dict:
    """Session 2026-11-11-idle-search: ``eqairbench_space`` with what earlier sessions settled fixed and what they never
    measured opened. Flip equivariance in the last two groups (1 never led); policy augmentation (TrivialAugment wide,
    RandAugment), off in every earlier eqairbench study, while one-view accuracy saturated past 80 epochs, a sign of
    regularisation; distillation's weight down to none. A fixed, short ``max_epochs`` (40) for idle-time trials; a
    point over ``cap`` is refused before it runs (``refuse_over_cap``). ``lows`` and ``m_max``: the narrowest widths and
    RandAugment's strongest magnitude (of 30), wider for ``frontaug_space``."""
    c, s = trial.suggest_categorical, trial.suggest_float
    p = {'arch': 'airbench', 'depth': 3, 'flip_eq': 2, 'srf': c('srf', [False, True]), 'widths': _widths(trial, lows)}
    p.update(_eq_recipe(trial, max_epochs), param_cap=cap)
    p['augment'] = c('augment', ['none', 'ta_wide', 'randaugment'])
    if p['augment'] == 'randaugment':
        p['augment_m'] = trial.suggest_int('augment_m', 2, m_max)
    if kd:
        p.update(kd_teacher=kd, kd_weight=s('kd_weight', 0.0, 0.4), kd_temp=s('kd_temp', 2.0, 8.0, log=True))
    return refuse_over_cap(trial, p)


def frontaug_space(trial: optuna.Trial, kd: str = '', max_epochs: float = 40) -> dict:
    """The accuracy / MACs front with policy augmentation (session 2026-11-11-idle-search): ``eqaug_space`` with the
    front's widths (down to half the finalist's) and RandAugment's magnitude up to its maximum, 30 (the 24 h eqaug
    study's best 15 trials all sat at 13-15, the top of its range). Every point of the 2026-11-05 front trained without
    policy augmentation."""
    return eqaug_space(trial, max_epochs=max_epochs, kd=kd, lows=FRONT_LOWS, m_max=30)


def refuse_over_cap(trial: optuna.Trial, p: dict) -> dict:
    """``p``, its stored parameters kept on the trial; over its ``param_cap``, refused here (raised as
    ``optuna.TrialPruned``: told pruned with no training, so the sampler still learns the boundary), as the trainer
    would refuse it at step 0 after a job's round trip (phase B, 2026-11-05: 3 of 12 trials)."""
    import optuna

    from .airbench import DEFAULTS, build, param_counts

    stored = param_counts(build({**DEFAULTS, **p}, abstract=True)[0])['stored']
    trial.set_user_attr('params', stored)
    if stored > p['param_cap']:
        raise optuna.TrialPruned(f'{stored} parameters, over the cap of {p["param_cap"]}')
    return p


SPACES = {
    'airbench': airbench_space,
    'resnet9': resnet9_space,
    'convmixer': convmixer_space,
    'pyramidnet': pyramidnet_space,
    'eqairbench': eqairbench_space,
    'eqaug': eqaug_space,
}

# airbench94's recipe (1.97 M, widths 64/256/256) as a point of the airbench space, at the cap's epochs
_AB94 = {
    'depth': 2,
    'w1': 64,
    'w2': 256,
    'w3': 256,
    'epochs': 9.9,
    'batch_size': 1024,
    'lr': 11.5,
    'momentum': 0.85,
    'weight_decay': 0.0153,
    'bias_scaler': 64.0,
    'label_smoothing': 0.2,
    'warmup_frac': 0.23,
    'final_lr_frac': 0.07,
    'bn_momentum': 0.4,
    'translate': 2,
    'cutout': 0,
    'whiten_bias_epochs': 3,
    'scaling_factor': 1 / 9,
    'tta_level': 2,
    'lookahead': True,
    'augment': 'none',
}
# airbench96's recipe at the cap's 40 epochs (M2: depth 3 at 64/192/192 the best screen, 95.48 % val; 64/160/224
# 95.34 %; depth 2 at 64/256/256 94.96 %)
_R96 = {
    **_AB94,
    'epochs': 40.0,
    'cutout': 12,
    'translate': 4,
    'lr': 9.0,
    'weight_decay': 0.012,
    'warmup_frac': 0.1,
    'final_lr_frac': 0.0,
    'bn_momentum': 0.1,
}
WARM = {
    'airbench': [{**_R96, 'depth': 3, 'w2': 192, 'w3': 192}, {**_R96, 'depth': 3, 'w2': 160, 'w3': 224}, _R96],
    'resnet9': [{**{k: v for k, v in _AB94.items() if k not in ('depth', 'w1', 'w2', 'w3')}, 'width': 32, 'activation': 'gelu', 'epochs': 30.0}],
    'convmixer': [
        {
            **{k: v for k, v in _AB94.items() if k not in ('depth', 'w1', 'w2', 'w3')},
            'dim': 256,
            'cm_depth': 8,
            'kernel': 5,
            'patch': 2,
            'epochs': 30.0,
            'augment': 'randaugment',
            'augment_m': 12,
        }
    ],
    'pyramidnet': [
        {**{k: v for k, v in _R96.items() if k not in ('depth', 'w1', 'w2', 'w3')}, 'bottleneck': False, 'pn_n': 18, 'alpha': 48, 'batch_size': 256}
    ],
}
# M5's starting points: the finalist (D1 D1, 128/112/144) and its SRF variant (160/136/176), M4's best, same recipe
_FIN = {
    'batch_size': 1024,
    'lr': 9.3713,
    'momentum': 0.777184,
    'weight_decay': 0.007254,
    'bias_scaler': 57.61766,
    'label_smoothing': 0.167498,
    'warmup_frac': 0.148382,
    'final_lr_frac': 0.018644,
    'bn_momentum': 0.055161,
    'translate': 4,
    'cutout': 12,
    'scaling_factor': 0.053902,
    'kd_weight': 0.119,
    'kd_temp': 5.528057,
}
WARM['eqairbench'] = [
    {**_FIN, 'flip_eq': 2, 'srf': False, 'w1': 128, 'w2': 112, 'w3': 144},
    {**_FIN, 'flip_eq': 2, 'srf': True, 'w1': 160, 'w2': 136, 'w3': 176},
]
# the two best known one-view points, without the flip_eq the space fixes: FIN (D1 D1, 128/112/144) and T39 (the
# front's best, front-20261105-170738: SRF, 168/128/184, 95.73 % two-seed validation at 100 epochs)
WARM['eqaug'] = [
    {**{k: v for k, v in point.items() if k != 'flip_eq'}, 'augment': 'none'}
    for point in (
        WARM['eqairbench'][0],
        {
            'srf': True,
            'w1': 168,
            'w2': 128,
            'w3': 184,
            'batch_size': 512,
            'lr': 11.829422,
            'momentum': 0.802772,
            'weight_decay': 0.004272,
            'bias_scaler': 69.82426,
            'label_smoothing': 0.134185,
            'warmup_frac': 0.086056,
            'final_lr_frac': 0.03583,
            'bn_momentum': 0.038894,
            'translate': 3,
            'cutout': 12,
            'scaling_factor': 0.0527,
            'kd_weight': 0.233606,
            'kd_temp': 2.59562,
        },
    )
]
CALIBRATE = {
    'airbench': {'epochs': 2, 'widths': (64, 256, 256)},
    'resnet9': {'epochs': 2, 'arch': 'resnet9', 'arch_kw': {'width': 32}},
    'convmixer': {'epochs': 1, 'arch': 'convmixer', 'arch_kw': {'dim': 256, 'depth': 8, 'kernel': 5, 'patch': 2}},
    'pyramidnet': {'epochs': 1, 'arch': 'pyramidnet', 'arch_kw': {'depth': 110, 'alpha': 48}, 'batch_size': 256},
    'eqairbench': {'epochs': 2, 'widths': (128, 112, 144), 'depth': 3, 'flip_eq': 2},
}
CALIBRATE['eqaug'] = CALIBRATE['eqairbench']


# ---- the accuracy / compute front (2026-11-05): the two-objective study session 2026-10-30 planned (M7) and drew
# from existing points instead. Objectives: two-seed one-view validation (up) and MACs a view (down), the
# flip-equivariant airbench at 100 epochs, widths down to half the finalist's so the front's cheap end is searched.

FRONT_LOWS = (48, 48, 64)
TRAIN_FRONT = 'cifarq.families:train_front'
TEACHER_DIR = 'models/ab96-45k-s0'  # under results/: airbench96 on the 45k split, seed 0
# the teacher's recipe: E1.1-45k (results/autoresearch/E1.1-45k), 95.87 % at 6 views; its saved network (job 00023)
# was cleaned away with its job folder, so it is trained again from this record
AB96_45K = {
    'widths': [128, 384, 512],
    'depth': 3,
    'scaling_factor': 1 / 9,
    'bn_momentum': 0.1,
    'whiten_kernel': 2,
    'whiten_eps': 0.0005,
    'epochs': 37.0,
    'batch_size': 1024,
    'lr': 9.0,
    'momentum': 0.85,
    'weight_decay': 0.012,
    'bias_scaler': 64.0,
    'label_smoothing': 0.2,
    'whiten_bias_epochs': 3,
    'warmup_frac': 0.1,
    'final_lr_frac': 0.0,
    'lookahead': True,
    'flip': True,
    'translate': 4,
    'cutout': 12,
    'tta_level': 2,
    'arch': 'airbench',
    'arch_kw': {},
    'bn_scale': False,
    'param_cap': 0,
    'tie': 0,
    'flip_eq': 0,
    'pca_init': False,
    'tails': 0,
    'tail_epochs': 3,
}


def front_space(trial: optuna.Trial, kd: str = '') -> dict:
    """``eqairbench_space`` with the front's widths, a point over the parameter cap refused before it runs."""
    return refuse_over_cap(trial, eqairbench_space(trial, max_epochs=100, kd=kd, lows=FRONT_LOWS))


def macs_of(params: dict) -> int:
    """MACs a view of the network ``params`` describes (traced from its shapes, nothing allocated or run)."""
    from .airbench import DEFAULTS, build
    from .footprint import count_macs

    net, _ = build({**DEFAULTS, **params}, abstract=True)
    return int(count_macs(net))


def train_front(data: str | None = None, report: Any = None, **hp) -> dict:
    """A front trial: ``train_two_seeds`` plus its MACs a view (the second objective, exact, not trained for)."""
    r = train_two_seeds(data, report=report, **hp)
    return {**r, 'macs': macs_of(hp)}


def front_seeds(results: str | Path) -> list[dict]:
    """Completed trials for the front study from the session's saved runs (its M5 study was cleaned away; these
    survive with full parameters and per-seed scores): M6's confirmations (seeds 0-1, as ``train_two_seeds``) and,
    where a configuration has none, its retraining (seeds 1-3). Each: {params in ``front_space``'s names, values
    [one-view validation, MACs], source}."""
    import json
    from pathlib import Path

    out, seen = [], set()
    for sub in ('M6-confirm', 'M6-retrain'):
        for f in sorted((Path(results) / 'autoresearch' / sub).glob('*.json')):
            d = json.loads(f.read_text())
            name, p = f.stem, d['params']
            if name in seen or float(p.get('epochs', 0)) != 100.0:
                continue
            seen.add(name)
            w1, w2, w3 = p['widths']
            params = {
                'flip_eq': int(p.get('flip_eq', 2)),
                'srf': bool(p.get('srf', False)),
                'w1': w1,
                'w2': w2,
                'w3': w3,
                **{
                    k: p[k]
                    for k in (
                        'batch_size',
                        'lr',
                        'momentum',
                        'weight_decay',
                        'bias_scaler',
                        'label_smoothing',
                        'warmup_frac',
                        'final_lr_frac',
                        'bn_momentum',
                        'translate',
                        'cutout',
                        'scaling_factor',
                        'kd_weight',
                        'kd_temp',
                    )
                },
            }
            vals = [r['val_acc_notta'] for r in d['runs']]
            out.append(
                {'params': params, 'values': [sum(vals) / len(vals), macs_of(p)], 'source': f'{sub}/{name}', 'seeds': [r['seed'] for r in d['runs']]}
            )
    return out
