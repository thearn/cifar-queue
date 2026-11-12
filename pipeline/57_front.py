"""The accuracy / compute front of the flip-equivariant airbench (2026-11-05): the two-objective study session
2026-10-30 planned (M7) and drew from existing points instead. Two objectives: two-seed one-view validation accuracy
(up) and MACs a view (down, exact from the configuration), 100 epochs, distillation from airbench96 as in M5, widths
down to half the finalist's so the cheap end of the front is searched.

The study starts where that session ended: its saved runs (FIN, T39, T14, T33, SRF4: ``families.front_seeds``) enter
as completed trials, so the sampler models the known front from the first trial and no compute repeats them; the
first new trials are the finalist scaled to 3/4 and 1/2 of its widths (the cheap end, unmeasured), then multivariate
TPE. The teacher (airbench96 on the 45k split, seed 0) is trained once from its record if results/models lacks it.
Trials are jobs on the stage's targets (configs/compute.toml [stages.front]: Colab G4, RunPod via SkyPilot, a GCE
spot L4), so the search is also a measure of those providers (runq hours, runq live).

    python pipeline/57_front.py --teacher-only                       # the teacher, once
    python pipeline/57_front.py --trials 16                          # the study (results/front-study.db)
    python pipeline/57_front.py --trials 8 --resume                  # more trials, after a driver died or to extend
    python pipeline/57_front.py --dry-run                            # what it would do

``--space aug`` (2026-11-12): the front again with policy augmentation (``families.frontaug_space``: RandAugment up
to magnitude 30, 40 epochs) in results/front-aug-study.db, on configs/compute.toml [stages.front_aug] (the local GPU).
It starts from an eqaug study's completed trials (``--seed-from``: same epochs, same two-seed one-view score; their
MACs computed from the configuration), then that study's best made narrower (3/4 and 1/2 of its widths), then TPE.
The 2026-11-05 points (100 epochs, no policy augmentation) do not enter it: their values are on another scale.

    python pipeline/57_front.py --space aug --seed-from ~/.cache/runq/jobs/cifar/tune_on_vm/00003-on_vm/study.db --trials 40
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import optuna
from common import RESULTS, dataset_file, executor, read_study, record, stage_targets, targets_arg
from runq.search.hpo import optuna_optimize

from cifarq import families

# per --space: the study's name, its default database, its stage (configs/compute.toml) and its space
SPACES = {
    'plain': ('eqairbench-front', 'front-study.db', 'front', families.front_space),
    'aug': ('eqaug-front', 'front-aug-study.db', 'front_aug', families.frontaug_space),
}


def teacher(targets: dict) -> Path:
    """results/models/ab96-45k-s0: trained once (one job, ~10 min on a G4) from families.AB96_45K."""
    out = RESULTS / families.TEACHER_DIR
    if (out / 'model.safetensors').exists():
        return out
    one = dict(list(targets.items())[:1])
    print(f'training the teacher (airbench96, 45k split, seed 0) on {next(iter(one))}', flush=True)
    with executor('front', one, data=[dataset_file()], minutes=30) as ex:
        r = ex.submit(families.TRAIN, data=str(dataset_file()), save='model', seed=0, **families.AB96_45K)
        res, job = r.result(), Path(r.job)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(job / res['saved'], out)
    record(
        'front-teacher',
        {
            'model': str(out),
            'job': str(job),
            'val_acc': res.get('val_acc'),
            'val_acc_by_views': res.get('val_acc_by_views'),
            'train_seconds': res.get('train_seconds'),
        },
    )
    print(f'teacher: {out} (val {res.get("val_acc")})', flush=True)
    return out


def _snap(point: dict, scale: float) -> dict:
    """``point`` with its widths scaled (snapped to the space's steps of 8, not under the front's narrowest)."""
    snap = lambda w, lo: max(lo, int(round(w * scale / 8)) * 8)
    lo = families.FRONT_LOWS
    return {**point, 'w1': snap(point['w1'], lo[0]), 'w2': snap(point['w2'], lo[1]), 'w3': snap(point['w3'], lo[2])}


def _known(space: str, seed_from: Path | None) -> list[dict]:
    """Completed trials the study starts from, each {params, values, source}: the 2026-11-05 runs for ``plain``; an
    eqaug study's completed trials for ``aug``, their MACs from the configuration."""
    if space == 'plain':
        return [{'params': s['params'], 'values': s['values'], 'source': s['source'], 'seeds': s['seeds']} for s in families.front_seeds(RESULTS)]
    if seed_from is None:
        return []
    old, tmp = read_study(seed_from)
    try:
        done = [t for t in old.trials if t.state.name == 'COMPLETE']
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    hp = lambda t: families.frontaug_space(optuna.trial.FixedTrial(t.params), kd='teacher')
    return [{'params': t.params, 'values': [t.value, families.macs_of(hp(t))], 'source': f'{seed_from.parent.name} trial {t.number}'} for t in done]


def study_for(storage: str, seed: int, resume: bool, space: str = 'plain', seed_from: Path | None = None) -> optuna.Study:
    name, _, _, space_fn = SPACES[space]
    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=seed, n_startup_trials=4)
    if resume:
        return optuna.load_study(study_name=name, storage=storage, sampler=sampler)
    study = optuna.create_study(study_name=name, storage=storage, directions=['maximize', 'minimize'], sampler=sampler)
    probe = optuna.create_study(directions=['maximize', 'minimize']).ask()  # the space's distributions
    space_fn(probe, kd='teacher')
    known = _known(space, seed_from)
    for s in known:
        fixed = optuna.trial.FixedTrial(s['params'])
        space_fn(fixed, kd='teacher')  # (the distributions this point's branch of the space asks for)
        dist = {k: fixed.distributions[k] for k in s['params']}
        attrs = {'source': s['source'], **({'seeds': s['seeds']} if 'seeds' in s else {})}
        study.add_trial(optuna.trial.create_trial(params=s['params'], distributions=dist, values=s['values'], user_attrs=attrs))
    start = (
        next(s for s in known if s['source'].endswith('FIN'))['params']
        if space == 'plain'
        else max(known, key=lambda s: s['values'][0])['params']
        if known
        else families.WARM['eqaug'][0]
    )
    for scale in (0.75, 0.5):  # the cheap end, unmeasured: the best known made narrower
        study.enqueue_trial(_snap(start, scale))
    return study


if __name__ == '__main__':
    import sys

    sys.stdout.reconfigure(line_buffering=True)  # (its log is followed while it runs: a line a trial)
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=16)
    targets_arg(ap)
    ap.add_argument('--space', choices=sorted(SPACES), default='plain', help='plain: the 2026-11-05 front; aug: with policy augmentation')
    ap.add_argument('--storage', type=Path, default=None, help="the study database (default: the space's, under results/)")
    ap.add_argument('--seed-from', type=Path, help='--space aug: an eqaug study database whose completed trials start it')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--minutes', type=float, default=None, help='no new trial after this long')
    ap.add_argument('--teacher-only', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    name, db_name, stage, space_fn = SPACES[a.space]
    targets = stage_targets(stage, a.targets)
    storage = f'sqlite:///{(a.storage or RESULTS / db_name).resolve()}'
    seed_from = a.seed_from.expanduser() if a.seed_from else None
    if a.dry_run:
        seeds = _known(a.space, seed_from)
        print(
            json.dumps(
                {
                    'targets': targets,
                    'storage': storage,
                    'teacher': str(RESULTS / families.TEACHER_DIR),
                    'teacher_exists': (RESULTS / families.TEACHER_DIR / 'model.safetensors').exists(),
                    'seeds': [(s['source'], round(s['values'][0], 4), round(s['values'][1] / 1e6)) for s in seeds],
                    'trials': a.trials,
                },
                indent=1,
            )
        )
        raise SystemExit(0)
    kd = teacher(targets)
    if a.teacher_only:
        raise SystemExit(0)
    study = study_for(storage, a.seed, a.resume, a.space, seed_from)
    print(f'{name}: {len(study.trials)} trials so far; new ones on {targets}', flush=True)
    t0 = time.time()
    with executor(stage, targets, data=[dataset_file(), kd]) as ex:
        optuna_optimize(
            study,
            families.TRAIN_FRONT,
            lambda t: space_fn(t, kd=str(kd)),
            n_trials=a.trials,
            executor=ex,
            objective=['val_acc_notta', 'macs'],
            fixed={'data': str(dataset_file())},
            resume=True,
            minutes=a.minutes,
        )
    front = sorted(study.best_trials, key=lambda t: t.values[1])
    out = {
        'study': name,
        'storage': storage,
        'trials': len(study.trials),
        'search_seconds': round(time.time() - t0, 1),
        'front': [
            {
                'trial': t.number,
                'val_1view': t.values[0],
                'macs': t.values[1],
                'params': t.params,
                'source': t.user_attrs.get('source', 'new'),
                'device': t.user_attrs.get('device'),
            }
            for t in front
        ],
    }
    for f in out['front']:
        print(f'  {f["macs"] / 1e6:6.0f} M MACs  {f["val_1view"]:.4f}  trial {f["trial"]} ({f["source"]}, {f["device"]})')
    print('saved', record('front' if a.space == 'plain' else 'front-aug', out))
