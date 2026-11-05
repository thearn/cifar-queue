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
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import optuna

from common import RESULTS, dataset_file, executor, record, stage_targets, targets_arg
from cifarq import families
from runq.hpo import optuna_optimize

STUDY = 'eqairbench-front'


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
    record('front-teacher', {'model': str(out), 'job': str(job), 'val_acc': res.get('val_acc'),
                             'val_acc_by_views': res.get('val_acc_by_views'), 'train_seconds': res.get('train_seconds')})
    print(f'teacher: {out} (val {res.get("val_acc")})', flush=True)
    return out


def study_for(storage: str, seed: int, resume: bool) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=seed, n_startup_trials=4)
    if resume:
        return optuna.load_study(study_name=STUDY, storage=storage, sampler=sampler)
    study = optuna.create_study(study_name=STUDY, storage=storage, directions=['maximize', 'minimize'], sampler=sampler)
    probe = optuna.create_study(directions=['maximize', 'minimize']).ask()      # the space's distributions
    families.front_space(probe, kd='teacher')
    for s in families.front_seeds(RESULTS):
        dist = {k: probe.distributions[k] for k in s['params']}
        study.add_trial(optuna.trial.create_trial(params=s['params'], distributions=dist, values=s['values'],
                                                  user_attrs={'source': s['source'], 'seeds': s['seeds']}))
    fin = next(s for s in families.front_seeds(RESULTS) if s['source'].endswith('FIN'))['params']
    snap = lambda w, lo: max(lo, int(round(w / 8)) * 8)
    for scale in (0.75, 0.5):                                 # the cheap end, unmeasured: the finalist made narrower
        study.enqueue_trial({**fin, 'w1': snap(fin['w1'] * scale, 48), 'w2': snap(fin['w2'] * scale, 48),
                             'w3': snap(fin['w3'] * scale, 64)})
    return study


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(line_buffering=True)             # (its log is followed while it runs: a line a trial)
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=16)
    targets_arg(ap)
    ap.add_argument('--storage', type=Path, default=RESULTS / 'front-study.db')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--minutes', type=float, default=None, help='no new trial after this long')
    ap.add_argument('--teacher-only', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    targets = stage_targets('front', a.targets)
    storage = f'sqlite:///{a.storage.resolve()}'
    if a.dry_run:
        seeds = families.front_seeds(RESULTS)
        print(json.dumps({'targets': targets, 'storage': storage, 'teacher': str(RESULTS / families.TEACHER_DIR),
                          'teacher_exists': (RESULTS / families.TEACHER_DIR / 'model.safetensors').exists(),
                          'seeds': [(s['source'], round(s['values'][0], 4), round(s['values'][1] / 1e6)) for s in seeds],
                          'trials': a.trials}, indent=1))
        raise SystemExit(0)
    kd = teacher(targets)
    if a.teacher_only:
        raise SystemExit(0)
    study = study_for(storage, a.seed, a.resume)
    print(f'{STUDY}: {len(study.trials)} trials so far; new ones on {targets}', flush=True)
    t0 = time.time()
    with executor('front', targets, data=[dataset_file(), kd]) as ex:
        optuna_optimize(study, families.TRAIN_FRONT, lambda t: families.front_space(t, kd=str(kd)), n_trials=a.trials,
                        executor=ex, objective=['val_acc_notta', 'macs'], fixed={'data': str(dataset_file())},
                        resume=True, minutes=a.minutes)
    front = sorted(study.best_trials, key=lambda t: t.values[1])
    out = {'study': STUDY, 'storage': storage, 'trials': len(study.trials), 'search_seconds': round(time.time() - t0, 1),
           'front': [{'trial': t.number, 'val_1view': t.values[0], 'macs': t.values[1], 'params': t.params,
                      'source': t.user_attrs.get('source', 'new'), 'device': t.user_attrs.get('device')} for t in front]}
    for f in out['front']:
        print(f'  {f["macs"] / 1e6:6.0f} M MACs  {f["val_1view"]:.4f}  trial {f["trial"]} ({f["source"]}, {f["device"]})')
    print('saved', record('front', out))
