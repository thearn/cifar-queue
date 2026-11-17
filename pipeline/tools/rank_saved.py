"""Measure every saved model on this machine under one protocol and rank them by both scores.

Every concept any campaign saved a checkpoint for is scored on the same evidence: one-view validation accuracy from
the checkpoint's own weights on the pinned dataset, the parameters and bytes actually stored, MACs traced from the
network, and latency timed here with those weights. Nothing is retrained and the test split is never read, so this
answers "which saved model is the champion" from measurements rather than from numbers copied between reports.

Latency is measured once per inference graph: seeds of one arm share a graph, so they share its timing, while their
accuracies stay separate. The graph is identified by the saved configuration with the training-only keys removed, so
an unrecognised key splits a group rather than merging two graphs by accident.

    python pipeline/tools/rank_saved.py                              # measure, write docs/metrics/measured.json
    python pipeline/tools/rank_saved.py --limit 8 --batches 256      # a short pass over eight models
    python pipeline/tools/rank_saved.py --report                     # rank the saved file again, no GPU work
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from cifarq import efficiency, netscore  # noqa: E402

DATASET = ROOT / 'data' / 'dataset' / 'dataset-56cff0f67835.npz'
MEASURED = ROOT / 'docs' / 'metrics' / 'measured.json'
BATCHES = (256, 1)
WARMUP, REPEATS = 20, 100

# Keys of a saved configuration that do not change the inference graph: how the weights were obtained, not what runs.
TRAINING_ONLY = frozenset(
    {
        'epochs',
        'batch_size',
        'lr',
        'momentum',
        'weight_decay',
        'bias_scaler',
        'label_smoothing',
        'whiten_bias_epochs',
        'warmup_frac',
        'final_lr_frac',
        'lookahead',
        'flip',
        'translate',
        'cutout',
        'bn_momentum',
        'optimizer',
        'beta2',
        'augment',
        'augment_m',
        'train_on',
        'kd_teacher',
        'kd_weight',
        'kd_temp',
        'kd_weighting',
        'precision',
        'seed',
        'muon_lr',
        'muon_momentum',
        'tail_epochs',
        'filter_keep',
        'filter_proxy_widths',
        'filter_proxy_epochs',
        'pca_init',
        'subspace_learn_basis',
        'eval_batch',
        'test',
        'param_cap',
        'whiten_eps',
    }
)


def signature(config: dict) -> str:
    """A short digest of the inference graph: everything in the configuration that is not training-only."""
    graph = {k: v for k, v in sorted(config.items()) if k not in TRAINING_ONLY}
    return hashlib.sha256(json.dumps(graph, sort_keys=True, default=str).encode()).hexdigest()[:12]


def discover(root: Path) -> list[dict]:
    """Every saved model directory under ``root``, with the campaign, arm and seed of the attempt that wrote it."""
    found = []
    for cfg in sorted(root.rglob('model/config.json')):
        directory = cfg.parent
        config = json.loads(cfg.read_text())
        attempt = directory.parent / 'attempt.json'
        meta = json.loads(attempt.read_text()) if attempt.exists() else {}
        rel = directory.relative_to(root)
        found.append(
            {
                'id': str(rel.parent).replace('/', '::'),
                'path': str(directory),
                'campaign': rel.parts[0],
                'stage': meta.get('stage') or (rel.parts[1] if len(rel.parts) > 2 else ''),
                'arm': meta.get('arm', ''),
                'seed': meta.get('seed'),
                'signature': signature(config),
                'config': config,
                'reported': {k: meta.get('result', {}).get(k) for k in ('val_acc', 'val_acc_notta', 'views_per_image')},
            }
        )
    return found


def accuracies(models: list[dict], images, labels, chunk: int = 500) -> None:
    """One-view validation accuracy of each model's own weights, in float32 and in the T4's float16, added in place.

    Images go through in chunks: the widest networks here do not hold 5000 activations at once on a 16 GB card, and
    the chunk changes nothing but the memory, since every image is classified on its own.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from cifarq.airbench import Predictor

    graph = None
    for done, model in enumerate(sorted(models, key=lambda m: m['signature']), 1):  # one compilation per graph, reused by its seeds
        if model['signature'] != graph:
            jax.clear_caches()
            graph = model['signature']
        model['accuracy'] = {}
        for name, dtype in (('fp32', jnp.float32), ('fp16', jnp.float16)):
            predictor = Predictor(model['path'], 'cuda', dtype, tta_level=0)
            predicted = np.concatenate([np.asarray(predictor.predict(images[s : s + chunk])).argmax(1) for s in range(0, len(images), chunk)])
            model['accuracy'][name] = float((predicted == labels).mean())
            del predictor
        print(f'  [{done}/{len(models)}] {model["id"]}  fp32 {model["accuracy"]["fp32"]:.4f}  fp16 {model["accuracy"]["fp16"]:.4f}', flush=True)


def static(directory: str) -> dict:
    """Stored parameters, deployed bytes and traced MACs a view; views come from the saved configuration."""
    from cifarq.footprint import static as footprint_static

    out = footprint_static(directory)
    return {k: out[k] for k in ('params_stored', 'params_trainable', 'bytes', 'macs_per_view', 'views', 'macs_per_image')}


def bench(directory: str, batches: tuple[int, ...]) -> dict:
    """Median milliseconds of one device-resident forward pass at the checkpoint's trained weights, float16.

    The input is already on the device and the result is waited for, so this is the network's own time: it excludes
    the host transfer and whatever a server does around it, exactly as the campaign's gate benchmark did.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from cifarq import jx
    from cifarq.airbench import Predictor

    device = jx.pick('gpu')
    predictor = Predictor(directory, 'cuda', jnp.float16, tta_level=0)
    out = {'device': device.device_kind, 'precision': 'float16', 'warmup': WARMUP, 'repeats': REPEATS}
    rng = np.random.default_rng(0)
    for batch in batches:
        images = jax.device_put(jnp.asarray(rng.integers(0, 256, (batch, 32, 32, 3), dtype=np.uint8)), device)
        call = lambda: jax.block_until_ready(predictor._fwd(predictor.params, predictor.rest, images))  # noqa: B023
        for _ in range(WARMUP):
            call()
        times = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            call()
            times.append((time.perf_counter() - start) * 1e3)
        out[f'ms_batch{batch}'] = round(statistics.median(times), 4)
        out[f'ms_batch{batch}_p90'] = round(statistics.quantiles(times, n=10)[8], 4)
    return out


def bench_fresh(directory: str, batches: tuple[int, ...]) -> dict:
    """``bench`` in its own interpreter, so one graph's compilation and allocations cannot time the next one's."""
    cmd = [sys.executable, str(Path(__file__).resolve()), '--bench', directory, '--batches', ','.join(str(b) for b in batches)]
    run = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(run.stdout.strip().splitlines()[-1])


def reference_of(rows: list[dict]) -> str:
    """The cohort's reference: the campaign's own ``baseline`` arm where it was saved, else the smallest network.

    Only the scale of the scores depends on this, never their order, since each score is a ratio against it.
    """
    baselines = [row for row in rows if row['arm'] == 'baseline']
    return min(baselines or rows, key=lambda row: (row['params'], row['id']))['id']


def arm_rows(rows: list[dict]) -> list[dict]:
    """One row per arm and graph, its accuracy averaged over the seeds saved for it.

    A single checkpoint carries the run-to-run noise of its training, which is larger than most differences here, so
    this is the ranking to read; seeds of one arm share a graph and therefore its cost measurements exactly.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row['campaign'], row['arm'] or row['stage'], row['graph']), []).append(row)
    out = []
    for (campaign, arm, graph), group in groups.items():
        accuracies = [row['accuracy'] for row in group]
        first = group[0]
        out.append(
            {
                **{k: first[k] for k in ('latency', 'size', 'params', 'macs', 'accuracy_policy', 'latency_policy')},
                'id': f'{campaign}/{arm}/{graph}',
                'campaign': campaign,
                'arm': arm,
                'graph': graph,
                'seeds': len(group),
                'accuracy': round(sum(accuracies) / len(accuracies), 6),
                'accuracy_min': min(accuracies),
                'accuracy_max': max(accuracies),
                'accuracy_sd': round(statistics.stdev(accuracies), 6) if len(accuracies) > 1 else None,
            }
        )
    return out


def checkpoint_rows(models: list[dict], batch: int) -> list[dict]:
    """One row per saved checkpoint at one serving batch."""
    rows = []
    for model in models:
        rows.append(
            {
                'id': model['id'],
                'campaign': model['campaign'],
                'stage': model['stage'],
                'arm': model['arm'],
                'seed': model['seed'],
                'graph': model['signature'],
                'accuracy': round(model['accuracy']['fp32'], 6),
                'accuracy_fp16': round(model['accuracy']['fp16'], 6),
                'latency': model['latency'][f'ms_batch{batch}'],
                'size': model['static']['bytes'],
                'params': model['static']['params_stored'],
                'macs': model['static']['macs_per_image'],
                'accuracy_policy': 'one-view',
                'latency_policy': 'one-view',
                'reported_val_acc': model['reported'].get('val_acc'),
            }
        )
    return rows


def cohort(rows: list[dict], batch: int, scope: str, hardware: str) -> dict:
    """One scoring cohort: the rows of one scope at one serving batch, with the protocol they were measured under."""
    return {
        'id': f'measured-{scope}-one-view-b{batch}',
        'scope': scope,
        'reference': reference_of(rows),
        'protocol': {
            'dataset': '56cff0f67835b54960f47a6a6bd8d47e27e5d76791c2346206bf48ce1d0509f9',
            'split': 'validation',
            'hardware': hardware,
            'precision': 'float32 accuracy, float16 latency',
            'batch': batch,
            'timing_scope': 'device-resident one-view forward with normalisation and softmax, trained weights; '
            f'warmed, median of {REPEATS} passes after {WARMUP}',
            'latency_unit': 'ms/batch',
            'size_metric': 'stored_bytes',
            'accuracy_aggregation': 'one checkpoint, 5000 validation images' if scope == 'checkpoint' else 'mean over the seeds saved for the arm',
            'limitations': 'Latency is the network forward only, not end-to-end serving. One timed process per graph, '
            'so repeated-process spread is not measured. Seeds of one arm share their graph timing. Accuracy is one '
            'checkpoint, so differences below the run-to-run floor of about 0.2 points are not resolved.',
        },
        'sources': ['results/autoresearch/**/attempt.json', 'results/autoresearch/**/model/'],
        'rows': rows,
    }


def document(measurements: list[dict], batches: tuple[int, ...], hardware: str) -> dict:
    """The measurement file: every checkpoint, and every arm over its seeds, scored at each serving batch."""
    cohorts = []
    for batch in batches:
        rows = checkpoint_rows(measurements, batch)
        cohorts.append(cohort(rows, batch, 'checkpoint', hardware))
        cohorts.append(cohort(arm_rows(rows), batch, 'arm', hardware))
    return {
        'assessment_date': time.strftime('%Y-%m-%d'),
        'measured_on': hardware,
        'note': 'Measured from saved checkpoints by pipeline/tools/rank_saved.py; no retraining, no test split.',
        'cohorts': cohorts,
    }


def measure(limit: int | None, batches: tuple[int, ...]) -> dict:
    import numpy as np

    models = discover(ROOT / 'results' / 'autoresearch')
    if limit:
        models = models[:limit]
    print(f'{len(models)} saved models, {len({m["signature"] for m in models})} distinct inference graphs', flush=True)

    data = np.load(DATASET)
    images, labels = data['x_val'], data['y_val'].astype(np.int64)
    print(f'validation {images.shape[0]} images from {DATASET.name}', flush=True)
    accuracies(models, images, labels)

    graphs: dict[str, dict] = {}
    for model in models:
        model['static'] = static(model['path'])
    for i, (sig, first) in enumerate({m['signature']: m for m in reversed(models)}.items(), 1):
        print(f'  bench [{i}] graph {sig} ({first["arm"] or first["id"]})', flush=True)
        graphs[sig] = {**bench_fresh(first['path'], batches), 'timed_model': first['id']}
    for model in models:
        model['latency'] = graphs[model['signature']]

    return document(models, batches, graphs[models[0]['signature']]['device'])


def remeasured(saved: dict, batches: tuple[int, ...]) -> list[dict]:
    """The per-checkpoint measurements held in a saved file, so its cohorts can be built again without a GPU."""
    by_id: dict[str, dict] = {}
    for cohort_ in saved['cohorts']:
        if cohort_.get('scope', 'checkpoint') != 'checkpoint':
            continue
        batch = cohort_['protocol']['batch']
        for row in cohort_['rows']:
            model = by_id.setdefault(
                row['id'],
                {
                    'id': row['id'],
                    'campaign': row['campaign'],
                    'stage': row.get('stage', ''),
                    'arm': row['arm'],
                    'seed': row['seed'],
                    'signature': row['graph'],
                    'accuracy': {'fp32': row['accuracy'], 'fp16': row['accuracy_fp16']},
                    'static': {'bytes': row['size'], 'params_stored': row['params'], 'macs_per_image': row['macs']},
                    'latency': {'device': cohort_['protocol']['hardware']},
                    'reported': {'val_acc': row.get('reported_val_acc')},
                },
            )
            model['latency'][f'ms_batch{batch}'] = row['latency']
    missing = [b for b in batches if any(f'ms_batch{b}' not in m['latency'] for m in by_id.values())]
    if missing:
        raise ValueError(f'the saved file has no timing at batch {missing}; measure again')
    return list(by_id.values())


def report(document: dict, top: int) -> None:
    """The two rankings of each cohort side by side: the error-cost score and NetScore."""
    scored = efficiency.assess(document)
    netscored = netscore.assess(document)

    def label(row: dict) -> str:
        """A readable name: an arm and its seed, or an arm and how many seeds it averages."""
        tail = f'-s{row["seed"]}' if row.get('seed') is not None else f' x{row.get("seeds", 1)}'
        return f'{row["campaign"][:20]}/{row["arm"] or row.get("stage") or "?"}{tail}'[:46]

    for cost_cohort, net_cohort in zip(scored['cohorts'], netscored['cohorts'], strict=True):
        print(f'\n== {cost_cohort["id"]}  ({len(cost_cohort["rows"])} rows)  reference {cost_cohort["reference"]}')
        print(f'{"model":46} {"val":>7} {"params":>9} {"MB":>6} {"ms":>7} {"score":>7} {"NetScore":>9}')
        order = {row['id']: row['netscore'] for row in net_cohort['rows']}
        for row in cost_cohort['rows'][:top]:
            print(
                f'{label(row):46} {100 * row["accuracy"]:6.2f}% {row["params"]:9,} {row["size"] / 2**20:6.1f} '
                f'{row["latency"]:7.3f} {row["score"]:7.4f} {order[row["id"]]:9.2f}'
            )
        best = net_cohort['rows'][0]
        print(f'  NetScore best: {label(best)}  {best["netscore"]:.2f} dB  ({100 * best["accuracy"]:.2f} %, {best["params"]:,} params)')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', help='time one model directory and print the result (used for its own fresh process)')
    ap.add_argument('--batches', default=','.join(str(b) for b in BATCHES))
    ap.add_argument('--limit', type=int, help='measure only the first N saved models')
    ap.add_argument('--report', action='store_true', help='rank the saved measurements again without measuring')
    ap.add_argument('--rebuild', action='store_true', help='with --report, build the cohorts again from the saved per-checkpoint rows')
    ap.add_argument('--top', type=int, default=15)
    ap.add_argument('--out', type=Path, default=MEASURED)
    args = ap.parse_args()
    batches = tuple(int(b) for b in args.batches.split(','))

    if args.bench:
        print(json.dumps(bench(args.bench, batches)))
        return
    if args.report:
        saved = json.loads(args.out.read_text())
        report(document(remeasured(saved, batches), batches, saved['measured_on']) if args.rebuild else saved, args.top)
        return
    document_ = measure(args.limit, batches)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document_, indent=1) + '\n', encoding='utf-8')
    print(f'\nwrote {args.out}')
    report(document_, args.top)


if __name__ == '__main__':
    main()
