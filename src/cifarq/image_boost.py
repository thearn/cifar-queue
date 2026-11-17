"""Image features plus boosted trees, measured as a complete prediction pipeline."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from .paths import digest, sha256
from .pipeline_score import DATASET_SHA256, classes, evaluate, load_development, objective

FEATURE_VERSION = 'rgb-hog-frozen-cnn-v1'
DEFAULTS = {
    'backend': 'xgboost',
    'features': 'hog_color',
    'pca_components': 0,
    'rounds': 200,
    'depth': 6,
    'leaves': 31,
    'lr': 0.1,
    'min_child_weight': 1.0,
    'min_data_in_leaf': 20,
    'subsample': 0.9,
    'colsample': 0.9,
    'l1': 0.0,
    'l2': 1.0,
    'max_bin': 255,
    'seed': 40,
    'threads': 4,
    'train_device': 'cuda',
}


def settings(config: dict) -> dict:
    if set(config) - DEFAULTS.keys():
        raise ValueError('unknown image boosting options')
    hp = {**DEFAULTS, **config}
    for key, choices in {
        'backend': ('xgboost', 'lightgbm'),
        'features': ('pooled_rgb', 'hog_color', 'learned'),
        'train_device': ('cpu', 'cuda'),
    }.items():
        if hp[key] not in choices:
            raise ValueError(f'invalid {key}')
    for key in ('rounds', 'depth', 'leaves', 'min_data_in_leaf', 'max_bin', 'threads'):
        if not isinstance(hp[key], int) or hp[key] < 1:
            raise ValueError(f'invalid {key}')
    if not isinstance(hp['pca_components'], int) or not 0 <= hp['pca_components'] <= 128:
        raise ValueError('invalid pca_components')
    for key in ('lr', 'subsample', 'colsample'):
        if not math.isfinite(hp[key]) or not 0 < hp[key] <= 1:
            raise ValueError(f'invalid {key}')
    for key in ('l1', 'l2', 'min_child_weight'):
        if not math.isfinite(hp[key]) or hp[key] < 0:
            raise ValueError(f'invalid {key}')
    return hp


def engineered(images: np.ndarray, kind: str) -> np.ndarray:
    """8x8 pooled RGB, or pooled RGB plus spatial color statistics and normalized gradient histograms."""
    x = np.asarray(images)
    if x.dtype != np.uint8 or x.ndim != 4 or x.shape[1:] != (32, 32, 3):
        raise ValueError('expected uint8 NHWC 32x32 RGB images')
    x = x.astype(np.float32) / 255
    pooled = x.reshape(-1, 8, 4, 8, 4, 3).mean(axis=(2, 4)).reshape(len(x), -1)
    if kind == 'pooled_rgb':
        return pooled
    if kind != 'hog_color':
        raise ValueError('unknown engineered features')
    cells = x.reshape(-1, 4, 8, 4, 8, 3)
    color = np.concatenate((cells.mean(axis=(2, 4)).reshape(len(x), -1), cells.std(axis=(2, 4)).reshape(len(x), -1)), axis=1)
    gray = x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gy, gx = np.gradient(gray, axis=(1, 2))
    magnitude = np.hypot(gx, gy)
    bins = np.floor((np.arctan2(gy, gx) % np.pi) * (9 / np.pi)).astype(np.int32) % 9
    yy, xx = np.indices((32, 32))
    cell_ids = yy // 8 * 4 + xx // 8
    indices = ((np.arange(len(x))[:, None, None] * 16 + cell_ids) * 9 + bins).ravel()
    hist = np.bincount(indices, weights=magnitude.ravel(), minlength=len(x) * 16 * 9).reshape(len(x), 4, 4, 9)
    blocks = []
    for y in range(3):
        for z in range(3):
            block = hist[:, y : y + 2, z : z + 2].reshape(len(x), -1)
            block = block / np.sqrt((block * block).sum(1, keepdims=True) + 1e-8)
            block = np.minimum(block, 0.2)
            blocks.append(block / np.sqrt((block * block).sum(1, keepdims=True) + 1e-8))
    return np.concatenate([pooled, color, *blocks], axis=1).astype(np.float32)


class Extractor:
    def __init__(self, kind: str, encoder: Path | None = None):
        self.kind = kind
        if kind == 'learned':
            import torch

            from .resnet18 import build

            if encoder is None:
                raise ValueError('learned features need a serialized encoder')
            self.hp = json.loads((encoder / 'encoder.json').read_text())
            torch.set_num_threads(self.hp['threads'])
            self.net = build(self.hp, pretrained=False)
            self.net.fc = torch.nn.Identity()
            self.net.load_state_dict(torch.load(encoder / 'encoder.pt', map_location='cpu', weights_only=True))
            self.net = self.net.cuda().to(memory_format=torch.channels_last).eval()

    def __call__(self, images):
        if self.kind != 'learned':
            return engineered(images, self.kind)
        import torch

        from .resnet18 import inputs

        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=self.hp['precision'] == 'float16'):
            x = torch.from_numpy(np.ascontiguousarray(images)).cuda()
            return self.net(inputs(x, self.hp)).float().cpu().numpy()


def export_encoder(source: Path, destination: Path) -> None:
    import torch

    hp = json.loads((source / 'config.json').read_text())
    weights = torch.load(source / 'weights.pt', map_location='cpu', weights_only=True)
    torch.save({k: v for k, v in weights.items() if not k.startswith('fc.')}, destination / 'encoder.pt')
    (destination / 'encoder.json').write_text(json.dumps(hp, indent=2) + '\n')


def fit_projection(train: np.ndarray, count: int, seed: int) -> dict:
    if not count:
        return {}
    from sklearn.decomposition import PCA

    if count > min(train.shape):
        raise ValueError('PCA dimensionality exceeds training data')
    pca = PCA(n_components=count, svd_solver='randomized', random_state=seed)
    pca.fit(train)
    return {'mean': pca.mean_.astype(np.float32), 'components': pca.components_.astype(np.float32)}


def project(features: np.ndarray, transform: dict) -> np.ndarray:
    return np.ascontiguousarray((features - transform['mean']) @ transform['components'].T if transform else features, dtype=np.float32)


def artifact_files(directory: Path, hp: dict) -> list[Path]:
    names = ['config.json', 'projection.npz', 'trees.ubj' if hp['backend'] == 'xgboost' else 'trees.txt']
    if hp['features'] == 'learned':
        names += ['encoder.pt', 'encoder.json']
    return [directory / name for name in names]


class Predictor:
    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.hp = settings(json.loads((self.directory / 'config.json').read_text()))
        self.extract = Extractor(self.hp['features'], self.directory)
        with np.load(self.directory / 'projection.npz', allow_pickle=False) as archive:
            self.transform = {key: archive[key] for key in archive.files}
        if self.hp['backend'] == 'xgboost':
            import xgboost as xgb

            self.model = xgb.Booster()
            self.model.load_model(self.directory / 'trees.ubj')
            self.model.set_param({'device': 'cpu', 'nthread': self.hp['threads']})
        else:
            import lightgbm as lgb

            self.model = lgb.Booster(model_file=str(self.directory / 'trees.txt'))

    def predict(self, images: np.ndarray) -> np.ndarray:
        features = project(self.extract(images), self.transform)
        if not np.isfinite(features).all():
            raise ValueError('nonfinite image features')
        probabilities = (
            self.model.inplace_predict(features) if self.hp['backend'] == 'xgboost' else self.model.predict(features, num_threads=self.hp['threads'])
        )
        return classes(probabilities, len(images))


def cached_features(arrays: dict, hp: dict, directory: Path, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    identity = {
        'version': FEATURE_VERSION,
        'dataset_sha256': DATASET_SHA256,
        'features': hp['features'],
        'encoder_sha256': sha256(directory / 'encoder.pt') if hp['features'] == 'learned' else None,
    }
    key = digest(identity)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f'{key}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            return archive['train'], archive['validation']
    extract = Extractor(hp['features'], directory)
    rows = []
    for split in ('train', 'val'):
        images = arrays[f'x_{split}']
        rows.append(np.concatenate([extract(images[i : i + 256]) for i in range(0, len(images), 256)]))
    temporary = cache / f'{key}.tmp.npz'
    np.savez(temporary, train=rows[0], validation=rows[1])
    temporary.replace(path)
    (cache / f'{key}.json').write_text(json.dumps(identity, indent=2) + '\n')
    return rows[0], rows[1]


def train(data: str, output: str, reference: dict, config: dict, encoder: str | None = None, cache: str | None = None) -> dict:
    hp = settings(config)
    started = time.monotonic()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    arrays = load_development(data)
    if hp['features'] == 'learned':
        if encoder is None:
            raise ValueError('missing frozen encoder source')
        export_encoder(Path(encoder), out)
    (out / 'config.json').write_text(json.dumps(hp, indent=2) + '\n')
    x_train, _ = cached_features(arrays, hp, out, Path(cache) if cache else out / 'training-cache')
    transform = fit_projection(x_train, hp['pca_components'], hp['seed'])
    np.savez(out / 'projection.npz', **transform)
    x_train = project(x_train, transform)
    versions = {'numpy': np.__version__}
    if transform:
        import sklearn

        versions['scikit-learn'] = sklearn.__version__
    if hp['features'] == 'learned':
        import torch
        import torchvision

        versions.update(torch=torch.__version__, torchvision=torchvision.__version__)
    if hp['backend'] == 'xgboost':
        import xgboost as xgb

        versions['xgboost'] = xgb.__version__
        model = xgb.train(
            {
                'objective': 'multi:softprob',
                'num_class': 10,
                'tree_method': 'hist',
                'device': hp['train_device'],
                'max_depth': hp['depth'],
                'eta': hp['lr'],
                'min_child_weight': hp['min_child_weight'],
                'subsample': hp['subsample'],
                'colsample_bytree': hp['colsample'],
                'reg_alpha': hp['l1'],
                'reg_lambda': hp['l2'],
                'max_bin': hp['max_bin'],
                'seed': hp['seed'],
                'nthread': hp['threads'],
            },
            xgb.DMatrix(x_train, label=arrays['y_train'], nthread=hp['threads']),
            num_boost_round=hp['rounds'],
        )
        model.save_model(out / 'trees.ubj')
    else:
        import lightgbm as lgb

        versions['lightgbm'] = lgb.__version__
        model = lgb.train(
            {
                'objective': 'multiclass',
                'num_class': 10,
                'verbosity': -1,
                'num_threads': hp['threads'],
                'max_depth': hp['depth'],
                'num_leaves': hp['leaves'],
                'learning_rate': hp['lr'],
                'min_data_in_leaf': hp['min_data_in_leaf'],
                'min_sum_hessian_in_leaf': hp['min_child_weight'],
                'bagging_fraction': hp['subsample'],
                'bagging_freq': 1,
                'feature_fraction': hp['colsample'],
                'lambda_l1': hp['l1'],
                'lambda_l2': hp['l2'],
                'max_bin': hp['max_bin'],
                'seed': hp['seed'],
                'deterministic': True,
                'force_col_wise': True,
            },
            lgb.Dataset(x_train, label=arrays['y_train']),
            num_boost_round=hp['rounds'],
        )
        model.save_model(str(out / 'trees.txt'))
    del model, x_train
    predictor = Predictor(str(out))
    measured = evaluate(predictor.predict, arrays['x_val'], arrays['y_val'])
    files = artifact_files(out, hp)
    result = {
        **measured,
        'size': sum(path.stat().st_size for path in files),
        'artifacts': {path.name: sha256(path) for path in files},
        'config': hp,
        'saved': str(out),
        'encoder_source': encoder if hp['features'] == 'learned' else None,
        'external_data': (
            'ImageNet-1K'
            if hp['features'] == 'learned' and json.loads((Path(encoder) / 'config.json').read_text())['initialization'] == 'imagenet'
            else None
        ),
        'feature_version': FEATURE_VERSION,
        'train_seconds': time.monotonic() - started,
        'versions': versions,
        'prediction_precision': 'float32 features; native tree inference; learned encoder mixed-float16 when configured',
        'training_cache_excluded_from_inference': True,
    }
    result['score'] = objective(result, reference)
    (out / 'assessment.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('accuracy', 'latency', 'size', 'score')}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = train(**json.loads(args.request.read_text()))
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    shutil.move(temporary, args.output)


if __name__ == '__main__':
    main()
