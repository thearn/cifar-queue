"""Torchvision ResNet-18 training and host-to-host inference; no queue imports and no test-set access."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .pipeline_score import evaluate, load_development, objective

if TYPE_CHECKING:
    import torch

DEFAULTS = {
    'initialization': 'imagenet',
    'stem': 'imagenet',
    'resolution': 64,
    'epochs': 30,
    'batch_size': 256,
    'optimizer': 'sgd',
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'label_smoothing': 0.0,
    'warmup': 3,
    'min_lr_ratio': 0.01,
    'augment': 'basic',
    'mixup': 0.0,
    'cutmix': 0.0,
    'erase': 0.0,
    'backbone_lr_ratio': 1.0,
    'freeze_epochs': 0,
    'precision': 'float16',
    'seed': 40,
    'threads': 4,
}


def settings(overrides: dict) -> dict:
    unknown = set(overrides) - DEFAULTS.keys()
    if unknown:
        raise ValueError(f'unknown ResNet18 options: {sorted(unknown)}')
    hp = {**DEFAULTS, **overrides}
    for key, allowed in {
        'initialization': ('imagenet', 'scratch'),
        'stem': ('imagenet', 'cifar'),
        'optimizer': ('sgd', 'adamw'),
        'augment': ('basic', 'randaugment', 'trivialaugment'),
        'precision': ('float16', 'float32'),
    }.items():
        if hp[key] not in allowed:
            raise ValueError(f'invalid {key}')
    for key in ('epochs', 'batch_size', 'resolution', 'threads'):
        if not isinstance(hp[key], int) or hp[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('lr', 'backbone_lr_ratio'):
        if not math.isfinite(hp[key]) or hp[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('weight_decay', 'mixup', 'cutmix'):
        if not math.isfinite(hp[key]) or hp[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    for key in ('label_smoothing', 'erase', 'min_lr_ratio', 'momentum'):
        if not 0 <= hp[key] < 1:
            raise ValueError(f'{key} must be in [0,1)')
    if not 0 <= hp['freeze_epochs'] < hp['epochs'] or not 0 <= hp['warmup'] < hp['epochs']:
        raise ValueError('freeze/warmup must be shorter than training')
    return hp


def build(hp: dict, *, pretrained: bool = True):
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained and hp['initialization'] == 'imagenet' else None
    model = resnet18(weights=weights)
    if hp['stem'] == 'cifar':
        old = model.conv1.weight.detach()
        model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        if weights is not None:
            with torch.no_grad():
                model.conv1.weight.copy_(old[:, :, 2:5, 2:5])
        model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    return model


def inputs(images, hp: dict):
    import torch
    import torch.nn.functional as F

    x = images.permute(0, 3, 1, 2).float().div_(255)
    if hp['resolution'] != 32:
        x = F.interpolate(x, (hp['resolution'], hp['resolution']), mode='bilinear', align_corners=False, antialias=True)
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


class Predictor:
    def __init__(self, directory: str):
        import torch

        self.hp = json.loads((Path(directory) / 'config.json').read_text())
        torch.set_num_threads(self.hp['threads'])
        self.model = build(self.hp, pretrained=False).cuda().to(memory_format=torch.channels_last).eval()
        self.model.load_state_dict(torch.load(Path(directory) / 'weights.pt', map_location='cuda', weights_only=True))

    def predict(self, images: np.ndarray) -> np.ndarray:
        import torch

        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=self.hp['precision'] == 'float16'):
            x = torch.from_numpy(np.ascontiguousarray(images)).cuda()
            return self.model(inputs(x, self.hp)).argmax(1).cpu().numpy()


def augment(images, hp: dict):
    import torch
    import torch.nn.functional as F
    from torchvision.transforms import v2

    x = images.permute(0, 3, 1, 2)
    n = len(x)
    padded = F.pad(x, (4, 4, 4, 4), mode='reflect')
    dy, dx = torch.randint(0, 9, (2, n), device=x.device)
    grid = torch.arange(32, device=x.device)
    x = padded.permute(0, 2, 3, 1)[
        torch.arange(n, device=x.device)[:, None, None],
        dy[:, None, None] + grid[None, :, None],
        dx[:, None, None] + grid[None, None, :],
    ].permute(0, 3, 1, 2)
    x = torch.where(torch.rand(n, 1, 1, 1, device=x.device) < 0.5, x.flip(3), x)
    if hp['augment'] == 'randaugment':
        x = v2.RandAugment()(x)
    elif hp['augment'] == 'trivialaugment':
        x = v2.TrivialAugmentWide()(x)
    x = inputs(x.permute(0, 2, 3, 1), hp)
    if hp['erase']:
        x = v2.RandomErasing(p=hp['erase'])(x)
    return x


def _lr_factor(hp: dict, epoch: int) -> float:
    if epoch < hp['warmup']:
        return (epoch + 1) / max(1, hp['warmup'])
    phase = (epoch - hp['warmup']) / max(1, hp['epochs'] - hp['warmup'])
    return hp['min_lr_ratio'] + (1 - hp['min_lr_ratio']) * (1 + math.cos(math.pi * phase)) / 2


def _cutmix_box(size: int, alpha: float) -> tuple[int, int, int, int, float]:
    lam = float(np.random.beta(alpha, alpha))
    side = round(size * math.sqrt(1 - lam))
    cy, cx = np.random.randint(size, size=2)
    y0, y1 = max(0, int(cy) - side // 2), min(size, int(cy) + side // 2)
    x0, x1 = max(0, int(cx) - side // 2), min(size, int(cx) + side // 2)
    return y0, y1, x0, x1, 1 - (y1 - y0) * (x1 - x0) / size**2


def _set_epoch_mode(model: torch.nn.Module, optimizer: torch.optim.Optimizer, hp: dict, epoch: int) -> None:
    model.train()
    frozen = epoch < hp['freeze_epochs']
    for name, param in model.named_parameters():
        param.requires_grad_(not frozen or name.startswith('fc.'))
    if frozen:
        for name, layer in model.named_modules():
            if not name.startswith('fc'):
                layer.eval()
    factor = _lr_factor(hp, epoch)
    for group, ratio in zip(optimizer.param_groups, (hp['backbone_lr_ratio'], 1.0), strict=True):
        group['lr'] = hp['lr'] * ratio * factor


def _mix_batch(x: torch.Tensor, permutation: torch.Tensor, hp: dict) -> tuple[torch.Tensor, float]:
    if hp['cutmix'] and (not hp['mixup'] or random.random() < 0.5):
        y0, y1, x0, x1, lam = _cutmix_box(x.shape[-1], hp['cutmix'])
        x[:, :, y0:y1, x0:x1] = x[permutation, :, y0:y1, x0:x1]
        return x, lam
    if hp['mixup']:
        lam = float(np.random.beta(hp['mixup'], hp['mixup']))
        return lam * x + (1 - lam) * x[permutation], lam
    return x, 1.0


def _evaluate_saved(out: Path, arrays: dict, reference: dict, hp: dict, started: float) -> dict:
    import torch
    import torchvision

    from .paths import sha256

    predictor = Predictor(str(out))
    result = evaluate(predictor.predict, arrays['x_val'], arrays['y_val'])
    result.update(
        size=sum((out / f).stat().st_size for f in ('weights.pt', 'config.json')),
        train_seconds=time.monotonic() - started,
        config=hp,
        saved=str(out),
        external_data='ImageNet-1K' if hp['initialization'] == 'imagenet' else None,
        pretrained_weights='torchvision.ResNet18_Weights.IMAGENET1K_V1' if hp['initialization'] == 'imagenet' else None,
        parameters=sum(p.numel() for p in predictor.model.parameters()),
        weights_sha256=sha256(out / 'weights.pt'),
        torch_version=torch.__version__,
        torchvision_version=torchvision.__version__,
        gpu=torch.cuda.get_device_name(),
    )
    result['score'] = objective(result, reference)
    (out / 'assessment.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('accuracy', 'latency', 'size', 'score')}), flush=True)
    return result


def train(data: str, output: str, reference: dict, config: dict, *, seconds: float = 7200) -> dict:
    import torch
    import torch.nn.functional as F

    hp = settings(config)
    torch.set_num_threads(hp['threads'])
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('this campaign is frozen to the local T4')
    torch.manual_seed(hp['seed'])
    np.random.seed(hp['seed'])
    random.seed(hp['seed'])
    torch.backends.cudnn.benchmark = True
    arrays = load_development(data)
    images = torch.from_numpy(arrays['x_train']).cuda()
    labels = torch.from_numpy(arrays['y_train'].astype(np.int64)).cuda()
    model = build(hp).cuda().to(memory_format=torch.channels_last)
    groups = [
        {'params': [p for n, p in model.named_parameters() if not n.startswith('fc.')], 'lr': hp['lr'] * hp['backbone_lr_ratio']},
        {'params': model.fc.parameters(), 'lr': hp['lr']},
    ]
    optimizer = (
        torch.optim.SGD(groups, momentum=hp['momentum'], weight_decay=hp['weight_decay'], nesterov=hp['momentum'] > 0)
        if hp['optimizer'] == 'sgd'
        else torch.optim.AdamW(groups, weight_decay=hp['weight_decay'])
    )
    scaler = torch.amp.GradScaler('cuda', enabled=hp['precision'] == 'float16')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(hp, indent=2) + '\n')
    started = time.monotonic()
    curve = []
    for epoch in range(hp['epochs']):
        _set_epoch_mode(model, optimizer, hp, epoch)
        order = torch.randperm(len(images), device='cuda')
        total_loss = 0.0
        for start in range(0, len(images), hp['batch_size']):
            if time.monotonic() - started > seconds:
                raise TimeoutError('training reached the declared per-run budget; no completed result')
            idx = order[start : start + hp['batch_size']]
            x, y = augment(images[idx], hp), labels[idx]
            permutation = torch.randperm(len(x), device='cuda')
            x, lam = _mix_batch(x, permutation, hp)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=hp['precision'] == 'float16'):
                logits = model(x)
                loss = lam * F.cross_entropy(logits, y, label_smoothing=hp['label_smoothing'])
                loss += (1 - lam) * F.cross_entropy(logits, y[permutation], label_smoothing=hp['label_smoothing'])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(x)
        row = {'epoch': epoch + 1, 'loss': total_loss / len(images), 'seconds': time.monotonic() - started}
        curve.append(row)
        print(json.dumps(row), flush=True)
        (out / 'curve.json').write_text(json.dumps(curve) + '\n')
        torch.save(model.state_dict(), out / 'weights.pt')
    del model, optimizer, images, labels
    torch.cuda.empty_cache()
    return _evaluate_saved(out, arrays, reference, hp, started)
