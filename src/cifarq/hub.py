"""The trained network as a Hugging Face model: ``ResNet9`` is ``model.model`` behind ``PyTorchModelHubMixin``, so a
saved directory (``model.safetensors`` and ``config.json``) loads with ``ResNet9.from_pretrained(path or repo id)`` and
publishes with ``push_to_hub``. ``config.json`` holds the constructor's arguments (the architecture, the whitening
filters' shape only: their values are weights) and ``tta_flip``, which ``predict`` uses by default.

    from cifarq.hub import ResNet9
    net = ResNet9.from_pretrained('thearn/cifar10-resnet9')
    probs = net.predict(images)            # uint8 (n, 32, 32, 3), RGB: (n, 10) class probabilities
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .data import CLASSES
from .model import MEAN, STD, model

ARCHITECTURE = ('width', 'dropout', 'activation', 'celu_alpha', 'head', 'logit_scale', 'whiten')


class ResNet9(nn.Module, PyTorchModelHubMixin, library_name='pytorch', pipeline_tag='image-classification',
              tags=['image-classification', 'cifar10', 'resnet9', 'optuna'],
              repo_url='https://huggingface.co/thearn/cifar10-resnet9'):
    def __init__(self, width: int = 32, dropout: float = 0.0, activation: str = 'relu', celu_alpha: float = 0.075,
                 head: str = 'linear', logit_scale: float = 1.0, whiten: bool = False, tta_flip: bool = False,
                 classes: list[str] | None = None):
        super().__init__()
        self.classes, self.tta_flip = list(classes or CLASSES), bool(tta_flip)
        self.net = model(int(width), float(dropout), activation, classes=len(self.classes), celu_alpha=celu_alpha,
                         head=head, logit_scale=logit_scale,
                         whitening=torch.zeros(27, 3, 3, 3) if whiten else None)   # its values come with the weights
        self.register_buffer('mean', torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        """Logits of normalised images (n, 3, 32, 32)."""
        return self.net(x)

    @torch.no_grad()
    def predict(self, images, tta_flip: bool | None = None) -> np.ndarray:
        """Class probabilities (n, 10) of uint8 RGB images (n, 32, 32, 3), averaging each image's logits with its
        mirror's when ``tta_flip`` (default: as trained)."""
        self.eval()
        dev = next(self.parameters()).device
        x = torch.tensor(np.asarray(images, dtype=np.uint8), device=dev).permute(0, 3, 1, 2).float() / 255
        x = (x - self.mean) / self.std
        logits = self(x)
        if self.tta_flip if tta_flip is None else tta_flip:
            logits = logits + self(x.flip(3))
        return logits.float().softmax(1).cpu().numpy()


def save(net: nn.Module, h: dict, directory: Path | str) -> Path:
    """Write a trained ``train.model`` network (``h``: its hyperparameters) as a ``ResNet9`` directory: fp32 weights,
    the architecture and ``tta_flip`` in config.json, the full hyperparameters in hyperparameters.json."""
    import json
    out = ResNet9(**{k: h[k] for k in ARCHITECTURE}, tta_flip=h['tta_flip'])
    out.net.load_state_dict({k: v.detach().float().cpu() for k, v in net.state_dict().items()})
    directory = Path(directory)
    out.save_pretrained(directory)
    (directory / 'hyperparameters.json').write_text(json.dumps(h, indent=1) + '\n')
    return directory
