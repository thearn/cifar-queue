"""The trained network as a Hugging Face model: ``ResNet9`` is ``model.ResNet9`` (Flax NNX) behind ``ModelHubMixin``,
so a saved directory (``model.safetensors`` and ``config.json``) loads with ``ResNet9.from_pretrained(path or repo id)``
and publishes with ``push_to_hub``. ``config.json`` holds the constructor's arguments (the architecture, whether there
is a whitening layer: its filters are weights) and ``tta_flip``, which ``predict`` uses by default. The weights are
float32 arrays keyed by their path in the module (``layer1.conv.kernel``, ``prep.bn.mean``, ``white.kernel``), the
kernels in Flax's HWIO layout.

    from cifarq.hub import ResNet9
    net = ResNet9.from_pretrained('thearn/cifar10-resnet9')
    probs = net.predict(images)            # uint8 (n, 32, 32, 3), RGB: (n, 10) class probabilities
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from huggingface_hub import ModelHubMixin

from .data import CLASSES

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from flax import nnx
    from numpy.typing import ArrayLike

    from .jx import Split

ARCHITECTURE = ('width', 'dropout', 'activation', 'celu_alpha', 'head', 'logit_scale', 'whiten')
WEIGHTS = 'model.safetensors'
CARD = """
---
{{ card_data }}
---

A ResNet-9 for CIFAR-10 in JAX (Flax NNX), trained and tuned with [cifar-queue]({{ repo_url }}).
"""


def _kinds() -> tuple[type, ...]:
    from flax import nnx

    from . import jx

    return (nnx.Param, nnx.BatchStat, jx.Frozen)


def _key(path: Sequence[Any]) -> str:
    return '.'.join(str(getattr(p, 'key', getattr(p, 'idx', getattr(p, 'name', p)))) for p in path)


def flat_weights(net: nnx.Module) -> dict[str, np.ndarray]:
    """A module's saved arrays (parameters, batch statistics, frozen filters) as {path: float32 array}."""
    import jax
    from flax import nnx

    leaves = jax.tree_util.tree_leaves_with_path(nnx.state(net, _kinds()))
    return {_key(path): np.asarray(leaf, np.float32) for path, leaf in leaves}


def load_weights(net: nnx.Module, weights: dict[str, np.ndarray]) -> None:
    """Put ``flat_weights``' arrays back into a module of the same architecture (every key must match)."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    state = nnx.state(net, _kinds())
    leaves, treedef = jax.tree_util.tree_flatten_with_path(state)
    keys = [_key(p) for p, _ in leaves]
    if set(keys) != set(weights):
        raise ValueError(
            f'weights do not match the architecture: missing {sorted(set(keys) - set(weights))[:5]}, extra {sorted(set(weights) - set(keys))[:5]}'
        )
    nnx.update(net, jax.tree_util.tree_unflatten(treedef, [jnp.asarray(weights[k]) for k in keys]))


class ResNet9(
    ModelHubMixin,
    library_name='jax',
    pipeline_tag='image-classification',
    tags=['image-classification', 'cifar10', 'resnet9', 'optuna', 'jax', 'flax'],
    repo_url='https://github.com/thearn/cifar-queue',
    model_card_template=CARD,
):
    def __init__(
        self,
        width: int = 32,
        dropout: float = 0.0,
        activation: str = 'relu',
        celu_alpha: float = 0.075,
        head: str = 'linear',
        logit_scale: float = 1.0,
        whiten: bool = False,
        tta_flip: bool = False,
        classes: list[str] | None = None,
    ) -> None:
        from . import jx
        from .model import model

        self.classes, self.tta_flip = list(classes or CLASSES), bool(tta_flip)
        self.net = model(
            int(width),
            float(dropout),
            activation,
            classes=len(self.classes),
            celu_alpha=celu_alpha,
            head=head,
            logit_scale=logit_scale,
            whitening=np.zeros((3, 3, 3, 27), np.float32) if whiten else None,  # its values come with the weights
        )
        self.split = jx.Split(self.net)

    def logits(self, images: ArrayLike, tta_flip: bool | None = None) -> np.ndarray:
        """Logits (n, classes) of uint8 RGB images (n, 32, 32, 3), each image's summed with its mirror's when
        ``tta_flip`` (default: as trained)."""
        from . import evaluate, jx

        self.split = jx.Split(self.net)  # (the module's current arrays)
        x = np.asarray(images, dtype=np.uint8).reshape(-1, 32, 32, 3)
        flip = self.tta_flip if tta_flip is None else tta_flip
        return np.asarray(evaluate.logits(self.split, self.split.params, self.split.rest, x, flip, batch=min(len(x), 1000) or 1))

    def predict(self, images: ArrayLike, tta_flip: bool | None = None) -> np.ndarray:
        """Class probabilities (n, 10) of uint8 RGB images (n, 32, 32, 3) (``logits``, softmaxed)."""
        z = self.logits(images, tta_flip)
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)

    def _save_pretrained(self, save_directory: Path) -> None:
        from safetensors.numpy import save_file

        save_file(flat_weights(self.net), str(Path(save_directory) / WEIGHTS))

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        **model_kwargs,
    ) -> ResNet9:
        from huggingface_hub import hf_hub_download
        from safetensors.numpy import load_file

        out = cls(**model_kwargs)
        if Path(model_id).is_dir():
            path = Path(model_id) / WEIGHTS
        else:
            path = hf_hub_download(
                model_id,
                WEIGHTS,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
            )
        load_weights(out.net, load_file(str(path)))
        return out


def save(split: Split, params: Any, rest: Any, h: dict, directory: Path | str) -> Path:
    """Write a trained ``model.ResNet9`` (a ``jx.Split`` at ``params``, ``rest``; ``h``: its hyperparameters) as a
    ``ResNet9`` directory: float32 weights, the architecture and ``tta_flip`` in config.json, the full hyperparameters in
    hyperparameters.json."""
    out = ResNet9(**{k: h[k] for k in ARCHITECTURE}, tta_flip=h['tta_flip'])
    load_weights(out.net, flat_weights(split.merge(params, rest)))
    directory = Path(directory)
    out.save_pretrained(directory)
    (directory / 'hyperparameters.json').write_text(json.dumps(h, indent=1) + '\n')
    return directory
