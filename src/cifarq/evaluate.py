"""Accuracy, one function for training, the champion and the model card. ``OBJECTIVE`` is the metric searches optimise
(on the validation split, ``DIRECTION``) and the champion is chosen by; ``KEY`` is its name in ``train``'s result."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import jax
    from jax.typing import ArrayLike

    from .jx import Split

OBJECTIVE = 'acc'
DIRECTION = 'maximize'
KEY = f'val_{OBJECTIVE}'


@functools.lru_cache(maxsize=16)
def _forward(split: Split, tta_flip: bool) -> Callable[[Any, Any, jax.Array], jax.Array]:
    import jax
    import jax.numpy as jnp

    from . import jx
    from .model import MEAN, STD

    @jax.jit
    def fn(params: Any, rest: Any, xb: jax.Array) -> jax.Array:
        xb = jx.normalise(xb, MEAN, STD)
        z, _ = split.apply(params, rest, xb, train=False)
        if tta_flip:  # the mirror's logits too
            z = z + split.apply(params, rest, xb[:, :, ::-1], train=False)[0]
        return z.astype(jnp.float32)

    return fn


def logits(split: Split, params: Any, rest: Any, x: ArrayLike, tta_flip: bool = False, batch: int = 1000) -> jax.Array:
    """Evaluation-mode logits (float32) of uint8 NHWC images ``x`` through a ``jx.Split`` network at ``params`` and
    ``rest``, standardised with ``model.MEAN``/``STD``; each image's logits summed with its mirror's when ``tta_flip``.
    One compiled program per network and batch shape (``_forward``)."""
    import jax.numpy as jnp

    from . import jx

    fn = _forward(split, bool(tta_flip))
    return jx.padded_apply(lambda xb: fn(params, rest, xb), jnp.asarray(x), batch)


def accuracy(split: Split, params: Any, rest: Any, x: ArrayLike, y: ArrayLike, tta_flip: bool = False, batch: int = 1000) -> float:
    """The share of uint8 images ``x`` (NHWC) whose likeliest class is ``y`` (``logits``)."""
    import numpy as np

    z = np.asarray(logits(split, params, rest, x, tta_flip, batch))
    return float((z.argmax(1) == np.asarray(y)).mean())
