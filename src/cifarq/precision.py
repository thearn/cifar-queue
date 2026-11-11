"""What arithmetic a run uses, and how many images an evaluation forward pass holds. Both answers depend on the
card rather than on the recipe, which is why they live apart from the families that ask for them.

A GPU runs bfloat16 on its tensor cores from compute capability 8.0 (Ampere). A Turing card -- a Tesla T4 -- does
not: it computes bfloat16 off them, at about a fifth of its float16 throughput (measured on a T4: matmul fp32 4.87,
bf16 5.14, fp16 26.28 TFLOP/s). The published recipes pin bfloat16, so on such a card they trained at roughly a
fifth of the rate the hardware allows. ``precision='auto'`` (opt in; airbench's default stays 'fp16') is what fixes
that: on a T4 it trains airbench94 2.2x faster end to end with the accuracy unchanged. On a card with native bfloat16
``auto`` changes nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from . import jx

if TYPE_CHECKING:
    from typing import Any

    from jax.typing import DTypeLike


def native_bf16(device: str = 'auto') -> bool:
    """Whether ``device`` runs bfloat16 on its tensor cores: an NVIDIA GPU does from compute capability 8.0. A value
    that cannot be read is treated as native, the safe default, since bfloat16's range never needs loss scaling."""
    d = jx.pick(device)
    if d.platform != 'gpu':
        return False
    head = ''.join(c for c in str(getattr(d, 'compute_capability', '') or '') if c.isdigit() or c == '.').rstrip('.')
    try:
        return float(head) >= 8.0
    except ValueError:
        return True


def compute_dtype(h: dict, device: str = 'auto') -> DTypeLike:
    """The arithmetic of a run. ``precision``: ``fp32``; ``fp16`` and ``amp``, both bfloat16, what the published
    recipes and every record before 2026-11-10 were measured with; ``f16``, true float16, for a card whose tensor
    cores do not take bfloat16; ``auto``, bfloat16 where it is native and float16 where it is not.
    Weights, optimiser state and the loss stay float32 under all of them, and a CPU computes in float32.

    ``auto`` reaches for float16 only when the loss is summed over the batch -- ``sgd`` and ``muon`` -- where the
    gradients are already about ``batch_size`` times a mean loss's and so sit well inside float16's range. That sum
    is what stands in for the dynamic loss scaling a float16 recipe usually needs. Under a mean loss ``auto`` keeps
    bfloat16, slow on such a card but unable to underflow; ask for ``f16`` to override that deliberately."""
    p = h.get('precision', 'fp16')
    if p == 'fp32' or jx.pick(device).platform != 'gpu':
        return jnp.float32
    if p == 'f16':
        return jnp.float16
    if p == 'auto' and not native_bf16(device) and h.get('optimizer', 'sgd') in ('sgd', 'muon'):
        return jnp.float16
    return jnp.bfloat16


def eval_batch_size(h: dict, dev: Any = None) -> int:
    """Images an evaluation forward pass. ``eval_batch`` when set; otherwise 2000 on a device holding 24 GB or more
    and 500 below it, since a six-view pass keeps six times the batch in activations and a T4's 16 GB ran out of
    memory at 2000 x 6. It only divides the work -- each image's logits come from itself alone -- so no value of it
    moves an accuracy."""
    if h.get('eval_batch'):
        return int(h['eval_batch'])
    stats = (getattr(dev, 'memory_stats', lambda: None)() or {}) if dev is not None else {}
    total = stats.get('bytes_limit') or stats.get('bytes_reservable_limit') or 0
    return 2000 if total >= 24 * 1024**3 else 500
