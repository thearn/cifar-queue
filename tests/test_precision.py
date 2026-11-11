"""``cifarq.precision`` picks the arithmetic from the card: float16 under ``auto`` only on a GPU without native
bfloat16 and only for a summed loss, bfloat16 for the published 'fp16'/'amp' names, float32 on a CPU always; and the
evaluation batch from the card's memory. The cards are stand-ins, so this runs anywhere."""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from cifarq import precision

T4 = SimpleNamespace(platform='gpu', compute_capability='7.5')
AMPERE = SimpleNamespace(platform='gpu', compute_capability='8.6')
CPU = SimpleNamespace(platform='cpu')


@pytest.fixture
def card(monkeypatch):
    def use(dev):
        monkeypatch.setattr(precision.jx, 'pick', lambda device='auto': dev)

    return use


@pytest.mark.parametrize(
    ('dev', 'h', 'want'),
    [
        (T4, {'precision': 'auto', 'optimizer': 'sgd'}, jnp.float16),
        (T4, {'precision': 'auto', 'optimizer': 'muon'}, jnp.float16),
        (T4, {'precision': 'auto', 'optimizer': 'adamw'}, jnp.bfloat16),  # a mean loss: no float16 without loss scaling
        (T4, {'precision': 'fp16'}, jnp.bfloat16),  # the published name keeps what earlier records were measured with
        (T4, {'precision': 'amp'}, jnp.bfloat16),
        (T4, {}, jnp.bfloat16),
        (T4, {'precision': 'f16', 'optimizer': 'adamw'}, jnp.float16),
        (T4, {'precision': 'fp32'}, jnp.float32),
        (AMPERE, {'precision': 'auto', 'optimizer': 'sgd'}, jnp.bfloat16),
        (CPU, {'precision': 'auto'}, jnp.float32),
        (CPU, {'precision': 'f16'}, jnp.float32),
    ],
)
def test_compute_dtype(card, dev, h, want):
    card(dev)
    assert precision.compute_dtype(h) == want


@pytest.mark.parametrize(('cc', 'native'), [('7.5', False), ('8.0', True), ('8.9', True), ('', True), (None, True)])
def test_native_bf16_reads_the_compute_capability(card, cc, native):
    """An unreadable capability counts as native: bfloat16 is the choice that can never underflow."""
    card(SimpleNamespace(platform='gpu', compute_capability=cc))
    assert precision.native_bf16() is native


def test_eval_batch_follows_memory_unless_set():
    gb = lambda n: SimpleNamespace(memory_stats=lambda: {'bytes_limit': n * 1024**3})
    assert precision.eval_batch_size({}, gb(16)) == 500
    assert precision.eval_batch_size({}, gb(24)) == 2000
    assert precision.eval_batch_size({}, None) == 500
    assert precision.eval_batch_size({'eval_batch': 123}, gb(80)) == 123
