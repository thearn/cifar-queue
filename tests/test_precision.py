"""``cifarq.precision`` picks the arithmetic from the card: float16 under ``auto`` only on a GPU without native
bfloat16 and only for a summed loss, bfloat16 for the published 'fp16'/'amp' names, float32 on a CPU always.
The cards are stand-ins, so this runs anywhere."""

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
    ('dev', 'precision_', 'summed', 'want'),
    [
        (T4, 'auto', True, jnp.float16),
        (T4, 'auto', False, jnp.bfloat16),  # a mean loss: no float16 without loss scaling
        (T4, 'fp16', True, jnp.bfloat16),  # the published name keeps what earlier records were measured with
        (T4, 'amp', True, jnp.bfloat16),
        (T4, 'f16', False, jnp.float16),
        (T4, 'fp32', True, jnp.float32),
        (AMPERE, 'auto', True, jnp.bfloat16),
        (CPU, 'auto', True, jnp.float32),
        (CPU, 'f16', True, jnp.float32),
    ],
)
def test_compute_dtype(card, dev, precision_, summed, want):
    card(dev)
    assert precision.compute_dtype(precision_, summed_loss=summed) == want


def test_airbench_sums_the_loss_of_sgd_and_muon_only():
    """The one definition both the loss and ``auto``'s float16 choice read."""
    from cifarq import airbench

    assert [airbench.summed_loss({'optimizer': o}) for o in ('sgd', 'muon', 'adamw')] == [True, True, False]


@pytest.mark.parametrize(('cc', 'native'), [('7.5', False), ('8.0', True), ('8.9', True), ('', True), (None, True)])
def test_native_bf16_reads_the_compute_capability(card, cc, native):
    """An unreadable capability counts as native: bfloat16 is the choice that can never underflow."""
    card(SimpleNamespace(platform='gpu', compute_capability=cc))
    assert precision.native_bf16() is native
