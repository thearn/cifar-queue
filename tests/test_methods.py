"""MT6 ``gated_logits`` gathers the images under the margin and runs the full pass on them alone: the logits equal the
full pass kept where the margin is small (``masked_logits``, ``jnp.where`` over the whole batch), and the
full pass sees only the gathered images, padded to one of ``gate_sizes``."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cifarq import airbench, jx, methods


def masked(cheap, full, x, gate):
    """The reference: the full pass for every image, kept where the top-two margin is under ``gate``."""
    return methods.masked_logits(cheap, x, 0, gate, lambda _apply, v, _level: full(v))


def test_gate_sizes():
    assert methods.gate_sizes(500) == [16, 32, 64, 128, 256, 500]
    assert methods.gate_sizes(16) == [16] and methods.gate_sizes(5) == [5]


@pytest.mark.parametrize('gate', [0.0, 0.3, 0.7, 1.01])  # none, some, most and every image under the margin
def test_gathered_equals_masked_and_runs_the_full_pass_on_the_gathered_only(gate):
    x = jax.random.normal(jax.random.key(0), (100, 10))
    cheap = lambda v: 3 * jnp.tanh(v)
    seen = []

    def full(v):
        seen.append(len(v))
        return 3 * jnp.tanh(v + 0.5)

    got, need = methods.gated_logits(cheap, full, x, gate)
    assert jnp.array_equal(got, masked(cheap, lambda v: 3 * jnp.tanh(v + 0.5), x, gate))
    assert seen == ([] if need.sum() == 0 else [next(k for k in methods.gate_sizes(100) if k >= need.sum())])


def test_chunks_hold_at_most_a_chunk_and_compile_one_shape():
    x = jax.random.normal(jax.random.key(2), (100, 10))
    seen = []

    def full(v):
        seen.append(len(v))
        return 3 * jnp.tanh(v + 0.5)

    want = masked(lambda v: 3 * jnp.tanh(v), full, x, 1.01)
    seen.clear()
    got, need = methods.gated_logits(lambda v: 3 * jnp.tanh(v), full, x, 1.01, chunk=32)  # every image: 100 in 4 chunks
    assert need.all() and seen == [32, 32, 32, 32] and jnp.array_equal(got, want)
    assert jnp.array_equal(methods.chunked(lambda v: v * 2, np.arange(10.0)[:, None], 4), jnp.arange(10.0)[:, None] * 2)


def test_gathered_equals_masked_on_a_network():
    """Six views of a small airbench network: each image's logits come from itself alone, so the gathered subset gives
    the same logits as the whole batch (to float32 rounding: XLA may pick other kernels for another batch shape)."""
    net, _ = airbench.build({**airbench.DEFAULTS, 'widths': (16, 32, 32)}, jnp.float32)
    S = jx.Split(net)
    fwd = lambda level: lambda v: airbench.tta_logits(lambda u: S.apply(S.params, S.rest, u, train=False)[0], v, level)
    x = jax.random.normal(jax.random.key(1), (64, 32, 32, 3))
    gate = float(np.quantile(np.diff(np.sort(np.asarray(jax.nn.softmax(fwd(0)(x), 1)), 1)[:, -2:], axis=1), 0.4))
    got, need = methods.gated_logits(fwd(0), fwd(2), x, gate)
    assert 0 < need.sum() < len(x)
    want = masked(fwd(0), fwd(2), x, gate)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5) and jnp.array_equal(got.argmax(1), want.argmax(1))
