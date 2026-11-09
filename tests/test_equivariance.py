"""Session 2026-10-30-equivariant-small, Q1/Q2: the D4 layers are equivariant (transforming the input by g transforms
each output block spatially by g and permutes the blocks as g^-1 k), the blur pool commutes with the 8 transforms on
even maps, field pooling is invariant, and the expanded kernel is what the MAC counter counts."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from cifarq import airbench, footprint, methods

MUL, INV = methods._d4_tables()


def blocks_after(y, g, k=8):
    """What an equivariant layer's output should become when its input is transformed by g (NHWC, group-major)."""
    n, h, w, kc = y.shape
    b = y.reshape(n, h, w, k, kc // k)
    return jnp.stack([methods.d4_transform(b[..., MUL[INV[g]][j], :], g) for j in range(k)], 3).reshape(n, h, w, kc)


def transform_field(x, g, k=8):
    return blocks_after(x, g, k)


def randn(seed, shape):
    return jax.random.normal(jax.random.key(seed), shape)


def test_group_table_is_a_group():
    assert all(MUL[a][INV[a]] == 0 and MUL[INV[a]][a] == 0 for a in range(8))
    assert all(MUL[MUL[a][b]][c] == MUL[a][MUL[b][c]] for a in range(8) for b in range(8) for c in range(8))


def test_p4m_lifting_and_group_convolutions_are_equivariant():
    lift, conv = methods.P4MConv(3, 4, lift=True, rngs=nnx.Rngs(0)), methods.P4MConv(4, 5, rngs=nnx.Rngs(1))
    x = randn(0, (2, 8, 8, 3))
    for g in range(8):
        y = lift(x)
        assert jnp.allclose(lift(methods.d4_transform(x, g)), blocks_after(y, g), atol=1e-5)
        z = conv(y)
        assert jnp.allclose(conv(transform_field(y, g)), blocks_after(z, g), atol=1e-5)


def test_flipconv_matches_the_same_rule_for_its_two_elements():
    lift, conv = methods.FlipConv(3, 4, lift=True, rngs=nnx.Rngs(0)), methods.FlipConv(4, 4, rngs=nnx.Rngs(1))
    x = randn(0, (2, 8, 8, 3))
    flip = lambda t: t[:, :, ::-1]
    y = lift(x)
    c2 = y.shape[-1]
    swap = lambda t: jnp.concatenate([flip(t[..., c2 // 2 :]), flip(t[..., : c2 // 2])], -1)
    assert jnp.allclose(lift(flip(x)), swap(y), atol=1e-5)
    assert jnp.allclose(conv(swap(y)), swap(conv(y)), atol=1e-5)


def test_blur_pool_commutes_with_the_eight_transforms_and_pooling_is_invariant():
    pool, gp = methods.MaxBlurPool(), methods.GroupMaxPool(8)
    x = randn(0, (2, 16, 16, 4))
    for g in range(8):
        assert jnp.allclose(pool(methods.d4_transform(x, g)), methods.d4_transform(pool(x), g), atol=1e-6)
    f = randn(1, (2, 6, 6, 8 * 3))
    for g in range(8):
        assert jnp.allclose(gp(transform_field(f, g)), gp(f), atol=1e-6)
    assert pool(x).shape == (2, 8, 8, 4)


def test_d4_network_builds_counts_and_trains(built):
    h = {**airbench.DEFAULTS, 'widths': (16, 8, 8), 'depth': 3, 'flip_eq': 2, 'eq_group': 'd4', 'blur_pool': True}
    net, _ = airbench.build(h)
    expanded = methods.P4MConv(8, 8).full()
    assert expanded.shape == (3, 3, 64, 64)
    assert footprint.count_macs(methods.P4MConv(8, 8), (1, 4, 4, 64)) == 16 * 64 * 64 * 9
    r = airbench.train(widths=(16, 8, 8), depth=3, flip_eq=2, eq_group='d4', blur_pool=True, batch_size=64, epochs=1, test='sealed')
    assert 0 <= r['val_acc'] <= 1 and r['params'] == airbench.param_counts(net)['stored']


def test_field_se_keeps_equivariance_and_srf_reduces_stored_weights():
    se = methods.FieldSE(4, k=8, reduction=2)
    se.fc2.kernel = nnx.Param(randn(5, se.fc2.kernel.shape))  # (away from the identity start)
    f = randn(0, (2, 6, 6, 32))
    for g in range(8):
        assert jnp.allclose(se(transform_field(f, g)), blocks_after(se(f), g), atol=1e-5)
    conv = methods.P4MConv(4, 5)
    methods.use_srf(conv)
    assert conv.weight.shape == (32, 5, 6)
    x = randn(1, (1, 8, 8, 32))
    for g in range(8):  # (the basis is closed under D4: still equivariant)
        assert jnp.allclose(conv(transform_field(x, g)), blocks_after(conv(x), g), atol=1e-4)


def test_se_srf_and_filtering_train(built):
    base = dict(widths=(16, 16, 16), depth=3, flip_eq=2, batch_size=64, epochs=1, test='sealed')
    plain = airbench.train(**base)
    r = airbench.train(**base, se=4, srf=True)
    assert r['params'] < plain['params'] + 2000 and r['params'] != plain['params']
    f = airbench.train(**base, filter_keep=0.5, filter_proxy_widths=(8, 8, 8))
    assert f['epochs_charged'] == 0.5 and f['proxy']['params'] > 0 and 0 <= f['val_acc'] <= 1


def test_muon_trains_and_orthogonalises(built):
    g = randn(0, (16, 72))
    o = np.asarray(methods.newton_schulz(g), np.float32)
    sv = np.linalg.svd(o, compute_uv=False)
    assert sv.max() < 1.5 and sv.min() > 0.3  # (Muon's iteration: singular values near 1, not exactly)
    perm = np.random.default_rng(0).permutation(72)  # (column order does not matter: HWIO vs OIHW flattening)
    assert np.allclose(np.asarray(methods.newton_schulz(g[:, perm]), np.float32), o[:, perm], atol=2e-2)
    r = airbench.train(widths=(16, 16, 16), depth=3, flip_eq=2, batch_size=64, epochs=1, test='sealed', optimizer='muon')
    assert 0.1 < r['val_acc'] <= 1


def test_levers_run_in_bf16_on_the_gpu(built):
    import pytest

    if jax.default_backend() != 'gpu':
        pytest.skip('needs a GPU (the bf16 path)')
    for kw in ({'se': 4}, {'srf': True}, {'optimizer': 'muon'}, {'eq_group': 'd4'}, {'blur_pool': True}):
        r = airbench.train(widths=(16, 16, 16), depth=3, flip_eq=2, batch_size=64, epochs=1, test='sealed', **kw)
        assert 0 <= r['val_acc'] <= 1, kw
