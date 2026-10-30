"""Session 2026-10-30-equivariant-small, Q1/Q2: the D4 layers are equivariant (transforming the input by g transforms
each output block spatially by g and permutes the blocks as g^-1 k), the blur pool commutes with the 8 transforms on
even maps, field pooling is invariant, and the expanded kernel is what the MAC counter counts."""
import torch

from cifarq import airbench, footprint, methods

MUL, INV = methods._d4_tables()


def blocks_after(y, g, k=8):
    """What an equivariant layer's output should become when its input is transformed by g."""
    n, kc, h, w = y.shape
    b = y.view(n, k, kc // k, h, w)
    return torch.stack([methods.d4_transform(b[:, MUL[INV[g]][j]], g) for j in range(k)], 1).view(n, kc, h, w)


def transform_field(x, g, k=8):
    return blocks_after(x, g, k)


def test_group_table_is_a_group():
    assert all(MUL[a][INV[a]] == 0 and MUL[INV[a]][a] == 0 for a in range(8))
    assert all(MUL[MUL[a][b]][c] == MUL[a][MUL[b][c]] for a in range(8) for b in range(8) for c in range(8))


def test_p4m_lifting_and_group_convolutions_are_equivariant():
    torch.manual_seed(0)
    lift, conv = methods.P4MConv(3, 4, lift=True), methods.P4MConv(4, 5)
    x = torch.randn(2, 3, 8, 8)
    for g in range(8):
        y = lift(x)
        assert torch.allclose(lift(methods.d4_transform(x, g)), blocks_after(y, g), atol=1e-5)
        z = conv(y)
        assert torch.allclose(conv(transform_field(y, g)), blocks_after(z, g), atol=1e-5)


def test_flipconv_matches_the_same_rule_for_its_two_elements():
    torch.manual_seed(0)
    lift, conv = methods.FlipConv(3, 4, lift=True), methods.FlipConv(4, 4)
    x = torch.randn(2, 3, 8, 8)
    flip = lambda t: t.flip(-1)
    y = lift(x)
    n, c2, h, w = y.shape
    swap = lambda t: torch.cat([flip(t[:, c2 // 2:]), flip(t[:, :c2 // 2])], 1)
    assert torch.allclose(lift(flip(x)), swap(y), atol=1e-5)
    assert torch.allclose(conv(swap(y)), swap(conv(y)), atol=1e-5)


def test_blur_pool_commutes_with_the_eight_transforms_and_pooling_is_invariant():
    torch.manual_seed(0)
    pool, gp = methods.MaxBlurPool(), methods.GroupMaxPool(8)
    x = torch.randn(2, 4, 16, 16)
    for g in range(8):
        assert torch.allclose(pool(methods.d4_transform(x, g)), methods.d4_transform(pool(x), g), atol=1e-6)
    f = torch.randn(2, 8 * 3, 6, 6)
    for g in range(8):
        assert torch.allclose(gp(transform_field(f, g)), gp(f), atol=1e-6)
    assert pool(x).shape == (2, 4, 8, 8)


def test_d4_network_builds_counts_and_trains(built):
    h = {**airbench.DEFAULTS, 'widths': (16, 8, 8), 'depth': 3, 'flip_eq': 2, 'eq_group': 'd4', 'blur_pool': True}
    net, _ = airbench.build(h)
    expanded = methods.P4MConv(8, 8).full()
    assert expanded.shape == (64, 64, 3, 3)
    assert footprint.count_macs(methods.P4MConv(8, 8), (1, 64, 4, 4)) == 16 * 64 * 64 * 9
    r = airbench.train(widths=(16, 8, 8), depth=3, flip_eq=2, eq_group='d4', blur_pool=True, batch_size=64, epochs=1,
                       test='sealed')
    assert 0 <= r['val_acc'] <= 1 and r['params'] == airbench.param_counts(net)['stored']
