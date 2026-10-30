"""``airbench.train`` keeps ``train``'s job contract on a CPU, its networks have the published parameter counts, and a
saved network loads back and predicts what training measured."""
import numpy as np
import pytest

from cifarq import airbench

SMALL = {'widths': (16, 32, 32), 'batch_size': 64, 'epochs': 2}


def test_published_parameter_counts():
    count = lambda h: airbench.param_counts(airbench.model(h['widths'], h['depth']))['stored']
    assert abs(count(airbench.AIRBENCH94) - 1.97e6) < 0.01e6           # the paper's 1.97 M
    assert abs(count(airbench.AIRBENCH96) - 9.92e6) < 0.01e6           # 128/384/512, three convs a group


def test_result_contract(built):
    r = airbench.train(**SMALL)
    for k in ('val_acc', 'val_acc_notta', 'test_acc', 'test_acc_notta', 'train_seconds', 'params', 'params_trainable',
              'history', 'hyperparameters'):
        assert k in r
    assert r['device'] == 'cpu' and len(r['history']) == 2 and r['params'] > r['params_trainable']


def test_report_stops_a_trial(built):
    r = airbench.train(**{**SMALL, 'epochs': 4}, report=lambda step, value: step >= 50)
    assert r['pruned'] and r['progress'] == 50 and 'test_acc' not in r


def test_unknown_hyperparameters_are_refused(built):
    with pytest.raises(TypeError, match='unknown hyperparameters'):
        airbench.train(epochs=1, width=16)


@pytest.mark.parametrize('extra', [{'depth': 3, 'cutout': 4, 'translate': 4}, {'train_on': 'train+val', 'tta_level': 1},
                                   {'tta_level': 3, 'flip_eq': 2}])
def test_saved_network_predicts_what_training_measured(built, tmp_path, extra):
    r = airbench.train(**SMALL, **extra, save=str(tmp_path / 'model'))
    probs = airbench.Predictor(r['saved']).predict(built['x_test'])
    assert probs.shape == (200, 10) and np.allclose(probs.sum(1), 1, atol=1e-5)
    assert abs((probs.argmax(1) == built['y_test']).mean() - r['test_acc']) <= 2 / len(probs)


def test_distillation_from_a_saved_teacher(built, tmp_path):
    teacher = airbench.train(**SMALL, save=str(tmp_path / 'teacher'))
    r = airbench.train(**SMALL, kd_teacher=teacher['saved'], kd_weight=0.5)
    assert np.isfinite(r['history'][-1]['loss'])
    r = airbench.train(**SMALL, kd_teacher=f"{teacher['saved']},{teacher['saved']}", kd_weight=0.5)   # an ensemble
    assert np.isfinite(r['history'][-1]['loss'])


@pytest.mark.parametrize('arch,kw', [('convmixer', {'dim': 32, 'depth': 2, 'kernel': 5, 'patch': 2}),
                                     ('resnet9', {'width': 8}), ('pyramidnet', {'depth': 11, 'alpha': 8}),
                                     ('densenet', {'depth': 10, 'growth': 4}), ('shakeshake', {'depth': 8, 'width': 8}),
                                     ('wrn', {'depth': 10, 'k': 1})])
def test_other_families_train_save_and_load(built, tmp_path, arch, kw):
    r = airbench.train(**SMALL, arch=arch, arch_kw=kw, tta_level=1, save=str(tmp_path / 'm'))
    probs = airbench.Predictor(r['saved']).predict(built['x_test'])
    assert abs((probs.argmax(1) == built['y_test']).mean() - r['test_acc']) <= 2 / len(probs)
    assert r['params_trainable'] < r['params']                       # (batch norm's scale frozen)


@pytest.mark.parametrize('augment', ['ta_wide', 'randaugment'])
def test_policy_augmentation_and_adamw(built, augment):
    r = airbench.train(**SMALL, augment=augment, optimizer='adamw', lr=2e-3, weight_decay=0.05, momentum=0.9)
    assert np.isfinite(r['history'][-1]['loss'])


def test_policy_operations_keep_range_and_shape():
    import torch
    from cifarq import augment
    x = torch.rand(64, 3, 32, 32)
    for name in augment.OPS:
        v = augment._strength(name, torch.rand(64), True)
        y = augment._op(name, x, v)
        assert y.shape == x.shape and float(y.min()) >= -1e-5 and float(y.max()) <= 1 + 1e-5, name
    assert torch.equal(augment._op('identity', x, None), x)
    same = augment._op('rotate', x, torch.zeros(64))
    assert torch.allclose(same, x, atol=1e-5)


def test_flip_equivariant_groups_are_invariant_to_mirroring():
    import torch
    net = airbench.model((16, 32, 32), depth=2, flip_eq=1).eval()
    x = torch.randn(4, 32, 8, 8)                                      # (the last group alone, on even-sized maps)
    g3 = net[4]
    feats, mirrored = g3(x), g3(x.flip(-1))
    c = feats.shape[1] // 2
    assert torch.allclose(mirrored[:, :c], feats[:, c:].flip(-1), atol=1e-4)    # mirroring swaps and mirrors the pair
    pooled = net[5]
    assert torch.allclose(pooled(feats), pooled(mirrored), atol=1e-4)


@pytest.mark.parametrize('extra', [{'tie': 1}, {'flip_eq': 1}, {'flip_eq': 2, 'depth': 3}, {'pca_init': True},
                                   {'tails': 2, 'tail_epochs': 1, 'epochs': 3}, {'tta_gate': 0.5, 'tta_level': 2}])
def test_methods_train_save_and_reload(built, tmp_path, extra):
    r = airbench.train(**{**SMALL, **extra}, save=str(tmp_path / 'm'))
    probs = airbench.Predictor(r['saved']).predict(built['x_test'])
    assert abs((probs.argmax(1) == built['y_test']).mean() - r['test_acc']) <= 2 / len(probs)
    if 'tails' in extra:
        assert r['epochs_charged'] == 4 and r['history'][-1]['epoch'] == 'tails-averaged'
    if 'tta_gate' in extra:
        assert 1 <= r['views_per_image'] <= 7


def test_tied_depth_adds_no_convolution_weights():
    count = lambda **kw: airbench.param_counts(airbench.model((16, 32, 32), depth=2, **kw))['stored']
    assert count(tie=2) - count() == 2 * 2 * (16 + 32 + 32)          # (only the extra batch norms: a scale and a bias each)


def test_distillation_weighting(built, tmp_path):
    teacher = airbench.train(**SMALL, save=str(tmp_path / 'teacher'))
    for kind in ('disagree', 'hard'):
        r = airbench.train(**SMALL, kd_teacher=teacher['saved'], kd_weight=0.5, kd_weighting=kind)
        assert np.isfinite(r['history'][-1]['loss'])


def test_sealed_test_set_is_not_evaluated(built):
    r = airbench.train(**SMALL, test='sealed')
    assert 'test_acc' not in r and 'test_acc_notta' not in r and 0 <= r['val_acc'] <= 1
