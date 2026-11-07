import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_weights

from src.compression.bn_magnitude import bn_group_scores, bn_removal_order
from src.models.quaternion_conv_torch import QuaternionConv2d, hamilton_kernel


def test_group_norm_matches_official_folding_and_forward_pass():
    torch.manual_seed(313)
    conv = QuaternionConv2d(2, 3, bias=True).double().eval()
    bn = torch.nn.BatchNorm2d(12, eps=.03).double().eval()
    with torch.no_grad():
        bn.weight.copy_(torch.linspace(-2, 3, 12))
        bn.bias.normal_()
        bn.running_mean.normal_()
        bn.running_var.copy_(torch.linspace(.01, 4, 12))
        conv.bias.normal_()
    weight, bias = fuse_conv_bn_weights(hamilton_kernel(conv.kernel), conv.bias, bn.running_mean,
                                       bn.running_var, bn.eps, bn.weight, bn.bias)
    expected = weight.reshape(4, 3, 8, 3, 3).square().sum(dim=(0, 2, 3, 4)).sqrt()
    actual = torch.tensor(bn_group_scores(conv, bn)["bn_folded_group_l2"], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    inputs = torch.randn(4, 8, 9, 9, dtype=torch.float64)
    torch.testing.assert_close(bn(conv(inputs)), F.conv2d(inputs, weight, bias, padding=1), atol=1e-12, rtol=1e-12)


def test_effective_norm_preserves_ranking_under_bn_compensated_rescaling():
    conv = QuaternionConv2d(1, 2).double().eval()
    bn = torch.nn.BatchNorm2d(8, eps=.01).double().eval()
    with torch.no_grad():
        conv.kernel[:, 0].fill_(1)
        conv.kernel[:, 1].fill_(2)
        bn.running_var.fill_(1)
    before = bn_group_scores(conv, bn)
    assert bn_removal_order(before["bn_folded_group_l2"]) == [0, 1]
    with torch.no_grad():
        conv.kernel[:, 0].mul_(10)
        channels = [0, 2, 4, 6]
        bn.running_var[channels] = 100 * (bn.running_var[channels] + bn.eps) - bn.eps
    after = bn_group_scores(conv, bn)
    assert bn_removal_order(after["raw_kernel_l2"]) == [1, 0]
    torch.testing.assert_close(torch.tensor(after["bn_folded_group_l2"]), torch.tensor(before["bn_folded_group_l2"]))
    assert bn_removal_order(after["bn_folded_group_l2"]) == [0, 1]


def test_zero_gamma_and_invalid_bn_are_handled_explicitly():
    conv = QuaternionConv2d(1, 2).eval()
    bn = torch.nn.BatchNorm2d(8).eval()
    with torch.no_grad():
        bn.weight[[0, 2, 4, 6]] = 0
    assert bn_group_scores(conv, bn)["bn_folded_group_l2"][0] == 0
    bn.train()
    with pytest.raises(ValueError, match="evaluation-mode"):
        bn_group_scores(conv, bn)
