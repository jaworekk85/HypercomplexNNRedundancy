from copy import deepcopy

import pytest
import torch
from torch import nn

from scripts.paper_a_controls import cumulative_bn
from src.cnn_training import prepare_batch


def test_cumulative_bn_uses_only_statistics_and_keeps_dropout_inactive():
    torch.manual_seed(19)
    model = nn.Sequential(nn.Dropout2d(.9), nn.BatchNorm2d(3, momentum=.17)).eval()
    images = torch.randint(256, (16, 3, 4, 4), dtype=torch.uint8)
    mean, std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
    normalized = prepare_batch(images, mean, std)
    before = deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    report = cumulative_bn(model, images, mean, std, 4)
    assert all(report["checks"].values())
    assert torch.equal(rng, torch.get_rng_state())
    assert model[1].momentum == .17
    assert torch.equal(before["1.weight"], model[1].weight)
    assert torch.equal(before["1.bias"], model[1].bias)
    torch.testing.assert_close(model[1].running_mean, normalized.mean((0, 2, 3)))
    expected_var = torch.stack([part.var((0, 2, 3), unbiased=True) for part in normalized.split(4)]).mean(0)
    torch.testing.assert_close(model[1].running_var, expected_var)


def test_cumulative_bn_rejects_unequal_batches():
    model = nn.Sequential(nn.BatchNorm2d(3)).eval()
    with pytest.raises(ValueError, match="equal batches"):
        cumulative_bn(model, torch.zeros(7, 3, 4, 4), torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1), 4)
