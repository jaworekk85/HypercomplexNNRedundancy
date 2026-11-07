from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from src.cnn_training import prepare_batch
from src.evaluation.routing_hardening import recalibrate_bn_layers


class SmallBNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.upstream = nn.BatchNorm2d(3)
        self.first = nn.BatchNorm2d(3)
        self.dropout = nn.Dropout2d(.9)
        self.conv = nn.Conv2d(3, 2, 1)
        self.second = nn.BatchNorm2d(2)

    def forward(self, inputs):
        values = self.first(self.upstream(inputs))
        return self.second(self.conv(self.dropout(values)))


def inputs():
    images = torch.randint(256, (13, 3, 4, 4), generator=torch.Generator().manual_seed(11), dtype=torch.uint8)
    return images, torch.full((1, 3, 1, 1), .5), torch.full((1, 3, 1, 1), .2)


def test_layerwise_refit_matches_full_data_moments_without_weight_or_rng_changes():
    torch.manual_seed(7)
    model = SmallBNModel().eval()
    images, mean, std = inputs()
    before = deepcopy(model.state_dict())
    with torch.no_grad():
        first_input = model.upstream(prepare_batch(images, mean, std))
    rng = torch.get_rng_state().clone()
    result = recalibrate_bn_layers(model, images, mean, std, ["first", "second"], batch_size=5)
    assert all(result["checks"].values())
    assert torch.equal(torch.get_rng_state(), rng)  # dropout must not run
    expected_mean = first_input.double().mean(dim=(0, 2, 3))
    expected_var = first_input.double().var(dim=(0, 2, 3), correction=1)
    torch.testing.assert_close(model.first.running_mean.double(), expected_mean, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(model.first.running_var.double(), expected_var, rtol=1e-6, atol=1e-7)
    with torch.no_grad():
        second_input = model.conv(model.first(first_input))
    torch.testing.assert_close(model.second.running_mean.double(), second_input.double().mean(dim=(0, 2, 3)), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(model.second.running_var.double(), second_input.double().var(dim=(0, 2, 3), correction=1), rtol=1e-6, atol=1e-7)
    changed = {name for name, value in model.state_dict().items() if not torch.equal(before[name], value)}
    assert changed == {"first.running_mean", "first.running_var", "second.running_mean", "second.running_var"}
    assert all(row["observations_per_channel"] == 13*4*4 for row in result["layers"])


def test_refit_is_independent_of_minibatch_partition():
    torch.manual_seed(3)
    a = SmallBNModel()
    b = deepcopy(a)
    images, mean, std = inputs()
    recalibrate_bn_layers(a, images, mean, std, ["first", "second"], batch_size=4)
    recalibrate_bn_layers(b, images, mean, std, ["first", "second"], batch_size=13)
    for name in a.state_dict():
        torch.testing.assert_close(a.state_dict()[name], b.state_dict()[name], rtol=1e-6, atol=1e-7)


def test_refit_rejects_invalid_layer_or_empty_input():
    model = SmallBNModel()
    images, mean, std = inputs()
    with pytest.raises(ValueError, match="Not a BN"):
        recalibrate_bn_layers(model, images, mean, std, ["conv"])
    with pytest.raises(ValueError, match="Need training"):
        recalibrate_bn_layers(model, images[:0], mean, std, ["first"])
