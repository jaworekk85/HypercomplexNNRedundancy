import pytest
import torch

from src.compression.resnet_core import choose, compact, fold_error, refit
from src.compression.structure_catalogue import collect
from src.models.cifar_resnet import CifarResNet20


@pytest.mark.parametrize("kind", ["real", "quaternion"])
def test_residual_pruning_fold_and_direct_fit(kind):
    torch.set_num_threads(2)
    torch.manual_seed(34)
    model = CifarResNet20(kind, real_widths=[4, 8, 16], quaternion_widths=[2, 4, 8]).eval()
    x = torch.randn(8, 3, 8, 8)
    unmodified, _ = compact(model, 0, [])
    torch.testing.assert_close(model(x), unmodified(x), atol=0, rtol=0)
    fit = collect(model, 0, x[:6], 3)
    check = collect(model, 0, x[6:], 2)
    victims, maps = choose(model.blocks[0], fit[0], 1, .001)
    folded, kept = compact(model, 0, victims, maps)
    assert fold_error(model, folded, 0, maps, x) < 1e-5
    pruned, _ = compact(model, 0, victims)
    assert sum(p.numel() for p in folded.parameters()) < sum(p.numel() for p in model.parameters())
    assert sum(p.numel() for p in folded.parameters()) == sum(p.numel() for p in pruned.parameters())
    direct, audit = refit(model, folded, 0, kept, fit, check, .001)
    assert audit["fit_direct_mse"] <= audit["fit_prior_mse"]+1e-8
    assert direct(x).shape == (8, 10)
    for name, value in folded.state_dict().items():
        if name != "blocks.0.conv2."+("kernel" if kind == "quaternion" else "weight"):
            assert torch.equal(value, direct.state_dict()[name]), name
