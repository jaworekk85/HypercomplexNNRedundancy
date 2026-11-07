import pytest
import torch
from torch import nn

from src.cnn_training import first_update_audit, seed_everything
from src.models.cifar_cnn_torch import CifarCNN
from src.models.cifar_no_bn import make_no_bn


@pytest.mark.parametrize("kind", ["quaternion", "real"])
def test_removing_bn_preserves_other_initialization_and_trains(kind):
    config = {"quaternion_widths": [2, 4, 8], "real_widths": [4, 8, 16], "dropout": [0., 0., 0.]}
    seed_everything(73)
    original = CifarCNN(kind, config["quaternion_widths"], config["real_widths"], config["dropout"])
    seed_everything(73)
    model = make_no_bn(config, kind)
    assert not any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules())
    assert len(list(model.buffers())) == 0
    old = dict(original.named_parameters())
    assert all(torch.equal(old[name], value) for name, value in model.named_parameters())
    images = torch.randn(8, 3, 32, 32)
    labels = torch.arange(8)
    audit = first_update_audit(model, images, labels, torch.optim.AdamW(model.parameters(), lr=.001))
    assert all(audit["checks"].values())
    model.train()
    train = model(images)
    model.eval()
    torch.testing.assert_close(train, model(images))
