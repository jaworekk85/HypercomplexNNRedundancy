from copy import deepcopy

import pytest
import torch
from torch import nn

from src.cnn_training import first_update_audit
from src.models.cifar_resnet import BasicBlock, CifarResNet20
from src.models.quaternion_conv_torch import QuaternionConv2d
from src.resnet_training import train


@pytest.mark.parametrize("kind", ["real", "quaternion"])
def test_resnet_shape_all_gradients_and_physical_internal_widths(kind):
    torch.set_num_threads(2)
    torch.manual_seed(21)
    config = {"real_widths": [2, 4, 8], "quaternion_widths": [1, 2, 4]}
    model = CifarResNet20(kind, **config)
    x, y = torch.randn(6, 3, 16, 16), torch.arange(6)
    assert model(x).shape == (6, 10)
    audit = first_update_audit(model, x, y, torch.optim.SGD(model.parameters(), lr=.01))
    assert all(audit["checks"].values())
    expected = 19 if kind == "quaternion" else 0
    assert sum(isinstance(m, QuaternionConv2d) for m in model.modules()) == expected
    widths = [max(1, block.conv1.out_channels//2) for block in model.blocks]
    compact = CifarResNet20(kind, **config, hidden_widths=widths)
    assert compact(x).shape == (6, 10)
    assert sum(p.numel() for p in compact.parameters()) < sum(p.numel() for p in model.parameters())
    full_widths = [block.conv1.out_channels for block in model.blocks]
    replica = CifarResNet20(kind, **config, hidden_widths=full_widths).eval()
    replica.load_state_dict(model.state_dict())
    torch.testing.assert_close(model.eval()(x), replica(x), atol=0, rtol=0)


def test_option_A_padding_preserves_quaternion_components():
    block = BasicBlock("quaternion", 2, 4, 2)
    x = torch.arange(8*4*4).reshape(1, 8, 4, 4).float()
    actual = block.shortcut(x).reshape(1, 4, 4, 2, 2)
    torch.testing.assert_close(actual[:, :, 1:3], x[:, :, ::2, ::2].reshape(1, 4, 2, 2, 2), rtol=0, atol=0)
    assert torch.count_nonzero(actual[:, :, [0, 3]]) == 0


def test_sgd_resumption_replays_momentum_bn_rng_and_best_selection(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(8)
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.Dropout(.2),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))
    x = torch.randint(256, (12, 3, 8, 8), dtype=torch.uint8)
    y = torch.arange(12) % 2
    tensors, labels = {"train": x, "validation": x}, {"train": y, "validation": y}
    mean, std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
    cfg = {"seed": 9, "epochs": 2, "batch_size": 4, "learning_rate": .01, "warmup_epochs": 1,
           "warmup_start_lr": .005, "milestones": [2], "gamma": .1, "momentum": .9, "weight_decay": .0001,
           "mixed_precision": False, "order_seed_offset": 40, "augment_seed_offset": 50}
    full, a, done = train(deepcopy(model), tensors, labels, mean, std, cfg, tmp_path/"full.pt")
    stop = tmp_path/"STOP"
    def interrupt(step):
        if step == 4: stop.touch()
    _, partial, completed = train(deepcopy(model), tensors, labels, mean, std, cfg, tmp_path/"resume.pt", stop, after_batch=interrupt)
    assert not completed and len(partial["history"]) == 1
    stop.unlink()
    resumed, b, done2 = train(deepcopy(model), tensors, labels, mean, std, cfg, tmp_path/"resume.pt")
    assert done and done2 and a["counts"] == b["counts"] == {"attempted": 6, "successful": 6}
    assert a["selected_epoch"] == b["selected_epoch"] and a["selected_validation"] == b["selected_validation"]
    for name, value in full.state_dict().items(): assert torch.equal(value, resumed.state_dict()[name]), name
    for one, two in zip(a["history"], b["history"]):
        assert {k: v for k, v in one.items() if k != "epoch_seconds"} == {k: v for k, v in two.items() if k != "epoch_seconds"}
