from copy import deepcopy

import pytest
import torch
from torch import nn

from src.compression.trainable_maps import MapQuaternionConv2d
from src.models.quaternion_conv_torch import QuaternionConv2d
from src.trainable_map_repair import repair
from src.finetuning import finetune


def product(a, b):
    r, i, j, k = a.unbind(0)
    s, x, y, z = b.unbind(0)
    return torch.stack((r*s-i*x-j*y-k*z, r*x+i*s+j*z-k*y,
                        r*y-i*z+j*s+k*x, r*z+i*y-j*x+k*s))


@pytest.mark.parametrize("learn", [False, True])
def test_product_folding_and_four_component_updates(learn):
    torch.set_num_threads(2)
    torch.manual_seed(91)
    layer = QuaternionConv2d(2, 3).double()
    c = torch.randn(4, 3, 2, 3, 3, dtype=torch.float64)
    temporary = MapQuaternionConv2d(layer, c, learn)
    x = torch.randn(4, 8, 7, 7, dtype=torch.float64)
    torch.testing.assert_close(layer(x), temporary(x), rtol=0, atol=0)
    with torch.no_grad(): temporary.map_delta.normal_(0, .1)
    literal = layer.kernel.detach().clone()
    for o in range(3):
        for k in range(2):
            for v in range(2):
                literal[:, o, k] += product(temporary.map_delta[:, k, v, None, None], c[:, o, v])
    torch.testing.assert_close(temporary.effective_kernel(), literal, rtol=1e-12, atol=1e-12)
    before = {n: p.detach().clone() for n, p in temporary.named_parameters()}
    optimizer = torch.optim.AdamW(temporary.parameters(), lr=.001)
    temporary(x).square().mean().backward()
    for name, p in temporary.named_parameters():
        assert torch.isfinite(p.grad).all()
        assert (p.grad.flatten(1).norm(dim=1) > 0).all()
    optimizer.step()
    for name, p in temporary.named_parameters(): assert (p.detach()-before[name]).abs().max() > 0
    assert torch.equal(c, temporary.removed_kernel)
    assert ("map_delta" in dict(temporary.named_parameters())) is learn
    rng = torch.get_rng_state().clone()
    folded = temporary.folded()
    assert torch.equal(rng, torch.get_rng_state())
    assert type(folded) is QuaternionConv2d
    assert sum(p.numel() for p in folded.parameters()) == 216
    torch.testing.assert_close(temporary(x), folded(x), rtol=1e-12, atol=1e-12)


def test_interrupted_epoch_resume_replays_rng_optimizer_and_bn(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(55)
    initial = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                            nn.Dropout(.2), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))
    images = torch.randint(256, (12, 3, 8, 8), dtype=torch.uint8)
    labels = torch.arange(12) % 2
    tensors = {s: images for s in ("train", "validation", "check_calibration")}
    targets = {s: labels for s in tensors}
    cfg = {"epochs": 2, "batch_size": 4, "learning_rate": .001, "minimum_learning_rate": .00001,
           "weight_decay": .0001, "mixed_precision": False, "order_seed_offset": 80, "augment_seed_offset": 90}
    mean, std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
    baseline, history, counts, done = repair(deepcopy(initial), tensors, targets, mean, std, cfg, 22, tmp_path/"full.pt")
    original, original_history = finetune(deepcopy(initial), tensors, targets, mean, std, cfg, 22, ("validation", "check_calibration"))
    for k, v in original.state_dict().items(): assert torch.equal(v, baseline.state_dict()[k]), k
    stop = tmp_path/"STOP"
    def interrupt(step):
        if step == 4: stop.touch()
    _, partial, _, complete = repair(deepcopy(initial), tensors, targets, mean, std, cfg, 22, tmp_path/"resume.pt", stop, after_batch=interrupt)
    assert not complete and len(partial) == 1
    stop.unlink()
    resumed, resumed_history, resumed_counts, done2 = repair(deepcopy(initial), tensors, targets, mean, std, cfg, 22, tmp_path/"resume.pt")
    assert done and done2 and counts == resumed_counts == {"successful": 6, "attempted": 6}
    for k, v in baseline.state_dict().items(): assert torch.equal(v, resumed.state_dict()[k]), k
    for a, b in zip(history, resumed_history):
        assert {k: v for k, v in a.items() if k != "epoch_seconds"} == {k: v for k, v in b.items() if k != "epoch_seconds"}
