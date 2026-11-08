"""Full-batch local activation fitting, with identical objectives across starts."""
from copy import deepcopy

import torch

from src.models.phm_conv import PHMConv2d
from src.models.quaternion_conv_torch import QuaternionConv2d, hamilton_kernel


def effective(layer):
    return hamilton_kernel(layer.kernel) if isinstance(layer, QuaternionConv2d) else layer.effective_kernel()


def make_initialization(q_layer, svd_layer, name, seed):
    if name == "q_control":
        return deepcopy(q_layer).double()
    if name == "from_q":
        return PHMConv2d.from_kernel(hamilton_kernel(q_layer.kernel.detach()).double(), 4)
    if name == "from_svd":
        return deepcopy(svd_layer).double()
    if name != "random":
        raise ValueError("Unknown initialization")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return PHMConv2d(svd_layer.in_channels, svd_layer.out_channels, n=4).double()


def objective(layer, normalized_moments, normalizer):
    g, b, yy = normalized_moments
    weight = effective(layer)
    matrix = weight.permute(1, 2, 3, 0).reshape(-1, weight.shape[0])
    return (yy - 2*(matrix*b).sum() + (matrix*(g@matrix)).sum()) / normalizer


def normalized(moments):
    g, b, yy, count = moments
    return (g/count, b/count, yy/count), max(yy/count, 1e-12)


def gradient_audit(layer, moments):
    pack, scale = normalized(moments)
    probe = deepcopy(layer)
    optimizer = torch.optim.Adam(probe.parameters(), lr=.01)
    before = {name: p.detach().clone() for name, p in probe.named_parameters()}
    loss = objective(probe, pack, scale)
    loss.backward()
    grads = {name: p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.abs().sum() > 0)
             for name, p in probe.named_parameters()}
    optimizer.step()
    changes = {name: bool((p.detach() != before[name]).any()) for name, p in probe.named_parameters()}
    if not all(grads.values()) or not all(changes.values()):
        raise AssertionError("A factor has no finite gradient or update")
    return {"finite_nonzero_gradients": grads, "updated": changes}
