"""Physical internal-channel ResNet pruning and same-victim compensation."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from src.compression.internal_merging import outgoing_addition
from src.compression.multilayer import SiteMoments, greedy_victims, site_maps
from src.compression.multisource import substitute_multi
from src.compression.quaternion_channels import component_indices
from src.compression.direct_consumer import q_normal_equations, solve_centered, theta, kernel_of, sse
from src.compression.structure_catalogue import collect, raw_moments
from src.models.quaternion_conv_torch import QuaternionConv2d


def moments(features, quaternion):
    x = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).double()
    return SiteMoments((x.T@x).numpy(), len(x), quaternion)


def choose(block, features, count, alpha):
    q = isinstance(block.conv2, QuaternionConv2d)
    m = moments(features, q)
    w = block.conv2.kernel if q else block.conv2.weight
    energy = w.detach().cpu().double().square().sum((0, 1, 3, 4) if q else (0, 2, 3)).numpy()
    mode = "quaternion_right" if q else "real"
    victims = greedy_victims(m, count, energy, alpha, mode)
    return victims, site_maps(m, victims, alpha, mode)


@torch.no_grad()
def compact(model, index, victims, maps=None):
    result = deepcopy(model).eval()
    source, target = model.blocks[index], result.blocks[index]
    old, consumer = source.conv1, source.conv2
    q = isinstance(old, QuaternionConv2d)
    width = old.out_channels
    victims = sorted(int(v) for v in victims)
    if len(set(victims)) != len(victims) or any(v < 0 or v >= width for v in victims): raise ValueError("Invalid victims")
    kept = sorted(set(range(width))-set(victims))
    if not kept: raise ValueError("Cannot remove all channels")
    channels = component_indices(kept, width) if q else kept
    if q:
        target.conv1 = QuaternionConv2d(old.in_channels, len(kept), 3, old.stride, old.padding).to(old.kernel)
        target.conv1.kernel.copy_(old.kernel[:, kept])
        target.conv2 = QuaternionConv2d(len(kept), consumer.out_channels, 3, consumer.stride, consumer.padding).to(consumer.kernel)
        target.conv2.kernel.copy_(consumer.kernel[:, :, kept])
    else:
        target.conv1 = nn.Conv2d(old.in_channels, len(kept), 3, old.stride, old.padding, bias=False).to(old.weight)
        target.conv1.weight.copy_(old.weight[kept])
        target.conv2 = nn.Conv2d(len(kept), consumer.out_channels, 3, consumer.stride, consumer.padding, bias=False).to(consumer.weight)
        target.conv2.weight.copy_(consumer.weight[:, kept])
    bn = source.bn1
    target.bn1 = nn.BatchNorm2d(len(channels), eps=bn.eps, momentum=bn.momentum).to(bn.weight)
    for name in ("weight", "bias", "running_mean", "running_var"):
        getattr(target.bn1, name).copy_(getattr(bn, name)[channels])
    target.bn1.num_batches_tracked.copy_(bn.num_batches_tracked)
    if maps is not None:
        if sorted(row["victim"] for row in maps) != victims or any(row["kept"] != kept for row in maps): raise ValueError("Wrong map support")
        for row in maps:
            if q:
                kernel = consumer.kernel.detach().cpu().double().numpy()
                for pos, matrix in enumerate(row["matrices"]):
                    addition, diagnostic = outgoing_addition(kernel, row["victim"], matrix)
                    if diagnostic["discarded_energy_fraction"] > 1e-20: raise ValueError("Non-Hamilton fold")
                    target.conv2.kernel[:, :, pos].add_(torch.as_tensor(addition, device=consumer.kernel.device, dtype=consumer.kernel.dtype))
            else:
                coeff = torch.as_tensor(row["coefficients"], device=consumer.weight.device, dtype=consumer.weight.dtype)
                target.conv2.weight.add_(coeff[None, :, None, None]*consumer.weight[:, row["victim"]][:, None])
    return result.eval(), kept


@torch.no_grad()
def fold_error(source, result, index, maps, images):
    q = isinstance(source.blocks[index].conv2, QuaternionConv2d)
    handle = source.blocks[index].conv2.register_forward_pre_hook(lambda module, args: (substitute_multi(args[0], maps, q),))
    try: return float((source(images)-result(images)).abs().max())
    finally: handle.remove()


def refit(reference, folded, index, kept, fit, check, alpha):
    """Ridge to pre-BN consumer outputs, all other weights fixed."""
    q = isinstance(reference.blocks[index].conv2, QuaternionConv2d)
    width = reference.blocks[index].conv1.out_channels
    channels = component_indices(kept, width) if q else kept
    packs = {"fit": fit, "check": check}
    values = {}
    for name, (x, y) in packs.items():
        g, b, squares, count = raw_moments(x[:, channels], y)
        if q: g, b = map(torch.from_numpy, q_normal_equations(g.numpy(), b.numpy(), reference.blocks[index].outputs))
        values[name] = (g.numpy(), b.numpy(), squares, count)
    layer = folded.blocks[index].conv2
    if q: prior = theta(layer.kernel.detach().cpu().double().numpy())
    else: prior = layer.weight.detach().cpu().double().permute(1, 2, 3, 0).reshape(-1, layer.out_channels).numpy()
    g, b, squares, _ = values["fit"]
    fitted, audit = solve_centered(g, b, squares, prior, alpha)
    result = deepcopy(folded)
    with torch.no_grad():
        if q: result.blocks[index].conv2.kernel.copy_(torch.as_tensor(kernel_of(fitted, tuple(layer.kernel.shape)), device=layer.kernel.device, dtype=layer.kernel.dtype))
        else: result.blocks[index].conv2.weight.copy_(torch.as_tensor(fitted.reshape(layer.in_channels, 3, 3, layer.out_channels).transpose(3, 0, 1, 2), device=layer.weight.device, dtype=layer.weight.dtype))
    for name, (gram, rhs, yy, count) in values.items():
        audit[name+"_prior_mse"] = sse(gram, rhs, yy, prior)/count
        audit[name+"_direct_mse"] = sse(gram, rhs, yy, fitted)/count
    return result.eval(), audit
