"""Fixed-budget Q/dense/PHM catalogue on internal ResNet channels.

Local fits/scores are surrogates, not a global classification guarantee.
Budgeted choice is standard multiple-choice knapsack (ALDS/SPDY context).
"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.compression.direct_consumer import q_normal_equations, theta, kernel_of
from src.compression.quaternion_channels import component_indices
from src.models.cifar_resnet import CifarResNet20
from src.models.phm_conv import PHMConv2d
from src.models.quaternion_conv_torch import QuaternionConv2d, hamilton_kernel


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


def mutable_count(block):
    return sum(parameter_count(m) for m in (block.conv1, block.bn1, block.conv2))


@torch.no_grad()
def narrow_producer(block, kept):
    """Only the internal Q producer is narrowed; residual outputs never change."""
    result = deepcopy(block)
    old = block.conv1
    if not isinstance(old, QuaternionConv2d): raise ValueError("First catalogue scope is Q-parent ResNet")
    if not kept or sorted(set(kept)) != list(kept) or kept[-1] >= old.out_channels or kept[0] < 0: raise ValueError("Invalid Q channels")
    result.conv1 = QuaternionConv2d(old.in_channels, len(kept), old.kernel.shape[-1], old.stride, old.padding).to(old.kernel)
    result.conv1.kernel.copy_(old.kernel[:, kept])
    channels = component_indices(kept, old.out_channels)
    bn = block.bn1
    result.bn1 = nn.BatchNorm2d(4*len(kept), eps=bn.eps, momentum=bn.momentum).to(bn.weight)
    for name in ("weight", "bias", "running_mean", "running_var"):
        getattr(result.bn1, name).copy_(getattr(bn, name)[channels])
    result.bn1.num_batches_tracked.copy_(bn.num_batches_tracked)
    return result.eval()


@torch.no_grad()
def collect(model, index, images, batch_size=32):
    """Actual post-ReLU consumer inputs and pre-BN outputs, streamed to CPU."""
    model.eval()
    inputs, outputs = [], []
    def hook(module, args, result):
        inputs.append(args[0].detach().cpu())
        outputs.append(result.detach().cpu())
    handle = model.blocks[index].conv2.register_forward_hook(hook)
    try:
        for x in images.split(batch_size): model(x)
    finally: handle.remove()
    return torch.cat(inputs), torch.cat(outputs)


def raw_moments(inputs, targets, batch_size=16):
    channels = inputs.shape[1]
    gram = torch.zeros(channels*9, channels*9, dtype=torch.float64)
    rhs = torch.zeros(channels*9, targets.shape[1], dtype=torch.float64)
    squares = 0.
    for start in range(0, len(inputs), batch_size):
        x = F.unfold(inputs[start:start+batch_size].double(), 3, padding=1).transpose(1, 2).reshape(-1, channels*9)
        y = targets[start:start+batch_size].double().permute(0, 2, 3, 1).reshape(-1, targets.shape[1])
        gram.add_(x.T@x); rhs.add_(x.T@y); squares += float(y.square().sum())
    return gram, rhs, squares, targets.numel()


def weight_matrix(weight):
    return weight.permute(1, 2, 3, 0).reshape(-1, weight.shape[0]).double()


def mse(weight, moments):
    g, b, yy, count = moments
    w = weight_matrix(weight)
    value = float(yy-2*(w*b).sum()+(w*(g@w)).sum())
    if value < -1e-7*max(1., yy): raise AssertionError("Negative calibration SSE")
    return max(0., value)/count


def ridge(g, b, prior, penalty):
    update = torch.linalg.solve(g+penalty*torch.eye(len(g), dtype=g.dtype), b-g@prior)
    result = prior+update
    if not torch.isfinite(result).all(): raise ValueError("Nonfinite calibration solution")
    return result


def candidates(block, fit, widths, dimensions, alpha):
    """Fit on FIT ONLY. Held-out images never select weights/types/widths."""
    if not isinstance(block.conv2, QuaternionConv2d): raise ValueError("Expected Q source consumer")
    original = deepcopy(block).cpu().eval()
    full = hamilton_kernel(original.conv2.kernel).detach().double()
    # Deterministic BN-folded producer norm ordering, no validation-based ranking.
    p = original.conv1.kernel.detach().double().square().sum((0, 2, 3, 4))
    scale = (original.bn1.weight.detach().double()/torch.sqrt(original.bn1.running_var.double()+original.bn1.eps)).reshape(4, -1)
    energy = p*scale.square().sum(0)
    order = sorted(range(len(p)), key=lambda k: (-float(energy[k]), k))
    result, skipped = [], []
    for width in sorted(set(widths)):
        kept = sorted(order[:width])
        if width > len(order) or width < 1: raise ValueError("Invalid catalogue width")
        scalar = component_indices(kept, len(order))
        moments = raw_moments(fit[0][:, scalar], fit[1])
        g, b, _, _ = moments
        lam = alpha*float(g.diag().mean())
        if not math.isfinite(lam) or lam <= 0: raise ValueError("Degenerate input Gram")
        prior = full[:, scalar]
        dense_values = ridge(g, b, weight_matrix(prior), lam)
        dense_weight = dense_values.reshape(4*width, 3, 3, full.shape[0]).permute(3, 0, 1, 2)
        qg, qb = q_normal_equations(g.numpy(), b.numpy(), original.conv2.out_channels)
        qprior = original.conv2.kernel.detach().double()[:, :, kept]
        # ||Hamilton(deltaQ)||_F^2=4||deltaQ||^2: same dense-kernel ridge.
        qvalues = ridge(torch.from_numpy(qg), torch.from_numpy(qb), torch.from_numpy(theta(qprior.numpy())), 4*lam)
        qlayer = QuaternionConv2d(width, original.conv2.out_channels).double()
        with torch.no_grad(): qlayer.kernel.copy_(torch.from_numpy(kernel_of(qvalues.numpy(), tuple(qprior.shape))))
        dense = nn.Conv2d(4*width, full.shape[0], 3, padding=1, bias=False).double()
        with torch.no_grad(): dense.weight.copy_(dense_weight)
        layers = [("q", None, qlayer, hamilton_kernel(qlayer.kernel)), ("real", 1, dense, dense.weight)]
        for n in dimensions:
            if (4*width) % n or full.shape[0] % n:
                skipped.append({"width": width, "n": n, "reason": "n does not divide scalar inputs/outputs"}); continue
            phm = PHMConv2d.from_kernel(dense_weight, n)
            layers.append(("phm", n, phm, phm.effective_kernel()))
        for family, n, layer, weight in layers:
            candidate = narrow_producer(original, kept)
            candidate.conv2 = layer.float()
            # Score the ACTUAL float32 artifact, not only its float64 fit.
            actual = candidate.conv2
            actual_weight = hamilton_kernel(actual.kernel) if family == "q" else actual.effective_kernel() if family == "phm" else actual.weight
            score = mse(actual_weight.detach(), moments)
            normalizer = max(float(fit[1].double().square().mean()), 1e-12)
            result.append({"id": f"{family}_n{n or 4}_k{width}", "family": family, "n": n or 4, "rank": n if family == "phm" else None,
                           "kept": kept, "cost": mutable_count(candidate), "fit_mse": score, "error": score/normalizer,
                           "block": candidate, "dense_oracle_fit_mse": mse(dense_weight, moments),
                           "frobenius_to_dense_oracle": float((actual_weight.double()-dense_weight).square().sum()),
                           "weight_error": float((actual_weight.double()-dense_weight).square().sum()/dense_weight.square().sum().clamp_min(1e-12))})
    return result, skipped


def select(catalogues, budget, metric="error"):
    """Exact Pareto-frontier multiple-choice knapsack for the given local proxy."""
    frontier = {0: (0., [])}
    for catalogue in catalogues:
        following = {}
        for spent, (error, choices) in frontier.items():
            for item in sorted(catalogue, key=lambda row: row["id"]):
                cost, value = spent+item["cost"], error+item[metric]
                if item["cost"] < 0 or not math.isfinite(item[metric]) or item[metric] < 0: raise ValueError("Invalid candidate")
                if cost > budget: continue
                proposal = (value, choices+[item["id"]])
                if cost not in following or proposal < following[cost]: following[cost] = proposal
        frontier, best = {}, float("inf")
        for cost, proposal in sorted(following.items()):
            if proposal[0] < best:
                frontier[cost] = proposal; best = proposal[0]
        if not frontier: raise ValueError("No feasible structure assignment under actual budget")
    cost, (value, choices) = min(frontier.items(), key=lambda row: (row[1][0], row[0], row[1][1]))
    return {"cost": cost, "proxy_error": value, "choices": choices, "budget": budget}


def architecture(model, config):
    consumers = []
    for block in model.blocks:
        layer = block.conv2
        consumers.append({"family": "q"} if isinstance(layer, QuaternionConv2d) else
                         {"family": "phm", "n": layer.n, "rank": layer.rank} if isinstance(layer, PHMConv2d) else {"family": "real"})
    return {"kind": model.kind, "real_widths": config["real_widths"], "quaternion_widths": config["quaternion_widths"],
            "classes": config["classes"], "hidden_widths": [b.conv1.out_channels for b in model.blocks], "consumers": consumers}


def restore(spec):
    model = CifarResNet20(spec["kind"], spec["real_widths"], spec["quaternion_widths"], spec["classes"], spec["hidden_widths"])
    components = 4 if spec["kind"] == "quaternion" else 1
    for block, definition in zip(model.blocks, spec["consumers"]):
        if definition["family"] == "q": continue
        inputs, outputs = components*block.conv1.out_channels, components*block.outputs
        block.conv2 = PHMConv2d(inputs, outputs, n=definition["n"], rank=definition["rank"]) if definition["family"] == "phm" else nn.Conv2d(inputs, outputs, 3, padding=1, bias=False)
    return model
