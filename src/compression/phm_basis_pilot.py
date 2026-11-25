"""Physical PHM compaction and fit-only basis-adaptation diagnostics.

Known PHM/reconstruction primitives: ZhangEtAl2021, Grassucci2024,
He2017ChannelPruning. No new-algebra, solver, or novelty claim.
"""
from copy import deepcopy
import time

import torch
from torch import nn

from src.compression.phm_activation_fit import normalized, objective
from src.compression.quaternion_channels import component_indices
from src.models.phm_conv import PHMConv2d
from src.trainable_map_repair import atomic_save


def kept_groups(block, retained_fraction):
    layer, bn = block.conv1, block.bn1
    if not isinstance(layer, PHMConv2d) or layer.n != 4:
        raise ValueError("This pilot requires PHM4 producers")
    width = layer.out_channels // 4
    count = max(1, round(width * retained_fraction))
    if not 0 < retained_fraction <= 1:
        raise ValueError("Invalid retained fraction")
    weight = layer.effective_kernel().detach().double().cpu()
    scale = bn.weight.detach().double().cpu() / torch.sqrt(bn.running_var.detach().double().cpu() + bn.eps)
    energy = (weight.square().sum((1, 2, 3)) * scale.square()).reshape(4, width).sum(0)
    order = sorted(range(width), key=lambda k: (-float(energy[k]), k))
    return sorted(order[:count])


@torch.no_grad()
def compact_block(block, kept):
    """Remove a shared within-component group from producer, BN and consumer."""
    producer, consumer = block.conv1, block.conv2
    width = producer.out_channels // 4
    if (not kept or list(kept) != sorted(set(kept)) or kept[0] < 0 or kept[-1] >= width
            or producer.n != 4 or consumer.n != 4):
        raise ValueError("Invalid PHM4 groups")
    result = deepcopy(block)
    result.conv1 = PHMConv2d(producer.in_channels, 4 * len(kept), producer.cores.shape[-1],
                            n=4, rank=producer.rank, stride=producer.stride, padding=producer.padding).to(producer.cores)
    result.conv1.basis.copy_(producer.basis)
    result.conv1.cores.copy_(producer.cores[:, kept])
    result.conv2 = PHMConv2d(4 * len(kept), consumer.out_channels, consumer.cores.shape[-1],
                            n=4, rank=consumer.rank, stride=consumer.stride, padding=consumer.padding).to(consumer.cores)
    result.conv2.basis.copy_(consumer.basis)
    result.conv2.cores.copy_(consumer.cores[:, :, kept])
    channels = component_indices(kept, width)
    bn = block.bn1
    result.bn1 = nn.BatchNorm2d(4 * len(kept), eps=bn.eps, momentum=bn.momentum).to(bn.weight)
    for name in ("weight", "bias", "running_mean", "running_var"):
        getattr(result.bn1, name).copy_(getattr(bn, name)[channels])
    result.bn1.num_batches_tracked.copy_(bn.num_batches_tracked)
    return result.eval()


def initialize_consumer(layer, policy, seed):
    result = deepcopy(layer).double().cpu()
    if policy not in ("preserve", "adapt", "reinitialize"):
        raise ValueError(policy)
    if policy == "reinitialize":
        generator = torch.Generator().manual_seed(seed)
        basis = torch.linalg.qr(torch.randn(16, 4, generator=generator, dtype=torch.float64)).Q.T
        kernel = result.effective_kernel().detach()
        o, i, h, w = kernel.shape
        arranged = kernel.reshape(4, o//4, 4, i//4, h, w).permute(0, 2, 1, 3, 4, 5).reshape(16, -1)
        with torch.no_grad():
            # Orthonormal rows make this the exact least-squares projection;
            # avoid pivoted-solver roundoff varying between identical starts.
            result.cores.copy_((basis @ arranged).reshape_as(result.cores))
            result.basis.copy_(basis.reshape_as(result.basis))
    result.basis.requires_grad_(policy != "preserve")
    return result


def set_repair_policy(model, policy, blocks):
    if policy not in ("preserve", "adapt", "reinitialize", "scratch"):
        raise ValueError(policy)
    for p in model.parameters():
        p.requires_grad_(True)
    for layer in model.modules():
        if isinstance(layer, PHMConv2d):
            layer.basis.requires_grad_(policy == "scratch")
    if policy in ("adapt", "reinitialize"):
        for index in blocks:
            model.blocks[index].conv2.basis.requires_grad_(True)


def counts(model):
    return {"stored_real_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_real_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_real_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
            "buffer_elements": sum(b.numel() for b in model.buffers())}


def update_audit(model, loss_function):
    probe = deepcopy(model)
    before = {n: p.detach().clone() for n,p in probe.named_parameters()}
    optimizer = torch.optim.SGD([p for p in probe.parameters() if p.requires_grad], lr=.001)
    loss = loss_function(probe)
    loss.backward()
    rows = {}
    for name, p in probe.named_parameters():
        if p.requires_grad and (p.grad is None or not bool(torch.isfinite(p.grad).all())):
            raise AssertionError(f"Missing/nonfinite trainable gradient: {name}")
        if not p.requires_grad and p.grad is not None:
            raise AssertionError(f"Frozen factor received a gradient: {name}")
        rows[name] = {"trainable": p.requires_grad, "gradient_norm": float(p.grad.norm()) if p.grad is not None else None}
    optimizer.step()
    for name, p in probe.named_parameters():
        delta = float((p.detach()-before[name]).abs().max())
        rows[name]["max_abs_update"] = delta
        if not p.requires_grad and delta != 0:
            raise AssertionError(f"Frozen factor changed: {name}")
        if p.requires_grad and name.endswith("basis") and delta == 0:
            raise AssertionError(f"Trainable basis did not update: {name}")
    if not any(r["max_abs_update"] > 0 for r in rows.values() if r["trainable"]):
        raise AssertionError("No trainable parameter updated")
    return {"loss": float(loss.detach()), "parameters": rows, "passed": True}


def fit_consumer(layer, moments, config, folder, stop, progress=None, after_step=None):
    """Resume exact full-batch Adam; select the best FIT objective, never CHECK."""
    folder.mkdir(parents=True, exist_ok=True)
    pack, scale = normalized(moments)
    optimizer = torch.optim.Adam([p for p in layer.parameters() if p.requires_grad], lr=config["learning_rate"])
    checkpoint = folder / "local_resume.pt"
    initial = float(objective(layer, pack, scale).detach())
    state = {"initial": initial, "best_loss": initial, "best_step": 0,
             "step": 0, "best": deepcopy(layer.state_dict())}
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["config"] != config or saved["trainable_names"] != [n for n,p in layer.named_parameters() if p.requires_grad]:
            raise ValueError("Local resume configuration changed")
        layer.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]

    def save():
        atomic_save({"model": layer.state_dict(), "optimizer": optimizer.state_dict(),
                     "state": state, "config": config,
                     "trainable_names": [n for n,p in layer.named_parameters() if p.requires_grad]}, checkpoint)

    if not checkpoint.exists():
        save()
    start = time.perf_counter()
    for step in range(state["step"] + 1, config["steps"] + 1):
        if stop.exists():
            save()
            return None
        optimizer.zero_grad(set_to_none=True)
        loss = objective(layer, pack, scale)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite local fit loss")
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in layer.parameters() if p.requires_grad):
            raise RuntimeError("Missing/nonfinite trainable factor gradient")
        optimizer.step()
        value = float(objective(layer, pack, scale).detach())
        if not torch.isfinite(torch.tensor(value)):
            raise RuntimeError("Nonfinite post-update objective")
        state["step"] = step
        if value < state["best_loss"]:
            state.update(best_loss=value, best_step=step, best=deepcopy(layer.state_dict()))
        if step % config["checkpoint_interval"] == 0 or step == config["steps"]:
            save()
            if progress is not None:
                progress(step, state["best_loss"])
        if after_step is not None:
            after_step(step)
    final = float(objective(layer, pack, scale).detach())
    layer.load_state_dict(state["best"])
    return {"initial_normalized_fit_mse": state["initial"], "best_normalized_fit_mse": state["best_loss"],
            "best_step": state["best_step"], "final_normalized_fit_mse": final,
            "attempted_steps": state["step"], "current_attempt_seconds": time.perf_counter() - start}


def change_audit(before, after):
    """Raw factor movement is an implementation check, not a gauge-invariant score."""
    changes = {"basis_max_abs_change": {}, "effective_kernel_relative_change": {}}
    old = dict(before.named_modules())
    for name, layer in after.named_modules():
        if isinstance(layer, PHMConv2d):
            original = old[name]
            changes["basis_max_abs_change"][name] = float((layer.basis.detach().cpu() - original.basis.detach().cpu()).abs().max())
            w0, w1 = original.effective_kernel().detach().cpu(), layer.effective_kernel().detach().cpu()
            changes["effective_kernel_relative_change"][name] = float((w1-w0).norm()/w0.norm().clamp_min(1e-12))
    return changes
