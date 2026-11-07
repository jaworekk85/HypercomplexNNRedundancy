"""Interventions on BN buffers, without training any model parameters."""

import math

import torch
from torch import nn

from src.cnn_training import prepare_batch


@torch.no_grad()
def recalibrate_bn_layers(model, images, mean, std, names, batch_size=256):
    """Refit selected BN input moments in forward order using supplied images.

    Every module stays in evaluation mode, including dropout and all BNs.
    Refit each BN separately, so subsequent layers see preceding BNs' updated
    evaluation statistics, rather than their per-minibatch training outputs.
    Only running_mean/running_var change; counters, affine weights, routing,
    and every other parameter/buffer stay bitwise unchanged.

    Callers must supply training images only. Labels are neither accepted nor
    used. Global moments accumulate in float64; variance uses N-1 observations
    across images and spatial positions, independently for each channel.
    """
    names = list(names)
    modules = dict(model.named_modules())
    if not len(images) or batch_size < 1 or not names or len(set(names)) != len(names):
        raise ValueError("Need training images, positive batch size and distinct BN names")
    for name in names:
        if name not in modules or not isinstance(modules[name], nn.modules.batchnorm._BatchNorm):
            raise ValueError(f"Not a BN layer: {name}")
        if not modules[name].track_running_stats:
            raise ValueError(f"BN has no running statistics: {name}")
    model.eval()
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    rows = []
    for name in names:
        layer = modules[name]
        moments = {"sum": torch.zeros_like(layer.running_mean, dtype=torch.float64),
                   "squares": torch.zeros_like(layer.running_var, dtype=torch.float64), "count": 0, "batches": 0}

        def accumulate(module, inputs):
            values = inputs[0].detach().double()
            axes = (0,) + tuple(range(2, values.ndim))
            moments["sum"].add_(values.sum(dim=axes))
            moments["squares"].add_(values.square().sum(dim=axes))
            moments["count"] += values.numel() // values.shape[1]
            moments["batches"] += 1

        handle = layer.register_forward_pre_hook(accumulate)
        try:
            for start in range(0, len(images), batch_size):
                model(prepare_batch(images[start:start+batch_size], mean, std))
        finally:
            handle.remove()
        count = moments["count"]
        if count <= 1 or moments["batches"] != math.ceil(len(images)/batch_size):
            raise RuntimeError("BN refit did not see the expected observations exactly once")
        new_mean = moments["sum"] / count
        new_variance = ((moments["squares"] - moments["sum"].square()/count)/(count-1)).clamp_min(0)
        if not bool(torch.isfinite(new_mean).all() and torch.isfinite(new_variance).all()):
            raise RuntimeError("Nonfinite BN input moments")
        old_mean, old_variance = layer.running_mean.clone(), layer.running_var.clone()
        layer.running_mean.copy_(new_mean)
        layer.running_var.copy_(new_variance)
        rows.append({"name": name, "images": len(images), "observations_per_channel": count,
                     "batches": moments["batches"],
                     "mean_absolute_mean_change": float((layer.running_mean-old_mean).abs().mean()),
                     "mean_absolute_variance_change": float((layer.running_var-old_variance).abs().mean()),
                     "old_mean_variance": float(old_variance.mean()),
                     "new_mean_variance": float(layer.running_var.mean())})
        print(f"BN refit {name}: {len(images)} training images, {count} observations/channel", flush=True)
    allowed = {f"{name}.{suffix}" for name in names for suffix in ("running_mean", "running_var")}
    checks = {"parameters_bitwise_unchanged": all(torch.equal(parameters[name], value) for name, value in model.named_parameters()),
              "other_buffers_bitwise_unchanged": all(name in allowed or torch.equal(buffers[name], value) for name, value in model.named_buffers()),
              "all_modules_in_evaluation_mode": all(not module.training for module in model.modules()),
              "no_parameter_gradients": all(value.grad is None for value in model.parameters())}
    if not all(checks.values()):
        raise AssertionError(f"BN-only intervention invariant failed: {checks}")
    return {"layers": rows, "checks": checks, "images": len(images), "labels_used": False,
            "estimator": "Global input mean and unbiased variance, float64 accumulation, layer-wise evaluation-mode refit",
            "batch_counters_unchanged": True}
