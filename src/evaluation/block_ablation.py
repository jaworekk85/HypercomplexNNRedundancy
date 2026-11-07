"""Post-block feature collection and single-channel functional diagnostics."""

import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def capture_blocks(model, normalized_inputs, batch_size=256):
    """Return CPU (N,C,H,W) tensors after each block; leave the model unchanged."""
    model.eval()
    outputs = [[] for _ in model.blocks]
    handles = []
    def hook_for(index):
        def hook(module, args, output):
            outputs[index].append(output.detach().cpu())
        return hook
    try:
        handles = [block.register_forward_hook(hook_for(index)) for index, block in enumerate(model.blocks)]
        for start in range(0, len(normalized_inputs), batch_size):
            model(normalized_inputs[start:start + batch_size])
    finally:
        for handle in handles:
            handle.remove()
    return [torch.cat(values) for values in outputs]


def grouped_spatial(values, quaternion):
    n, channels, height, width = values.shape
    if quaternion:
        if channels % 4:
            raise ValueError("Quaternion features must have four component blocks")
        return values.reshape(n, 4, channels // 4, height, width).permute(0, 3, 4, 2, 1).reshape(n, height * width, channels // 4, 4).numpy()
    return values.permute(0, 2, 3, 1).reshape(n, height * width, channels).numpy()


def unit_channels(channels, unit, quaternion):
    units = channels // 4 if quaternion else channels
    if quaternion and channels % 4 or not 0 <= unit < units:
        raise ValueError("Invalid feature width or unit index")
    return [unit + component * units for component in range(4)] if quaternion else [unit]


@torch.no_grad()
def predict_from_block(model, block_index, features, omitted_units=(), batch_size=256):
    """Temporarily zero complete output units and run the downstream network."""
    model.eval()
    device = next(model.parameters()).device
    channels = [channel for unit in omitted_units for channel in unit_channels(features.shape[1], unit, model.kind == "quaternion")]
    result = []
    for start in range(0, len(features), batch_size):
        values = features[start:start + batch_size].to(device).clone()
        if channels:
            values[:, channels] = 0
        for block in model.blocks[block_index + 1:]:
            values = block(values)
        result.append(model.readout(values.mean(dim=(2, 3))).cpu())
    return torch.cat(result).numpy()


def classification_metrics(logits, labels):
    values = torch.as_tensor(logits, dtype=torch.float64)
    target = torch.as_tensor(labels, dtype=torch.int64)
    return {"cross_entropy": float(F.cross_entropy(values, target)),
            "accuracy": float((values.argmax(1) == target).double().mean())}


def nearest_neighbor_scores(matrix):
    values = np.asarray(matrix, dtype=float).copy()
    np.fill_diagonal(values, np.nan)
    scores = []
    for row in values:
        finite = row[np.isfinite(row)]
        scores.append(float(finite.max()) if len(finite) else None)
    return scores


def spearman(first, second):
    pairs = [(a, b) for a, b in zip(first, second) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 3:
        return None
    def ranks(values):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        starts = np.cumsum(np.r_[0, counts[:-1]])
        return (starts + (counts - 1) / 2)[inverse]
    a, b = (ranks(np.array(values)) for values in zip(*pairs))
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def incoming_norms(model, block_index):
    conv = model.blocks[block_index][3]
    if model.kind == "quaternion":
        return conv.kernel.detach().square().sum(dim=(0, 2, 3, 4)).sqrt().cpu().tolist()
    return conv.weight.detach().flatten(1).norm(dim=1).cpu().tolist()
