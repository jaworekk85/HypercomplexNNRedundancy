"""Compact one block's second convolution, BN, and its immediate consumer."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from src.models.quaternion_conv_torch import QuaternionConv2d


def component_indices(kept, original_units):
    return [component * original_units + unit for component in range(4) for unit in kept]


def _convolution_like(conv, input_units, output_units):
    return QuaternionConv2d(input_units, output_units, conv.kernel.shape[-1],
                            conv.stride, conv.padding, conv.bias is not None).to(conv.kernel)


@torch.no_grad()
def compact_block(model, block_index, kept_units):
    """Return a separate compact model. Kept indices always refer to original units."""
    if model.kind != "quaternion" or not 0 <= block_index < len(model.blocks):
        raise ValueError("Expected a quaternion CNN and a valid block index")
    original = model.blocks[block_index][3]
    kept = [int(unit) for unit in kept_units]
    if not kept or kept != sorted(set(kept)) or min(kept) < 0 or max(kept) >= original.out_channels:
        raise ValueError("Keep a nonempty sorted unique subset of original units")
    result = deepcopy(model)
    channels = component_indices(kept, original.out_channels)
    replacement = _convolution_like(original, original.in_channels, len(kept))
    replacement.kernel.copy_(original.kernel[:, kept])
    if original.bias is not None:
        replacement.bias.copy_(original.bias[channels])
    result.blocks[block_index][3] = replacement
    bn = model.blocks[block_index][4]
    if not bn.affine or not bn.track_running_stats:
        raise ValueError("This experiment expects affine BN with running statistics")
    compact_bn = nn.BatchNorm2d(4 * len(kept), eps=bn.eps, momentum=bn.momentum).to(bn.weight)
    for name in ("weight", "bias", "running_mean", "running_var"):
        getattr(compact_bn, name).copy_(getattr(bn, name)[channels])
    compact_bn.num_batches_tracked.copy_(bn.num_batches_tracked)
    result.blocks[block_index][4] = compact_bn
    if block_index + 1 < len(model.blocks):
        consumer = model.blocks[block_index + 1][0]
        compact_consumer = _convolution_like(consumer, len(kept), consumer.out_channels)
        compact_consumer.kernel.copy_(consumer.kernel[:, :, kept])
        if consumer.bias is not None:
            compact_consumer.bias.copy_(consumer.bias)
        result.blocks[block_index + 1][0] = compact_consumer
    else:
        consumer = model.readout
        compact_consumer = nn.Linear(4 * len(kept), consumer.out_features, bias=consumer.bias is not None).to(consumer.weight)
        compact_consumer.weight.copy_(consumer.weight[:, channels])
        if consumer.bias is not None:
            compact_consumer.bias.copy_(consumer.bias)
        result.readout = compact_consumer
    result.train(model.training)
    return result


def removal_order(policy, norms, similarity, random_seed=0):
    """Fixed, transparent reference rankings; no labels used to choose units."""
    norms = np.asarray(norms, dtype=float)
    units = len(norms)
    if policy == "random":
        return np.random.default_rng(random_seed).permutation(units).tolist()
    if policy == "incoming_norm":
        return sorted(range(units), key=lambda unit: (norms[unit], unit))
    if policy != "activation_similarity":
        raise ValueError("Unknown removal policy")
    matrix = np.asarray(similarity, dtype=float)
    if matrix.shape != (units, units):
        raise ValueError("Similarity and norm widths differ")
    remaining, removed = set(range(units)), []
    while len(remaining) > 1:
        pairs = [(matrix[a, b], a, b) for a in sorted(remaining) for b in sorted(remaining) if a < b and np.isfinite(matrix[a, b])]
        if pairs:
            # Highest positive-coordinate centered cosine; exact ties favor lower indices.
            _, a, b = max(pairs, key=lambda pair: (pair[0], -pair[1], -pair[2]))
            victim = min((a, b), key=lambda unit: (norms[unit], unit))
        else:
            victim = min(remaining, key=lambda unit: (norms[unit], unit))
        remaining.remove(victim)
        removed.append(victim)
    removed.extend(sorted(remaining))
    return removed
