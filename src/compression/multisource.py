"""Multi-source linear compensation folded into the next convolution.

Unconstrained ridge reconstruction from all kept channels and folding into the
consumer is established prior art (GRAIL2026, Asiaee2026CMR, Kim2020). Here the
quaternion variant restricts every source map to a right-Hamilton block,
q_v ~= sum_k q_k*a_k, so folding w_k <- w_k + a_k*w_v keeps the consumer a
quaternion convolution. No new theorem or method is claimed.
"""

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from src.compression.internal_merging import BASIS, expand_right, outgoing_addition
from src.compression.quaternion_channels import compact_block, component_indices


def ridge(design, target, alpha):
    """Closed-form ridge, lambda = alpha * mean diagonal of the Gram matrix."""
    gram = design.T@design
    penalty = alpha*float(np.mean(np.diag(gram))) if alpha > 0 else 0.
    return np.linalg.solve(gram+penalty*np.eye(len(gram)), design.T@target)


def fit_quaternion_multi(values, victim, kept, alpha):
    """values: (samples, positions, units, 4). Returns (len(kept), 4) right-Hamilton coefficients."""
    x = np.asarray(values, dtype=np.float64)[..., kept, :].reshape(-1, len(kept), 4)
    y = np.asarray(values, dtype=np.float64)[..., victim, :].reshape(-1, 4)
    design = np.einsum("coi,nki->nokc", BASIS, x).reshape(-1, 4*len(kept))
    return ridge(design, y.reshape(-1), alpha).reshape(len(kept), 4)


def fit_full_multi(values, victim, kept, alpha):
    """Unconstrained 4x4 map per kept unit: returns (len(kept), 4, 4)."""
    x = np.asarray(values, dtype=np.float64)[..., kept, :].reshape(-1, 4*len(kept))
    y = np.asarray(values, dtype=np.float64)[..., victim, :].reshape(-1, 4)
    return ridge(x, y, alpha).T.reshape(4, len(kept), 4).transpose(1, 0, 2)


def fit_real_multi(values, victim, kept, alpha):
    """values: (samples, positions, channels). Returns (len(kept),) coefficients."""
    x = np.asarray(values, dtype=np.float64)[..., kept].reshape(-1, len(kept))
    y = np.asarray(values, dtype=np.float64)[..., victim].reshape(-1)
    return ridge(x, y, alpha)


def quaternion_maps(fit_values, victims, mode, alpha):
    """One map set per victim, sources are all non-victim units."""
    units = fit_values.shape[-2]
    kept = sorted(set(range(units))-set(victims))
    result = []
    for victim in victims:
        if mode == "quaternion_right":
            matrices = expand_right(fit_quaternion_multi(fit_values, victim, kept, alpha))
        elif mode == "full_real":
            matrices = fit_full_multi(fit_values, victim, kept, alpha)
        else:
            raise ValueError("Unknown multi-source family")
        result.append({"victim": int(victim), "kept": kept, "mode": mode, "matrices": matrices.tolist()})
    return result


@torch.no_grad()
def merge_quaternion_multi(model, block_index, maps):
    """Compact the block and fold every source map into the quaternion consumer."""
    units = model.blocks[block_index][3].out_channels
    victims = [row["victim"] for row in maps]
    kept = sorted(set(range(units))-set(victims))
    result = compact_block(model, block_index, kept).eval()
    source = model.blocks[block_index+1][0].kernel.detach().cpu().double().numpy()
    target = result.blocks[block_index+1][0].kernel
    discarded = energy = 0.
    for row in maps:
        if row["kept"] != kept:
            raise ValueError("Map sources must be exactly the kept units")
        for position, matrix in enumerate(np.asarray(row["matrices"])):
            addition, diagnostic = outgoing_addition(source, row["victim"], matrix)
            target[:, :, position].add_(torch.as_tensor(addition, dtype=target.dtype, device=target.device))
            discarded += diagnostic["discarded_squared_norm"]
            energy += diagnostic["full_squared_norm"]
    return result, kept, {"discarded_energy_fraction": discarded/energy if energy > 1e-20 else 0.}


@torch.no_grad()
def substitute_multi(features, maps, quaternion=True):
    """Replace victim features by their fitted reconstruction from kept features."""
    units = features.shape[1]//4 if quaternion else features.shape[1]
    result = features.clone()
    for row in maps:
        if quaternion:
            sources = features[:, component_indices(row["kept"], units)].reshape(len(features), 4, len(row["kept"]), *features.shape[2:])
            matrices = torch.as_tensor(row["matrices"], dtype=features.dtype, device=features.device)
            result[:, component_indices([row["victim"]], units)] = torch.einsum("koi,nikhw->nohw", matrices, sources)
        else:
            coefficients = torch.as_tensor(row["coefficients"], dtype=features.dtype, device=features.device)
            result[:, row["victim"]] = torch.einsum("k,nkhw->nhw", coefficients, features[:, row["kept"]])
    return result


def real_pair_fits(fit_values, consumer_weight):
    """Directed scalar fits h_v ~= c*h_k; ranking = residual MSE times outgoing kernel energy."""
    x = np.asarray(fit_values, dtype=np.float64).reshape(-1, fit_values.shape[-1])
    energy = np.square(np.asarray(consumer_weight, dtype=np.float64)).sum(axis=(0, 2, 3))
    gram = x.T@x
    records = []
    for keeper in range(x.shape[1]):
        for victim in range(x.shape[1]):
            if keeper != victim:
                coefficient = gram[keeper, victim]/gram[keeper, keeper] if gram[keeper, keeper] > 1e-20 else 0.
                mse = float(np.square(x[:, victim]-coefficient*x[:, keeper]).mean())
                records.append({"keeper": keeper, "victim": victim, "coefficient": float(coefficient), "fit_feature_mse": mse,
                                "fit_selection_score": mse*float(energy[victim])})
    return records


def select_disjoint(records, count):
    selected, used = [], set()
    for row in sorted(records, key=lambda row: (row["fit_selection_score"], row["keeper"], row["victim"])):
        if row["keeper"] not in used and row["victim"] not in used:
            selected.append(row)
            used.update((row["keeper"], row["victim"]))
            if len(selected) == count:
                return selected
    raise ValueError("Not enough disjoint pairs")


def real_maps(fit_values, victims, alpha):
    kept = sorted(set(range(fit_values.shape[-1]))-set(victims))
    return [{"victim": int(victim), "kept": kept, "coefficients": fit_real_multi(fit_values, victim, kept, alpha).tolist()} for victim in victims]


def pair_maps(pairs, channels):
    """Pairwise scalar merging written as sparse multi-source maps."""
    kept = sorted(set(range(channels))-{row["victim"] for row in pairs})
    maps = []
    for row in pairs:
        coefficients = np.zeros(len(kept))
        coefficients[kept.index(row["keeper"])] = row["coefficient"]
        maps.append({"victim": row["victim"], "kept": kept, "coefficients": coefficients.tolist()})
    return maps


@torch.no_grad()
def compact_real_block(model, block_index, kept):
    """Real-CNN counterpart of compact_block: slice conv outputs, BN and consumer inputs."""
    if model.kind != "real" or block_index+1 >= len(model.blocks):
        raise ValueError("Expected a real CNN with an intermediate consumer")
    kept = [int(channel) for channel in kept]
    result = deepcopy(model)
    conv, bn, consumer = model.blocks[block_index][3], model.blocks[block_index][4], model.blocks[block_index+1][0]
    if conv.bias is not None or consumer.bias is not None:
        raise ValueError("This experiment expects bias-free convolutions")
    compact = nn.Conv2d(conv.in_channels, len(kept), conv.kernel_size, conv.stride, conv.padding, bias=False).to(conv.weight)
    compact.weight.copy_(conv.weight[kept])
    compact_bn = nn.BatchNorm2d(len(kept), eps=bn.eps, momentum=bn.momentum).to(bn.weight)
    for name in ("weight", "bias", "running_mean", "running_var"):
        getattr(compact_bn, name).copy_(getattr(bn, name)[kept])
    compact_bn.num_batches_tracked.copy_(bn.num_batches_tracked)
    compact_consumer = nn.Conv2d(len(kept), consumer.out_channels, consumer.kernel_size, consumer.stride, consumer.padding, bias=False).to(consumer.weight)
    compact_consumer.weight.copy_(consumer.weight[:, kept])
    result.blocks[block_index][3], result.blocks[block_index][4], result.blocks[block_index+1][0] = compact, compact_bn, compact_consumer
    return result.eval()


@torch.no_grad()
def merge_real(model, block_index, maps, channels):
    victims = [row["victim"] for row in maps]
    kept = sorted(set(range(channels))-set(victims))
    result = compact_real_block(model, block_index, kept)
    source = model.blocks[block_index+1][0].weight
    target = result.blocks[block_index+1][0].weight
    for row in maps:
        if row["kept"] != kept:
            raise ValueError("Map sources must be exactly the kept channels")
        coefficients = torch.as_tensor(row["coefficients"], dtype=target.dtype, device=target.device)
        target.add_(coefficients[None, :, None, None]*source[:, row["victim"]][:, None])
    return result, kept
