"""Intermediate right-Hamilton merging and orthogonal projection diagnostics.

No affine offset is folded across padding; no map is commuted through ReLU.
Standard associativity and a four-dimensional linear projection are used,
without claiming a new theorem or method. Prior context: Kim2020,
Asiaee2026CMR, Kobayashi2010 and Nitta2002 in references.bib.
"""

import numpy as np
import torch

from src.compression.quaternion_channels import compact_block, component_indices
from src.compression.quaternion_merging import right_hamilton, validate_pairs


BASIS = np.stack([right_hamilton(row) for row in np.eye(4)])


def expand_right(coefficients):
    return np.einsum("...c,coi->...oi", np.asarray(coefficients, dtype=np.float64), BASIS)


def project_right(matrices):
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.shape[-2:] != (4, 4) or not np.isfinite(matrices).all():
        raise ValueError("Need finite (...,4,4) matrices")
    coefficients = np.einsum("...oi,coi->...c", matrices, BASIS)/4
    return coefficients, expand_right(coefficients)


def fit_linear(inputs, targets, mode):
    x, y = np.asarray(inputs, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 4 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Need matching finite (observations,4) arrays")
    if mode == "quaternion_right":
        design = np.einsum("coi,ni->noc", BASIS, x).reshape(-1, 4)
        coefficients, _, rank, _ = np.linalg.lstsq(design, y.reshape(-1), rcond=None)
        matrix = right_hamilton(coefficients)
    elif mode == "full_real":
        solution, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        matrix = solution.T
    else:
        raise ValueError("Unknown linear fit family")
    return {"mode": mode, "matrix": matrix.tolist(), "rank": int(rank), "intercept": False}


def outgoing_addition(consumer_kernel, victim, matrix):
    """R(w_v) A, followed by per-edge projection. Kernel layout is (4,O,I,K,K)."""
    coefficients = np.asarray(consumer_kernel, dtype=np.float64)[:, :, victim].transpose(1, 2, 3, 0)
    full = expand_right(coefficients)@np.asarray(matrix, dtype=np.float64)
    projected_coefficients, projected = project_right(full)
    discarded = float(np.square(full-projected).sum())
    energy = float(np.square(full).sum())
    return projected_coefficients.transpose(3, 0, 1, 2), {"discarded_squared_norm": discarded, "full_squared_norm": energy,
            "discarded_energy_fraction": discarded/energy if energy > 1e-20 else 0.}


def fit_pairs(values, consumer_kernel):
    """Fit all directed pairs; common ranking uses the full map's FIT half only."""
    values = np.asarray(values, dtype=np.float64)
    units = values.shape[-2]
    records = []
    for keeper in range(units):
        x = values[..., keeper, :].reshape(-1, 4)
        for victim in range(units):
            if keeper == victim:
                continue
            y = values[..., victim, :].reshape(-1, 4)
            maps = {}
            for mode in ("quaternion_right", "full_real"):
                fit = fit_linear(x, y, mode)
                fit["fit_feature_mse"] = float(np.square(y-x@np.asarray(fit["matrix"]).T).mean())
                _, projection = outgoing_addition(consumer_kernel, victim, fit["matrix"])
                fit["projection"] = projection
                maps[mode] = fit
            energy = float(np.square(consumer_kernel[:, :, victim]).sum())
            records.append({"keeper": keeper, "victim": victim, "maps": maps,
                            "fit_selection_score": maps["full_real"]["fit_feature_mse"]*energy})
    return records


def select_pairs(candidates, count):
    selected, used = [], set()
    for row in sorted(candidates, key=lambda row: (row["fit_selection_score"], row["keeper"], row["victim"])):
        if row["keeper"] not in used and row["victim"] not in used:
            selected.append(row)
            used.update((row["keeper"], row["victim"]))
            if len(selected) == count:
                return selected
    raise ValueError("Not enough disjoint pairs")


def grouped_to_tensor(values, device="cpu"):
    array = np.asarray(values)
    n, positions, units, components = array.shape
    side = round(positions**.5)
    if side*side != positions or components != 4:
        raise ValueError("Expected square spatial grid and four components")
    return torch.as_tensor(array.reshape(n, side, side, units, 4).transpose(0, 4, 3, 1, 2).reshape(n, 4*units, side, side).copy(), device=device)


@torch.no_grad()
def substitute_features(features, pairs, mode):
    units = features.shape[1]//4
    validate_pairs(pairs, units)
    result = features.clone()
    for pair in pairs:
        keeper = component_indices([pair["keeper"]], units)
        victim = component_indices([pair["victim"]], units)
        matrix = torch.as_tensor(pair["maps"][mode]["matrix"], dtype=features.dtype, device=features.device)
        result[:, victim] = torch.einsum("oi,nihw->nohw", matrix, features[:, keeper])
    return result


@torch.no_grad()
def merge_internal(model, block_index, pairs, policy):
    if block_index+1 >= len(model.blocks):
        raise ValueError("An intermediate quaternion consumer is required")
    units = model.blocks[block_index][3].out_channels
    validate_pairs(pairs, units)
    victims = {row["victim"] for row in pairs}
    kept = sorted(set(range(units))-victims)
    result = compact_block(model, block_index, kept).eval()
    source_kernel = model.blocks[block_index+1][0].kernel.detach().cpu().double().numpy()
    projections = []
    for row in pairs:
        if policy == "paired_pruned":
            continue
        mode = "quaternion_right" if policy == "quaternion_exact" else "full_real" if policy == "full_projected" else None
        if mode is None:
            raise ValueError("Unknown internal merge policy")
        matrix = np.asarray(row["maps"][mode]["matrix"])
        if mode == "quaternion_right":
            _, projection = project_right(matrix)
            if not np.allclose(matrix, projection, atol=1e-10, rtol=1e-10):
                raise ValueError("Exact quaternion folding requires a right-Hamilton map")
        addition, diagnostic = outgoing_addition(source_kernel, row["victim"], matrix)
        target = result.blocks[block_index+1][0].kernel[:, :, kept.index(row["keeper"])]
        target.add_(torch.as_tensor(addition, dtype=target.dtype, device=target.device))
        projections.append(diagnostic)
    return result, kept, projections
