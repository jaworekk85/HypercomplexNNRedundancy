"""Affine unit reconstruction and last-readout compensation, not a new method.

Prior context: Kim2020 and Ebid2025 in references.bib. We fit post-activation
GAP features; no claim that arbitrary quaternion maps commute with ReLU.
"""

from copy import deepcopy

import numpy as np
import torch

from src.compression.quaternion_channels import compact_block, component_indices


def right_hamilton(coefficients):
    """Column-vector matrix for x * a, matching QuaternionConv2d convention."""
    r, i, j, k = np.asarray(coefficients, dtype=np.float64)
    return np.array([[r, -i, -j, -k], [i, r, k, -j],
                     [j, -k, r, i], [k, j, -i, r]], dtype=np.float64)


def fit_affine(inputs, targets, mode):
    x, y = np.asarray(inputs, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or x.shape != y.shape or x.shape[1] != 4 or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Need matching finite (images,4) arrays")
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x-xm, y-ym
    if mode == "quaternion_right":
        basis = np.stack([right_hamilton(row) for row in np.eye(4)])
        design = np.einsum("coi,ni->noc", basis, xc).reshape(-1, 4)
        coefficients, _, rank, _ = np.linalg.lstsq(design, yc.reshape(-1), rcond=None)
        matrix, linear_coefficients = right_hamilton(coefficients), 4
    elif mode == "diagonal":
        denominator = (xc*xc).sum(0)
        values = np.divide((xc*yc).sum(0), denominator, out=np.zeros(4), where=denominator > 0)
        matrix, rank, linear_coefficients = np.diag(values), int(np.sum(denominator > 0)), 4
    elif mode == "full_real":
        solution, _, rank, _ = np.linalg.lstsq(xc, yc, rcond=None)
        matrix, linear_coefficients = solution.T, 16
    else:
        raise ValueError("Unknown affine reconstruction family")
    bias = ym-matrix@xm
    return {"mode": mode, "matrix": matrix.tolist(), "bias": bias.tolist(), "target_fit_mean": ym.tolist(),
            "linear_coefficients": linear_coefficients, "intercept_coefficients": 4, "design_rank": int(rank)}


def predict_affine(inputs, fit):
    return np.asarray(inputs, dtype=np.float64)@np.asarray(fit["matrix"]).T + np.asarray(fit["bias"])


def unit_readout(weight, units):
    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 2 or weight.shape[1] != 4*units:
        raise ValueError("Unexpected component-blocked readout")
    return weight.reshape(len(weight), 4, units).transpose(0, 2, 1)


def logits_from_units(values, weight, bias):
    values = np.asarray(values, dtype=np.float64)
    return np.einsum("nuc,kuc->nk", values, unit_readout(weight, values.shape[1])) + np.asarray(bias)


def center_logits(values):
    values = np.asarray(values, dtype=np.float64)
    return values-values.mean(axis=-1, keepdims=True)


def all_pair_fits(fit_values, check_values, weight, mode):
    fit_values, check_values = np.asarray(fit_values, dtype=np.float64), np.asarray(check_values, dtype=np.float64)
    if fit_values.ndim != 3 or check_values.ndim != 3 or fit_values.shape[1:] != check_values.shape[1:] or fit_values.shape[-1] != 4:
        raise ValueError("Expected (images,units,4) fit/check activations")
    units = fit_values.shape[1]
    weights = unit_readout(weight, units)
    records = []
    for keeper in range(units):
        for victim in range(units):
            if keeper == victim:
                continue
            fit = fit_affine(fit_values[:, keeper], fit_values[:, victim], mode)
            row = {"keeper": keeper, "victim": victim, **fit}
            for split, values in (("fit", fit_values), ("check", check_values)):
                residual = values[:, victim]-predict_affine(values[:, keeper], fit)
                constant = values[:, victim]-np.asarray(fit["target_fit_mean"])
                denominator = float(np.square(constant).mean())
                row[f"{split}_feature_mse"] = float(np.square(residual).mean())
                row[f"{split}_constant_feature_mse"] = denominator
                row[f"{split}_feature_nmse_vs_fit_mean"] = row[f"{split}_feature_mse"]/denominator if denominator > 1e-12 else None
                row[f"{split}_centered_logit_mse"] = float(np.square(center_logits(residual@weights[:, victim].T)).mean())
            row["nonconstant_fit_target"] = row["fit_constant_feature_mse"] > 1e-12
            records.append(row)
    return records


def choose_disjoint_pairs(records, removed_units):
    """Selection reads FIT errors only. No check-half metrics affect ranking."""
    ordered = sorted((row for row in records if row["nonconstant_fit_target"]),
                     key=lambda row: (row["fit_centered_logit_mse"], row["keeper"], row["victim"]))
    used, chosen = set(), []
    for row in ordered:
        if row["keeper"] not in used and row["victim"] not in used:
            chosen.append(deepcopy(row))
            used.update((row["keeper"], row["victim"]))
            if len(chosen) == removed_units:
                return chosen
    raise ValueError("Not enough disjoint nonconstant directed pairs")


def validate_pairs(pairs, units):
    endpoints = [int(row[key]) for row in pairs for key in ("keeper", "victim")]
    if len(set(endpoints)) != len(endpoints) or any(index < 0 or index >= units for index in endpoints):
        raise ValueError("Expected disjoint valid keeper/victim pairs")


def predicted_logits(values, weight, bias, pairs, compensation):
    values = np.asarray(values, dtype=np.float64)
    validate_pairs(pairs, values.shape[1])
    weights = unit_readout(weight, values.shape[1])
    logits = logits_from_units(values, weight, bias)
    for pair in pairs:
        keeper, victim = pair["keeper"], pair["victim"]
        if compensation == "affine":
            replacement = predict_affine(values[:, keeper], pair)
        elif compensation == "mean_only":
            replacement = np.broadcast_to(pair["target_fit_mean"], values[:, victim].shape)
        elif compensation == "none":
            replacement = np.zeros_like(values[:, victim])
        else:
            raise ValueError("Unknown compensation")
        logits += (replacement-values[:, victim])@weights[:, victim].T
    return logits


def aggregate_errors(values, weight, bias, pairs):
    original = logits_from_units(values, weight, bias)
    errors = {name: float(np.square(center_logits(predicted_logits(values, weight, bias, pairs, kind)-original)).mean())
              for name, kind in (("merged", "affine"), ("mean_only", "mean_only"), ("pruned", "none"))}
    errors["ratio_vs_mean_only"] = errors["merged"]/errors["mean_only"] if errors["mean_only"] > 1e-12 else None
    errors["ratio_vs_pruned"] = errors["merged"]/errors["pruned"] if errors["pruned"] > 1e-12 else None
    return errors


@torch.no_grad()
def merged_last_block(model, pairs, compensation="affine"):
    """Physically remove victims; fold affine maps into existing real readout.

    This is valid at the final GAP/real-linear consumer. We do not move maps
    across ReLU or claim an unrestricted map preserves an intermediate QConv.
    """
    units = model.blocks[-1][3].out_channels
    validate_pairs(pairs, units)
    victims = {row["victim"] for row in pairs}
    kept = sorted(set(range(units))-victims)
    result = compact_block(model, len(model.blocks)-1, kept).eval()
    if model.readout.bias is None:
        raise ValueError("Affine merging requires the existing classifier bias")
    for pair in pairs:
        keeper, victim = pair["keeper"], pair["victim"]
        victim_weight = model.readout.weight[:, component_indices([victim], units)]
        if compensation == "affine":
            matrix = torch.as_tensor(pair["matrix"], dtype=victim_weight.dtype, device=victim_weight.device)
            bias = torch.as_tensor(pair["bias"], dtype=victim_weight.dtype, device=victim_weight.device)
            channels = component_indices([kept.index(keeper)], len(kept))
            result.readout.weight[:, channels] += victim_weight@matrix
            result.readout.bias.add_(victim_weight@bias)
        elif compensation == "mean_only":
            mean = torch.as_tensor(pair["target_fit_mean"], dtype=victim_weight.dtype, device=victim_weight.device)
            result.readout.bias.add_(victim_weight@mean)
        elif compensation != "none":
            raise ValueError("Unknown compensation")
    return result, kept
