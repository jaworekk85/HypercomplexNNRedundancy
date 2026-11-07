"""Scalar correlation, grouped cosine, rank and calibration split diagnostics."""

import numpy as np

from src.activations.similarity import grouped_cosine


def scalar_pearson(activations):
    values = np.asarray(activations, dtype=np.float64)
    if values.ndim < 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected at least two finite activation observations")
    values = values.reshape(len(values), -1)
    values = values - values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(values, axis=0)
    valid = norms > 1e-12
    normalized = np.zeros_like(values)
    np.divide(values, norms, out=normalized, where=valid)
    matrix = np.clip(normalized.T @ normalized, -1, 1)
    matrix[~valid, :] = np.nan
    matrix[:, ~valid] = np.nan
    return matrix, valid


def effective_rank(activations):
    """Entropy effective rank of the centered scalar-feature covariance spectrum."""
    values = np.asarray(activations, dtype=np.float64)
    if values.ndim < 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected at least two finite activation observations")
    values = values.reshape(len(values), -1)
    values = values - values.mean(axis=0, keepdims=True)
    eigenvalues = np.maximum(np.linalg.eigvalsh(values.T @ values / (len(values) - 1)), 0)[::-1]
    total = eigenvalues.sum()
    probabilities = eigenvalues[eigenvalues > 0] / total if total > 0 else np.array([])
    rank = float(np.exp(-np.sum(probabilities * np.log(probabilities)))) if len(probabilities) else 0.0
    return {
        "effective_rank": rank,
        "normalized_effective_rank": rank / min(len(values) - 1, values.shape[1]),
        "scalar_feature_count": values.shape[1],
        "covariance_eigenvalues": eigenvalues.tolist(),
    }


def pair_summary(matrix):
    pairs = matrix[np.triu_indices(len(matrix), 1)]
    pairs = pairs[np.isfinite(pairs)]
    if not len(pairs):
        return {"valid_pairs": 0, "mean": None, "median": None, "p95": None, "maximum": None,
                "fraction_ge_0_9": None, "fraction_abs_ge_0_9": None}
    return {
        "valid_pairs": len(pairs), "mean": float(pairs.mean()),
        "median": float(np.median(pairs)), "p95": float(np.quantile(pairs, 0.95)),
        "maximum": float(pairs.max()), "fraction_ge_0_9": float(np.mean(pairs >= 0.9)),
        "fraction_abs_ge_0_9": float(np.mean(np.abs(pairs) >= 0.9)),
    }


def pair_stability(first, second):
    index = np.triu_indices(len(first), 1)
    a, b = first[index], second[index]
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    correlation = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.std(a) > 1e-12 and np.std(b) > 1e-12 else None
    return {"shared_valid_pairs": len(a), "pair_value_pearson": correlation,
            "mean_absolute_pair_difference": float(np.mean(np.abs(a - b))) if len(a) else None}


def activation_diagnostics(activations):
    values = np.asarray(activations)
    scalar, valid = scalar_pearson(values)
    flat = values.reshape(len(values), -1)
    first, _ = scalar_pearson(values[::2])
    second, _ = scalar_pearson(values[1::2])
    report = {
        "shape": list(values.shape),
        "scalar": {
            "valid_features": int(valid.sum()), "constant_features": int((~valid).sum()),
            "all_zero_features": int(np.sum(np.all(flat == 0, axis=0))),
            "pairs": pair_summary(scalar), "calibration_halves_stability": pair_stability(first, second),
        },
        "rank": effective_rank(values),
    }
    matrices = {"scalar": scalar}
    if values.ndim == 3:
        grouped, group_valid = grouped_cosine(values)
        first, _ = grouped_cosine(values[::2])
        second, _ = grouped_cosine(values[1::2])
        report["grouped"] = {
            "valid_units": int(group_valid.sum()), "constant_units": int((~group_valid).sum()),
            "pairs": pair_summary(grouped), "calibration_halves_stability": pair_stability(first, second),
        }
        matrices["grouped"] = grouped
    return report, matrices
