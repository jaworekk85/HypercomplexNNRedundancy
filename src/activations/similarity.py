"""A transparent grouped reference diagnostic, without a pruning policy."""

import numpy as np


def grouped_cosine(activations, center=True):
    """Compare units by flattening samples and four components together.

    Input: (samples, quaternion_units, 4). If center=True, remove each
    component's mean over samples first. This is a centered grouped cosine,
    not four independent Pearson channels and not a quaternion alignment fit.
    Constant units (after centering) and zero vectors get NaN rows/columns;
    the accompanying valid mask must be used when summarizing pairs.
    """
    values = np.asarray(activations, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 4 or min(values.shape[:2]) < 1:
        raise ValueError("Expected a nonempty (samples, quaternion_units, 4) array")
    if not np.isfinite(values).all():
        raise ValueError("Activations must be finite")
    if center:
        values = values - values.mean(axis=0, keepdims=True)
    features = values.transpose(1, 0, 2).reshape(values.shape[1], -1)
    norms = np.linalg.norm(features, axis=1)
    valid = norms > 1e-12
    normalized = np.zeros_like(features)
    np.divide(features, norms[:, None], out=normalized, where=valid[:, None])
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    similarity[~valid, :] = np.nan
    similarity[:, ~valid] = np.nan
    return similarity, valid
