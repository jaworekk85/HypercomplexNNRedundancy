"""Explicit RGB normalization, pixel-wise quaternion encoding and data splits."""

import numpy as np


def stratified_indices(labels, sizes, seed, classes=10):
    labels = np.asarray(labels).reshape(-1)
    if any(n <= 0 or n % classes for n in sizes.values()):
        raise ValueError("Each split size must be positive and divisible by classes")
    rng = np.random.default_rng(seed)
    result = {name: [] for name in sizes}
    for label in range(classes):
        candidates = rng.permutation(np.flatnonzero(labels == label))
        if len(candidates) < sum(sizes.values()) // classes:
            raise ValueError("Insufficient examples for disjoint stratified splits")
        offset = 0
        for name, size in sizes.items():
            count = size // classes
            result[name].extend(candidates[offset:offset + count])
            offset += count
    return {name: rng.permutation(indices).astype(np.int64) for name, indices in result.items()}


def _validate_images(images):
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8 or not len(images):
        raise ValueError("Expected nonempty uint8 RGB images with shape (N,H,W,3)")
    return images


def fit_channel_statistics(training_images):
    images = _validate_images(training_images).astype(np.float64) / 255.0
    mean = images.mean(axis=(0, 1, 2))
    std = images.std(axis=(0, 1, 2))
    if np.any(std <= 1e-12):
        raise ValueError("A training RGB channel has zero variance")
    return {"mean": mean.tolist(), "std": std.tolist(), "fit_samples": len(images)}


def normalize_rgb(images, statistics):
    images = _validate_images(images)
    mean = np.asarray(statistics["mean"], dtype=np.float32)
    std = np.asarray(statistics["std"], dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Expected three finite means and positive standard deviations")
    return (images.astype(np.float32) / 255.0 - mean) / std


def encode_pixels(normalized_rgb, encoding):
    """One pixel per quaternion; spatial order is row-major, alpha is exactly 0.

    RGB returns (N,3*H*W), ARGB/ABGR return (N,H*W,4).
    The real control receives RGB only, with no unused weights on zero inputs.
    """
    rgb = np.asarray(normalized_rgb, dtype=np.float32)
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise ValueError("Expected finite normalized RGB images (N,H,W,3)")
    if encoding == "RGB":
        return rgb.reshape(len(rgb), -1)
    if encoding not in ("ARGB", "ABGR"):
        raise ValueError("Encoding must be RGB, ARGB, or ABGR")
    components = rgb if encoding == "ARGB" else rgb[..., ::-1]
    alpha = np.zeros(rgb.shape[:-1] + (1,), dtype=np.float32)
    return np.concatenate((alpha, components), axis=-1).reshape(len(rgb), -1, 4)
