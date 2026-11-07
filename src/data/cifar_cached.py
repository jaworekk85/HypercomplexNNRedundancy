"""Read only the hash-verified, already downloaded official CIFAR archive."""

import hashlib
import pickle
import tarfile

import numpy as np

ARCHIVE_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cached_cifar(archive):
    if file_sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("CIFAR archive does not match the official SHA-256")
    with tarfile.open(archive, "r:gz") as source:
        def batch(name):
            with source.extractfile(f"cifar-10-batches-py/{name}") as handle:
                data = pickle.load(handle, encoding="bytes")
            return data[b"data"].reshape(-1, 3, 32, 32), np.asarray(data[b"labels"], dtype=np.int64)
        batches = [batch(f"data_batch_{i}") for i in range(1, 6)]
        test_x, test_y = batch("test_batch")
    return (np.concatenate([b[0] for b in batches]), np.concatenate([b[1] for b in batches])), (test_x, test_y)


def channel_statistics(images):
    total, squares, count = np.zeros(3), np.zeros(3), 0
    for start in range(0, len(images), 512):
        values = images[start:start + 512].astype(np.float64) / 255.0
        total += values.sum(axis=(0, 2, 3))
        squares += (values**2).sum(axis=(0, 2, 3))
        count += len(values) * values.shape[2] * values.shape[3]
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 0))
    if np.any(std <= 1e-12):
        raise ValueError("Constant RGB channel in training data")
    return {"mean": mean.tolist(), "std": std.tolist(), "fit_samples": len(images)}
