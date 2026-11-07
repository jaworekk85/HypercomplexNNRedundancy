"""Convert EuroSAT_MS (Zenodo 7711810, MIT; Copernicus Sentinel-2 data) to a four-band array and a fixed split.

Standalone on purpose (no project imports) so it can run with `python -I` on downloaded files:
    python -I scripts/prepare_eurosat.py <extracted EuroSAT_MS dir> <output dir> <archive path>
Bands: Sentinel-2 13-band order B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12. We keep
[B08 NIR, B04 red, B03 green, B02 blue] so a pixel is q = NIR + R i + G j + B k: an encoding CHOICE
(NIR in the real part), recorded here and in the study configs, not a claim that it is canonical.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import tifffile

BANDS = {"B08": 7, "B04": 3, "B03": 2, "B02": 1}
ARCHIVE_MD5 = "091174add3c8e680a49244acf185b9f0"
SPLIT_SEED = 20261104
VALIDATION, CALIBRATION, TEST_FRACTION = 1000, 1000, 0.2


def digest(path, algorithm):
    value = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def stratified_split(labels, rng):
    """Per class: shuffle, then allot test/validation/calibration proportionally; the rest is training."""
    total = len(labels)
    parts = {"train": [], "validation": [], "calibration": [], "test": []}
    classes = np.unique(labels)
    quotas = {"test": round(TEST_FRACTION*total), "validation": VALIDATION, "calibration": CALIBRATION}
    for index, label in enumerate(classes):
        members = rng.permutation(np.flatnonzero(labels == label))
        start = 0
        for name, quota in quotas.items():
            # Largest-remainder style: the last class absorbs rounding so totals are exact.
            count = quota-sum(len(parts[name][i]) for i in range(index)) if index == len(classes)-1 else round(quota*len(members)/total)
            parts[name].append(members[start:start+count])
            start += count
        parts["train"].append(members[start:])
    return {name: np.sort(np.concatenate(values)) for name, values in parts.items()}


def main(source, output, archive):
    source, output = Path(source), Path(output)
    if digest(archive, "md5") != ARCHIVE_MD5:
        raise ValueError("EuroSAT_MS.zip does not match the Zenodo v2 MD5")
    classes = sorted(path.name for path in source.iterdir() if path.is_dir())
    files = [(label, path) for label, name in enumerate(classes) for path in sorted((source / name).glob("*.tif"))]
    images = np.empty((len(files), 4, 64, 64), dtype=np.uint16)
    for row, (_, path) in enumerate(files):
        array = tifffile.imread(path)
        if array.shape != (64, 64, 13) or array.dtype != np.uint16:
            raise ValueError(f"Unexpected tile {path.name}: {array.shape} {array.dtype}")
        images[row] = array[..., list(BANDS.values())].transpose(2, 0, 1)
    labels = np.array([label for label, _ in files], dtype=np.int64)
    split = stratified_split(labels, np.random.default_rng(SPLIT_SEED))
    if sum(len(value) for value in split.values()) != len(labels) or len(np.unique(np.concatenate(list(split.values())))) != len(labels):
        raise AssertionError("Split is not a partition")
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "eurosat_nrgb.npz", images=images, labels=labels)
    np.savez(output / "split_indices.npz", **split)
    # prepare_batch divides by 255, so statistics are computed in those units on training images only.
    train = images[split["train"]].astype(np.float64)/255
    statistics = {"mean": train.mean(axis=(0, 2, 3)).tolist(), "std": train.std(axis=(0, 2, 3)).tolist()}
    meta = {"source": "https://zenodo.org/records/7711810 (EuroSAT_MS.zip, v2)", "license": "MIT (dataset); Copernicus Sentinel data terms apply",
            "archive_md5": ARCHIVE_MD5, "archive_sha256": digest(archive, "sha256"), "classes": classes, "band_order": list(BANDS),
            "quaternion_encoding": "q = B08(NIR) + B04(R) i + B03(G) j + B02(B) k", "split_seed": SPLIT_SEED,
            "split_sizes": {name: int(len(value)) for name, value in split.items()},
            "class_counts": {name: {classes[c]: int((labels[value] == c).sum()) for c in range(len(classes))} for name, value in split.items()},
            "statistics": statistics, "statistics_units": "uint16 value / 255, training split only",
            "arrays_sha256": digest(output / "eurosat_nrgb.npz", "sha256"), "split_sha256": digest(output / "split_indices.npz", "sha256"),
            "test_policy": "Test split held out; not used for any selection."}
    (output / "data.json").write_text(json.dumps(meta, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: meta[key] for key in ("split_sizes", "statistics")}, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
