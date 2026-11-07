"""Verify saved pilot artifacts without retraining or another model test pass."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(output_dir):
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    config = summary["config"]
    checks = {"summary_completed": summary["status"] == "passed"}
    with np.load(output_dir / "split_indices.npz") as splits:
        indices = np.concatenate([splits[name] for name in ("train", "validation", "calibration")])
        checks["splits_disjoint"] = len(indices) == len(np.unique(indices))
        checks["split_sizes"] = all(len(splits[name]) == config[f"{name}_samples"] for name in ("train", "validation", "calibration", "test"))
        checks["full_official_test_indices"] = np.array_equal(splits["test"], np.arange(config["test_samples"]))
    checks["split_hash_matches"] = sha256(output_dir / "split_indices.npz") == summary["split_indices_sha256"]
    checks["source_hashes_match"] = all(sha256(ROOT / name) == hashes["sha256"] for name, hashes in summary["source_hashes"].items())
    data = json.loads((output_dir / "data.json").read_text(encoding="utf-8"))
    archive = ROOT / ".cache" / "keras" / "datasets" / "cifar-10-batches-py-target_archive"
    checks["actual_archive_matches_official_sha256"] = sha256(archive) == data["expected_archive_sha256"]
    expected = {(seed, encoding) for seed in config["seeds"] for encoding in [*config["encodings"], "RGB"]}
    found = set()
    rows = []
    for path in sorted(output_dir.glob("seed_*/metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        found.add((record["seed"], record["encoding"]))
        prefix = path.parent.name
        checks[f"{prefix}/all_training_checks"] = record["status"] == "passed" and all(record["checks"].values())
        checks[f"{prefix}/test_complete"] = record["test_evaluated"]
        with (path.parent / "training.csv").open(encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        selected = min(history, key=lambda row: float(row["validation_loss"]))
        checks[f"{prefix}/minimum_validation_checkpoint"] = int(selected["epoch"]) == record["best_epoch"]
        checks[f"{prefix}/selected_validation_metric"] = abs(float(selected["validation_loss"]) - record["selected_validation"]["cross_entropy"]) < 1e-6
        with np.load(path.parent / "test_predictions.npz") as predictions:
            logits = predictions["logits"].astype(np.float64)
            labels = predictions["labels"]
            checks[f"{prefix}/test_shapes_and_finite"] = logits.shape == (config["test_samples"], config["classes"]) and bool(np.isfinite(logits).all())
            accuracy = float(np.mean(logits.argmax(axis=1) == labels))
            shifted = logits - logits.max(axis=1, keepdims=True)
            loss = float(np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]))
            checks[f"{prefix}/accuracy_recomputed_from_saved_logits"] = abs(accuracy - record["test"]["accuracy"]) < 1e-12
            checks[f"{prefix}/loss_recomputed_from_saved_logits"] = abs(loss - record["test"]["cross_entropy"]) < 1e-6
        parameter_sizes = []
        with zipfile.ZipFile(path.parent / "model.keras") as saved:
            with h5py.File(io.BytesIO(saved.read("model.weights.h5")), "r") as weights:
                weights.visititems(lambda name, item: parameter_sizes.append(item.size) if isinstance(item, h5py.Dataset) else None)
        checks[f"{prefix}/parameters_counted_in_actual_checkpoint"] = sum(parameter_sizes) == record["real_trainable_parameters"]
        with np.load(path.parent / "calibration_activations.npz") as arrays:
            shape = (config["calibration_samples"], config["quaternion_hidden_units"], 4) if record["encoding"] != "RGB" else (config["calibration_samples"], record["parameter_budgets"]["real_hidden_units"])
            checks[f"{prefix}/calibration_shape_and_finite"] = arrays["trained"].shape == shape and bool(np.isfinite(arrays["trained"]).all())
        rows.append({"run": prefix, "checkpoint_parameters": sum(parameter_sizes), "recomputed_test_accuracy": accuracy, "recomputed_test_cross_entropy": loss})
    checks["all_expected_runs_present"] = found == expected and len(rows) == len(expected)
    report = {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "runs": rows,
              "verifier_sha256": sha256(Path(__file__)), "scope": "No retraining, model inference or test-driven changes; recomputation from saved predictions and inspection of saved weights."}
    (output_dir / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Artifact verification: {report['status']}; {len(checks)} checks, {len(rows)} runs")
    if not all(checks.values()):
        raise AssertionError([name for name, passed in checks.items() if not passed])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, nargs="?", default=ROOT / "results" / "rgb_pilot")
    verify(parser.parse_args().output_dir)
