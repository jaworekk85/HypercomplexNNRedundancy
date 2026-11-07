"""Recompute saved CNN metrics and inspect checkpoints without new inference."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.data.cifar_cached import file_sha256
from src.models.cifar_cnn_torch import CifarCNN

ROOT = Path(__file__).resolve().parents[1]


def verify(folder):
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    config = summary["config"]
    checks = {"completed": summary["summary_status"] == "completed"}
    with np.load(folder / "split_indices.npz") as data:
        joined = np.concatenate([data[key] for key in ("train", "validation", "calibration")])
        checks["train_validation_calibration_disjoint"] = len(joined) == len(np.unique(joined)) == 50000
        checks["all_48000_training_examples"] = len(data["train"]) == 48000
        checks["official_10000_test_indices"] = np.array_equal(data["test"], np.arange(10000))
        with np.load(ROOT / config["split_source"]) as old:
            checks["validation_preserved"] = np.array_equal(data["validation"], old["validation"])
            checks["calibration_preserved"] = np.array_equal(data["calibration"], old["calibration"])
    checks["sources_match"] = all(file_sha256(ROOT / name) == digest for name, digest in summary["source_hashes"].items())
    rows = []
    for report in summary["models"]:
        kind = report["kind"]
        checkpoint = torch.load(folder / kind / "model.pt", map_location="cpu", weights_only=True)
        state = checkpoint["model"]
        model = CifarCNN(kind, config["quaternion_widths"], config["real_widths"], config["dropout"])
        model.load_state_dict(state, strict=True)
        parameter_names = {name for name, _ in model.named_parameters()}
        actual_parameters = sum(value.numel() for name, value in state.items() if name in parameter_names)
        actual_buffers = sum(value.numel() for name, value in state.items() if name not in parameter_names)
        checks[f"{kind}/actual_checkpoint_parameters"] = actual_parameters == report["real_trainable_parameters"]
        checks[f"{kind}/actual_checkpoint_buffers"] = actual_buffers == report["nontrainable_buffer_elements"]
        checks[f"{kind}/gradient_and_update_audit"] = all(report["first_update"]["checks"].values())
        checks[f"{kind}/checkpoint_reload_equivalent"] = report["checkpoint_reload_max_error"] <= 1e-6
        with (folder / kind / "training.csv").open(encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        minimum = min(float(row["validation_loss"]) for row in history)
        checks[f"{kind}/correct_training_length"] = len(history) == config["epochs"]
        checks[f"{kind}/epochs_contiguous_after_resume"] = [int(row["epoch"]) for row in history] == list(range(1, config["epochs"] + 1))
        expected_attempts = config["epochs"] * ((48000 + config["batch_size"] - 1) // config["batch_size"])
        checks[f"{kind}/update_attempts_preserved_after_resume"] = report["optimizer_updates_attempted"] == expected_attempts
        checks[f"{kind}/validation_minimum_selected"] = report["selected_validation"]["cross_entropy"] <= minimum + 1e-7
        checks[f"{kind}/selected_epoch_matches_checkpoint"] = checkpoint["best_epoch"] == report["best_epoch"]
        with np.load(folder / kind / "test_predictions.npz") as arrays:
            logits, labels = arrays["logits"].astype(np.float64), arrays["labels"]
        accuracy = float(np.mean(logits.argmax(1) == labels))
        shifted = logits - logits.max(1, keepdims=True)
        loss = float(np.mean(np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(labels)), labels]))
        checks[f"{kind}/correct_test_dimensions"] = logits.shape == (10000, 10) and labels.shape == (10000,)
        checks[f"{kind}/recomputed_test_accuracy"] = abs(accuracy - report["test"]["accuracy"]) < 1e-12
        checks[f"{kind}/recomputed_test_loss"] = abs(loss - report["test"]["cross_entropy"]) < 1e-5
        checks[f"{kind}/honest_target_status"] = report["target_80_percent_achieved"] == (accuracy >= config["target_accuracy"])
        with np.load(folder / kind / "calibration_activations.npz") as arrays:
            checks[f"{kind}/calibration_shape_and_finite"] = list(arrays["trained"].shape) == report["activation_shape"] and bool(np.isfinite(arrays["trained"]).all())
        rows.append({"kind": kind, "actual_checkpoint_parameters": actual_parameters, "recomputed_test_accuracy": accuracy, "recomputed_test_cross_entropy": loss})
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "models": rows,
              "verifier_sha256": file_sha256(Path(__file__)), "scope": "No retraining or model inference; inspect saved weights and recompute metrics from saved logits."}
    (folder / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"CNN artifact verification: {result['status']}; {len(checks)} checks")
    if not all(checks.values()):
        raise AssertionError([key for key, passed in checks.items() if not passed])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?", default=ROOT / "results" / "cifar_cnn")
    verify(parser.parse_args().folder)
