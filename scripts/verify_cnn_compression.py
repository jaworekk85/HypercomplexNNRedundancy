"""Verify physically compact saved states and recompute metrics from saved logits."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.cifar_cnn import make_model, write_json
from scripts.cnn_compression import ROOT, load_compact
from src.data.cifar_cached import file_sha256
from src.evaluation.block_ablation import classification_metrics, incoming_norms
from src.compression.quaternion_channels import removal_order


def verify(folder):
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    config = summary["config"]
    frozen = json.loads((folder / "validation_frozen.json").read_text(encoding="utf-8"))
    checks = {"completed": summary["status"] == "completed", "no_fine_tuning": summary["fine_tuning_updates"] == 0,
              "all_cases_frozen_before_test": frozen["all_manifests_frozen_before_test"],
              "validation_freeze_unchanged": file_sha256(folder / "validation_frozen.json") == summary["validation_freeze_sha256"],
              "source_hashes_match": all(file_sha256(ROOT / name) == digest for name, digest in summary["protocol"]["source_hashes"].items()),
              "three_seeds": sorted({r["seed"] for r in summary["rows"]}) == config["seeds"],
              "all_75_cases": len(summary["rows"]) == 75}
    frozen_by_case = {(row["seed"], row["case"]): row for row in frozen["rows"]}
    baseline_source = json.loads((ROOT / config["source_multiseed"] / "summary.json").read_text(encoding="utf-8"))
    policy_inputs = {}
    for run in baseline_source["runs"]:
        path = ROOT / run["path"] / "quaternion" / "model.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = make_model(checkpoint["config"], "quaternion")
        model.load_state_dict(checkpoint["model"], strict=True)
        if run["seed"] == 2026:
            matrix_path = ROOT / config["source_block_analysis"] / "quaternion" / "block_3" / "calibration_similarity.npz"
        else:
            matrix_path = folder / "block_analysis" / f"seed_{run['seed']}" / "quaternion" / "block_3" / "calibration_similarity.npz"
        with np.load(matrix_path) as arrays:
            matrix = arrays["trained_grouped"]
        policy_inputs[run["seed"]] = (incoming_norms(model, 2), matrix, file_sha256(path), file_sha256(matrix_path))
    for row in summary["rows"]:
        prefix = f"{row['seed']}/{row['case']}"
        directory = folder / f"seed_{row['seed']}" / row["case"]
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        norms, similarity, source_hash, matrix_hash = policy_inputs[row["seed"]]
        checks[prefix + "/manifest_unchanged"] = file_sha256(directory / "manifest.json") == row["manifest_sha256"]
        checks[prefix + "/source_checkpoint_unchanged"] = source_hash == manifest["source_checkpoint_sha256"]
        checks[prefix + "/calibration_matrix_unchanged"] = matrix_hash == manifest["calibration_matrix_sha256"]
        expected_order = removal_order(row["policy"], norms, similarity, manifest["random_seed"] or 0) if row["policy"] != "baseline" else []
        checks[prefix + "/selection_reproduces_policy"] = manifest["removed_units"] == expected_order[:round(manifest["original_units"] * row["removed_fraction"])]
        model = load_compact(directory, manifest["base_config"], "cpu")
        checks[prefix + "/checkpoint_hash"] = file_sha256(directory / "model.pt") == row["checkpoint_sha256"]
        checks[prefix + "/actual_parameters"] = sum(p.numel() for p in model.parameters()) == row["real_trainable_parameters"]
        checks[prefix + "/actual_buffers"] = sum(b.numel() for b in model.buffers()) == row["nontrainable_buffer_elements"]
        checks[prefix + "/checkpoint_bytes"] = (directory / "model.pt").stat().st_size == row["checkpoint_bytes"]
        checks[prefix + "/kept_and_removed_partition"] = sorted(manifest["kept_units"] + manifest["removed_units"]) == list(range(manifest["original_units"]))
        removed = len(manifest["removed_units"])
        checks[prefix + "/physical_block3_parameter_saving"] = row["real_trainable_parameters"] == 73898 - 1200 * removed
        checks[prefix + "/physical_block3_buffer_saving"] = row["nontrainable_buffer_elements"] == 902 - 8 * removed
        checks[prefix + "/measured_compact_mask_equivalence"] = row["compact_vs_masked_max_abs_logit_error"] <= 1e-4
        checks[prefix + "/measured_reload_equivalence"] = row["checkpoint_reload_max_abs_error"] == 0
        checks[prefix + "/validation_frozen_values_preserved"] = all(row[key] == value for key, value in frozen_by_case[row["seed"], row["case"]].items())
        with np.load(directory / "test_predictions.npz") as arrays:
            logits, labels = arrays["logits"], arrays["labels"]
        metrics = classification_metrics(logits, labels)
        checks[prefix + "/official_test_dimensions"] = logits.shape == (10000, 10) and labels.shape == (10000,)
        checks[prefix + "/recomputed_accuracy"] = abs(metrics["accuracy"] - row["test"]["accuracy"]) < 1e-12
        checks[prefix + "/recomputed_loss"] = abs(metrics["cross_entropy"] - row["test"]["cross_entropy"]) < 1e-5
    for run in baseline_source["runs"]:
        row = next(r for r in summary["rows"] if r["seed"] == run["seed"] and r["policy"] == "baseline")
        old = next(r for r in run["models"] if r["kind"] == "quaternion")
        checks[f"{run['seed']}/zero_removal_reproduces_original_test"] = abs(row["test"]["accuracy"] - old["test"]["accuracy"]) < 1e-12 and abs(row["test"]["cross_entropy"] - old["test"]["cross_entropy"]) < 1e-5
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
              "verifier_sha256": file_sha256(Path(__file__)), "scope": "Load actual compact states and recompute saved test logits; no training or new model inference."}
    write_json(folder / "verification.json", result)
    print(f"Physical compression verification: {result['status']}; {len(checks)} checks", flush=True)
    if not all(checks.values()):
        raise AssertionError([name for name, passed in checks.items() if not passed])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?", default=ROOT / "results" / "cnn_compression")
    verify(parser.parse_args().folder)
