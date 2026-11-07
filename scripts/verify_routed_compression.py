"""Audit conditional compact routing states and saved prediction metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.cifar_cnn import write_json
from scripts.learned_grouping import load_export, make_model
from scripts.learned_grouping_compression import load_case
from src.compression.quaternion_channels import compact_block, removal_order
from src.data.cifar_cached import file_sha256
from src.evaluation.block_ablation import classification_metrics, incoming_norms

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(folder):
    torch.set_num_threads(4)
    summary = read(folder / "summary.json")
    protocol = summary["protocol"]
    config = protocol["config"]
    pilot_folder = ROOT / config["source_pilot"]
    training = read(pilot_folder / "summary.json")
    checks = {"frozen_sources_unchanged": all(file_sha256(ROOT / name) == digest for name,digest in protocol["source_hashes"].items()),
              "pilot_summary_unchanged": file_sha256(pilot_folder / "summary.json") == protocol["training_summary_sha256"],
              "pretest_gate_unchanged": file_sha256(pilot_folder / "selection_frozen.json") == protocol["selection_freeze_sha256"],
              "original_controls_unchanged": file_sha256(ROOT / config["baseline_compression"] / "summary.json") == protocol["baseline_summary_sha256"]}
    gate = training["compression_gate"]["supports_compression_followup"]
    if summary["status"] == "not_run_gate_failed":
        checks.update({"honest_gate_failure": not gate, "no_new_compact_models": summary["new_compact_models"] == 0,
                       "no_new_test_inference": summary["new_test_inference"] is False})
    else:
        checks.update({"completed_after_gate_pass": summary["status"] == "completed" and gate,
                       "all_75_cases": len(summary["rows"]) == 75, "no_fine_tuning": summary["fine_tuning_updates"] == 0,
                       "validation_freeze_unchanged": file_sha256(folder / "validation_frozen.json") == summary["validation_freeze_sha256"]})
        frozen = read(folder / "validation_frozen.json")
        checks["validation_frozen_before_new_tests"] = frozen["all_new_cases_frozen_before_new_test"]
        frozen_rows = {(r["mode"],r["case"]):r for r in frozen["rows"]}
        sources = {}
        for mode in config["modes"]:
            if mode == "identity":
                source_path = ROOT / training["config"]["baseline_source"] / "quaternion" / "model.pt"
                model = make_model(training["config"], "identity").eval()
                model.load_state_dict(torch.load(source_path, weights_only=True, map_location="cpu")["model"], strict=False)
            else:
                source_path = pilot_folder / mode / "model.pt"
                model = load_export(source_path, "cpu")
            with np.load(folder / f"{mode}_calibration_similarity.npz") as arrays:
                matrix = arrays["grouped"]
            sources[mode] = (model, incoming_norms(model, 2), matrix, file_sha256(source_path))
        for row in summary["rows"]:
            mode, case = row["mode"], row["case"]
            prefix = f"{mode}/{case}"
            directory = folder / mode / case
            manifest = read(directory / "manifest.json")
            source_model, norms, matrix, source_hash = sources[mode]
            order = removal_order(row["policy"], norms, matrix, manifest["random_seed"] or 0) if row["policy"] != "baseline" else []
            removed_count = round(32*row["removed_fraction"])
            checks[prefix + "/selection_matches_fixed_policy"] = manifest["removed_units"] == order[:removed_count]
            checks[prefix + "/whole_unit_partition"] = manifest["kept_units"] == sorted(set(range(32))-set(manifest["removed_units"]))
            checks[prefix + "/source_weights_unchanged"] = source_hash == manifest["source_checkpoint_sha256"]
            checks[prefix + "/matrix_unchanged"] = file_sha256(folder / f"{mode}_calibration_similarity.npz") == manifest["calibration_matrix_sha256"]
            checks[prefix + "/manifest_unchanged"] = file_sha256(directory / "manifest.json") == row["manifest_sha256"]
            checks[prefix + "/checkpoint_unchanged"] = file_sha256(directory / "model.pt") == row["checkpoint_sha256"]
            actual = load_case(directory, training["config"], "cpu")
            expected = compact_block(source_model, 2, manifest["kept_units"])
            checks[prefix + "/exact_original_slices_and_routing_preserved"] = all(torch.equal(value,expected.state_dict()[name]) for name,value in actual.state_dict().items())
            checks[prefix + "/actual_real_parameters"] = sum(p.numel() for p in actual.parameters()) == row["real_trainable_parameters"] == 73898-1200*removed_count
            checks[prefix + "/actual_buffers"] = sum(b.numel() for b in actual.buffers()) == row["buffer_elements"] == 934-8*removed_count
            checks[prefix + "/actual_checkpoint_bytes"] = (directory / "model.pt").stat().st_size == row["checkpoint_bytes"]
            checks[prefix + "/measured_compact_mask_equivalence"] = row["compact_mask_max_abs_error"] <= 1e-4
            checks[prefix + "/measured_reload_equivalence"] = row["reload_max_abs_error"] == 0
            checks[prefix + "/validation_values_frozen"] = all(row[key] == value for key,value in frozen_rows[mode,case].items())
            with np.load(directory / "test_predictions.npz") as arrays:
                logits, labels = arrays["logits"], arrays["labels"]
            metrics = classification_metrics(logits, labels)
            checks[prefix + "/official_prediction_shapes"] = logits.shape == (10000,10) and labels.shape == (10000,)
            checks[prefix + "/test_accuracy_recomputed"] = abs(metrics["accuracy"]-row["test"]["accuracy"]) < 1e-12
            checks[prefix + "/test_loss_recomputed"] = abs(metrics["cross_entropy"]-row["test"]["cross_entropy"]) < 1e-5
            if mode == "identity":
                prior = ROOT / config["baseline_compression"] / f"seed_{config['seed']}" / case / "test_predictions.npz"
                checks[prefix + "/identity_predictions_reused_unchanged"] = file_sha256(prior) == file_sha256(directory / "test_predictions.npz")
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
              "verifier_sha256": file_sha256(Path(__file__)), "scope": "Saved compact states/metrics or an honest failed gate; no new inference or training."}
    write_json(folder / "verification.json", result)
    print(f"Conditional routing-compression verification: {result['status']}; {len(checks)} checks", flush=True)
    if not all(checks.values()):
        raise AssertionError([key for key,passed in checks.items() if not passed])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?", default=ROOT / "results" / "learned_grouping_compression")
    verify(parser.parse_args().folder)
