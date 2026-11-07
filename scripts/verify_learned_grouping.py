"""Verify one-seed learned-routing training, hard exports and saved predictions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.cifar_cnn import write_json
from scripts.learned_grouping import load_export
from src.data.cifar_cached import file_sha256
from src.evaluation.block_ablation import classification_metrics
from src.models.channel_routing import maximum_assignment

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def unordered_groups(indices):
    indices = np.asarray(indices)
    return {tuple(sorted(indices[unit+np.arange(4)*(len(indices)//4)].tolist())) for unit in range(len(indices)//4)}


def verify(folder):
    torch.set_num_threads(4)
    summary = read(folder / "summary.json")
    config, protocol = summary["config"], summary["protocol"]
    baseline_folder = ROOT / config["baseline_source"]
    frozen = read(folder / "selection_frozen.json")
    checks = {"completed": summary["status"] == "completed",
              "frozen_numerical_sources_unchanged": all(file_sha256(ROOT / name) == digest for name,digest in protocol["source_hashes"].items()),
              "original_identity_results_unchanged": file_sha256(baseline_folder / "summary.json") == protocol["original_summary_sha256"],
              "selection_freeze_unchanged": file_sha256(folder / "selection_frozen.json") == summary["selection_freeze_sha256"],
              "choices_frozen_before_new_test": frozen["all_choices_frozen_before_new_test"],
              "one_training_seed_honestly_reported": config["seed"] == 2026,
              "original_split_preserved": file_sha256(folder / "split_indices.npz") == file_sha256(baseline_folder / "split_indices.npz"),
              "original_normalization_preserved": file_sha256(folder / "data.json") == file_sha256(baseline_folder / "data.json")}
    with np.load(folder / "split_indices.npz") as split:
        joined = np.concatenate([split[key] for key in ("train", "validation", "calibration")])
        checks["disjoint_full_training_and_heldout_data"] = len(joined) == len(np.unique(joined)) == 50000
        checks["full_48000_training_images"] = len(split["train"]) == 48000
    identity = read(folder / "identity_reference.json")
    checks["identity_reference_exact_forward_equivalence"] = identity["reference_forward_max_abs_error"] == 0
    checks["identity_checkpoint_unchanged"] = identity["source_checkpoint_sha256"] == file_sha256(baseline_folder / "quaternion" / "model.pt")
    checks["identity_selected_epoch_in_same_hard_window"] = identity["best_epoch"] >= config["soft_epochs"]
    mechanism = {}
    last_states = {}
    for report in summary["models"]:
        mode = report["mode"]
        source = folder / mode
        exported = torch.load(source / "model.pt", map_location="cpu", weights_only=True)
        last = torch.load(source / "last.pt", map_location="cpu", weights_only=True)
        last_states[mode] = last
        model = load_export(source / "model.pt", "cpu")
        checks[f"{mode}/actual_exported_parameters"] = sum(p.numel() for p in model.parameters()) == report["exported_real_parameters"] == 73898
        checks[f"{mode}/actual_exported_buffers"] = sum(b.numel() for b in model.buffers()) == report["exported_buffer_elements"] == 934
        checks[f"{mode}/actual_checkpoint_bytes"] = (source / "model.pt").stat().st_size == report["checkpoint_bytes"]
        checks[f"{mode}/dense_router_not_in_export"] = "router.logits" not in exported["model"]
        indices = exported["model"]["router.indices"].tolist()
        checks[f"{mode}/valid_one_to_one_hard_permutation"] = sorted(indices) == list(range(32))
        checks[f"{mode}/permutation_matches_report"] = indices == report["routing"]["matching"]
        checks[f"{mode}/export_exactly_matches_selected_hard_state"] = all(torch.equal(value,last["best_state"][name]) for name,value in exported["model"].items())
        checks[f"{mode}/expected_initial_training_parameters"] = report["training_real_parameters"] == (74922 if mode == "learned" else 73898)
        checks[f"{mode}/first_update_integrity"] = all(report["first_update"]["checks"].values())
        checks[f"{mode}/hard_export_equivalence"] = report["hard_export_max_abs_error"] == 0
        checks[f"{mode}/reload_equivalence"] = report["reload_max_abs_error"] == 0
        with (source / "training.csv").open(encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        checks[f"{mode}/contiguous_50_epochs"] = [int(row["epoch"]) for row in history] == list(range(1,config["epochs"]+1))
        checks[f"{mode}/last_checkpoint_final_epoch"] = last["epoch"] == config["epochs"]
        checks[f"{mode}/full_resume_state"] = all(name in last for name in ("optimizer", "scheduler", "scaler", "torch_rng", "cuda_rng", "order_rng", "augment_rng"))
        checks[f"{mode}/correct_update_attempts"] = report["optimizer_updates_attempted"] == config["epochs"]*((48000+config["batch_size"]-1)//config["batch_size"])
        eligible = [row for row in history if int(row["epoch"]) >= config["soft_epochs"]]
        best = min(eligible, key=lambda row: float(row["validation_loss"]))
        checks[f"{mode}/hard_validation_minimum_selected"] = report["best_epoch"] == int(best["epoch"]) == exported["best_epoch"]
        checks[f"{mode}/selected_validation_reproduces_csv"] = abs(report["selected_validation"]["cross_entropy"] - float(best["validation_loss"])) < 1e-12
        checks[f"{mode}/selected_epoch_was_hard"] = best["phase_after_epoch"] == "hard"
        prior = next(r for r in frozen["records"] if r["mode"] == mode)
        checks[f"{mode}/frozen_validation_values_preserved"] = all(report[key] == value for key,value in prior.items() if key != "test_evaluated")
        with np.load(source / "test_predictions.npz") as saved:
            logits, labels = saved["logits"], saved["labels"]
        metrics = classification_metrics(logits, labels)
        checks[f"{mode}/official_test_shapes_finite"] = logits.shape == (10000,10) and labels.shape == (10000,) and bool(np.isfinite(logits).all())
        checks[f"{mode}/test_accuracy_recomputed"] = abs(metrics["accuracy"]-report["test"]["accuracy"]) < 1e-12
        checks[f"{mode}/test_loss_recomputed"] = abs(metrics["cross_entropy"]-report["test"]["cross_entropy"]) < 1e-5
        if mode == "learned":
            best_state = last["best_state"]
            matching = maximum_assignment(best_state["router.logits"].double().numpy()).tolist()
            checks["learned/export_is_exact_optimal_assignment"] = matching == indices
            checks["learned/selected_router_hard_enabled"] = bool(best_state["router.hard_enabled"])
            gradient = next(row for row in report["first_update"]["parameters"] if row["name"] == "router.logits")
            checks["learned/routing_gradient_and_update_nonzero"] = gradient["gradient_norm"] > 0 and gradient["max_abs_update"] > 0
            initial = best_state["router.initial_indices"].tolist()
            changed = sum(a != b for a,b in zip(initial,indices))
            checks["learned/changed_positions_recomputed"] = changed == report["routing"]["changed_output_positions"]
            before_groups, after_groups = unordered_groups(initial), unordered_groups(indices)
            mechanism = {"initial_permutation": initial, "exported_permutation": indices, "changed_positions": changed,
                         "initial_unordered_groups": [list(value) for value in sorted(before_groups)],
                         "exported_unordered_groups": [list(value) for value in sorted(after_groups)],
                         "new_unordered_member_groups": len(after_groups-before_groups),
                         "unordered_membership_actually_changed": before_groups != after_groups}
            hardening = report["hardening"]
            checks["learned/explicit_hardening_measurement"] = hardening["epoch"] == config["soft_epochs"] and "soft_validation" in hardening and "immediate_hard_validation" in hardening
    checks["paired_random_initial_assignment"] = torch.equal(last_states["random"]["model"]["router.indices"], last_states["learned"]["model"]["router.initial_indices"])
    random, learned = (next(r for r in summary["models"] if r["mode"] == name) for name in ("random", "learned"))
    quality = learned["selected_validation"]["accuracy"] >= max(identity["selected_validation"]["accuracy"],random["selected_validation"]["accuracy"])-config["maximum_validation_accuracy_deficit_for_compression_gate"]-1e-12
    expected_gate = quality and learned["routing"]["changed_output_positions"] > 0
    checks["compression_gate_recomputed_without_test"] = summary["compression_gate"]["supports_compression_followup"] == expected_gate
    write_json(folder / "membership_analysis.json", mechanism)
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "membership_analysis": mechanism,
              "verifier_sha256": file_sha256(Path(__file__)), "scope": "Inspect saved models/checkpoints/metrics only; no new training or inference."}
    write_json(folder / "verification.json", result)
    print(f"Learned grouping verification: {result['status']}; {len(checks)} checks", flush=True)
    if not all(checks.values()):
        raise AssertionError([key for key,passed in checks.items() if not passed])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?", default=ROOT / "results" / "learned_grouping_pilot")
    verify(parser.parse_args().folder)
