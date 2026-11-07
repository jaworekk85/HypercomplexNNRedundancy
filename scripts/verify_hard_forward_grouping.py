"""Verify the completed discrete-forward pilot and independently reload models."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.cifar_cnn import write_json
from scripts.learned_grouping import load_export
from src.cnn_training import evaluate, seed_everything
from src.data.cifar_cached import file_sha256, load_cached_cifar

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def member_sets(indices):
    positions = np.asarray(indices).reshape(4, -1).T
    return {tuple(sorted(row.tolist())) for row in positions}


def verify(output):
    output = Path(output)
    summary, protocol = read(output / "summary.json"), read(output / "protocol.json")
    study, config = protocol["study"], protocol["training_config"]
    reference = ROOT / study["pilot_reference"]
    checks = {}

    def check(name, value):
        checks[name] = bool(value)

    check("completed", summary["status"] == "completed")
    check("same_protocol", summary["protocol"] == protocol)
    for name, digest in {**protocol["source_hashes"], **protocol["protected_inputs"]}.items():
        check(f"unchanged:{name}", file_sha256(ROOT / name) == digest)
    check("original_config_preserved", config == read(reference / "protocol.json")["config"])
    check("no_official_test", not summary["official_test_evaluated"] and not protocol["official_test_evaluated"])
    check("not_claimed_as_compression", summary["physical_channel_savings"] == 0)
    check("extra_training_variables_disclosed", not protocol["trainable_parameter_budgets_equal"])
    check("reused_control_disclosed", protocol["control_training_reused"])
    check("synthetic_integrity", all(read(output / "smoke.json")["checks"].values()))
    selection = read(output / "selection_frozen.json")
    check("selection_manifest_hash", file_sha256(output / "selection_frozen.json") == summary["selection_freeze_sha256"])
    check("selection_before_refit_comparison", selection["all_selected_before_refit_comparison"])
    last = torch.load(output / "learned" / "last.pt", map_location="cpu", weights_only=True)
    trained = summary["learned_training"]
    check("50_epochs", last["epoch"] == config["epochs"] == 50)
    check("full_resume_state", all(key in last for key in ("optimizer", "scheduler", "scaler", "torch_rng", "cuda_rng", "order_rng", "augment_rng", "assignment_trace")))
    check("18750_update_attempts", last["updates"] == trained["optimizer_updates_attempted"] == 18750)
    check("74922_training_variables", trained["training_real_parameters"] == 74922)
    check("tracked_gradients_and_updates", all(trained["first_update"]["checks"].values()))
    route_audit = next(row for row in trained["first_update"]["parameters"] if row["name"] == "router.logits")
    check("routing_gradient_and_update", route_audit["gradient_norm"] > 0 and route_audit["max_abs_update"] > 0)
    check("exact_hard_export", trained["hard_export_max_abs_error"] == trained["reload_max_abs_error"] == 0)
    check("freeze_does_not_change_predictions", trained["hardening"]["max_abs_logit_shift"] == trained["hardening"]["accuracy_drop"] == 0)
    before = torch.load(output / "learned" / "freeze_before.pt", map_location="cpu", weights_only=True)
    after = torch.load(output / "learned" / "freeze_after.pt", map_location="cpu", weights_only=True)
    check("both_full_freeze_boundaries", before["epoch"] == after["epoch"] == 40 and before["updates"] == after["updates"] == 15000)
    check("freeze_only_changes_hard_flag", all(torch.equal(value, after["model"][name]) for name, value in before["model"].items() if name != "router.hard_enabled"))
    check("correct_phase_flags", not bool(before["model"]["router.hard_enabled"]) and bool(after["model"]["router.hard_enabled"]))
    for key in ("torch_rng", "order_rng", "augment_rng"):
        check(f"freeze_preserves_rng:{key}", torch.equal(before[key], after[key]))
    with (output / "learned" / "training_discrete.csv").open(encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    check("canonical_contiguous_history", [int(row["epoch"]) for row in history] == list(range(1, 51)))
    check("canonical_discrete_phase_names", all(row["phase_after_epoch"] == ("adaptive_hard" if int(row["epoch"]) < 40 else "fixed_hard") for row in history))
    eligible = [row for row in history if int(row["epoch"]) >= 40]
    best = min(eligible, key=lambda row: float(row["validation_loss"]))
    check("learned_minimum_validation_ce_selected", int(best["epoch"]) == trained["best_epoch"] == selection["selected_epochs"]["learned"])
    check("selected_validation_preserved", float(best["validation_loss"]) == trained["selected_validation"]["cross_entropy"])
    with (reference / "random" / "training.csv").open(encoding="utf-8") as handle:
        control_history = list(csv.DictReader(handle))
    control_best = min((row for row in control_history if int(row["epoch"]) >= 40), key=lambda row: float(row["validation_loss"]))
    check("same_control_selection_rule", int(control_best["epoch"]) == selection["selected_epochs"]["random"])
    trace = read(output / "learned" / "assignment_trace.json")
    check("trace_matches_checkpoint", trace == last["assignment_trace"])
    check("full_assignment_trace", [row["epoch"] for row in trace] == list(range(1, 51)))
    check("trace_permutations_valid", all(sorted(row["matching"]) == list(range(32)) for row in trace))
    check("assignment_fixed_after_40", all(row["matching"] == trace[39]["matching"] for row in trace[40:]))
    check("switch_counter_monotonic", all(a["assignment_switch_events"] <= b["assignment_switch_events"] for a, b in zip(trace, trace[1:])))
    static_prefix = [row["epoch"] for row in trace if row["assignment_switch_events"] == 0]
    trajectory_keys = ("train_augmented_loss", "train_augmented_accuracy", "validation_loss", "validation_accuracy", "learning_rate")
    check("paired_trajectory_before_first_switch", all(
        float(history[epoch-1][key]) == float(control_history[epoch-1][key])
        for epoch in static_prefix for key in trajectory_keys))
    torch.set_num_threads(config["cpu_threads"])
    seed_everything(config["seed"])
    device = torch.device(config["device"])
    (images, labels), unused_test = load_cached_cifar(ROOT / config["archive"])
    del unused_test
    with np.load(reference / "split_indices.npz") as split:
        indices = {key: split[key] for key in ("train", "validation", "calibration")}
    check("disjoint_data", len(np.unique(np.concatenate(list(indices.values())))) == 50000 and len(indices["train"]) == 48000)
    with np.load(output / "heldout_targets.npz") as targets:
        for split in study["evaluation_splits"]:
            check(f"{split}:same_indices", np.array_equal(targets[f"{split}_indices"], indices[split]))
            check(f"{split}:same_labels", np.array_equal(targets[f"{split}_labels"], labels[indices[split]]))
    stats = read(reference / "data.json")["statistics"]
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    paths = {"random": reference / "random" / "model.pt", "learned": output / "learned" / "model.pt"}
    bases = {name: torch.load(path, map_location="cpu", weights_only=True)["model"] for name, path in paths.items()}
    for name, path in paths.items():
        check(f"{name}:selected_source_hash", file_sha256(path) == selection["selected_checkpoint_hashes"][name])
        check(f"{name}:no_dense_router_in_export", "router.logits" not in bases[name])
    initial, chosen = bases["random"]["router.indices"].tolist(), bases["learned"]["router.indices"].tolist()
    check("paired_initial_assignment", initial == last["model"]["router.initial_indices"].tolist())
    check("export_matches_best_state", all(torch.equal(value, last["best_state"][name]) for name, value in bases["learned"].items()))
    check("learned_membership_accounting", summary["membership"]["changed_positions"] == sum(a != b for a, b in zip(initial, chosen)) and summary["membership"]["new_unordered_groups"] == len(member_sets(chosen)-member_sets(initial)))
    allowed = {f"{name}.{suffix}" for name in study["bn_layers"] for suffix in ("running_mean", "running_var")}
    for condition, row in summary["conditions"].items():
        name = condition.split("_")[0]
        folder = output / condition
        checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        state = checkpoint["model"]
        changed = {key for key, value in state.items() if not torch.equal(value, bases[name][key])}
        refitted = condition.endswith("recalibrated")
        check(f"{condition}:only_bn_changes", changed <= allowed if refitted else not changed)
        check(f"{condition}:state_changed_as_declared", bool(changed) if refitted else not changed)
        model = load_export(folder / "model.pt", device)
        check(f"{condition}:weights_bitwise_unchanged", all(torch.equal(value.detach().cpu(), bases[name][key]) for key, value in model.named_parameters()))
        check(f"{condition}:real_parameter_count", sum(p.numel() for p in model.parameters()) == row["real_parameters"] == 73898)
        check(f"{condition}:buffer_count", sum(b.numel() for b in model.buffers()) == row["buffer_elements"] == 934)
        check(f"{condition}:permutation_preserved", torch.equal(model.router.indices.cpu(), bases[name]["router.indices"]))
        check(f"{condition}:checkpoint_hash", file_sha256(folder / "model.pt") == row["checkpoint_sha256"])
        check(f"{condition}:predictions_hash", file_sha256(folder / "predictions.npz") == row["predictions_sha256"])
        check(f"{condition}:metrics_record", read(folder / "metrics.json") == row)
        if refitted:
            check(f"{condition}:bn_refit_invariants", all(row["bn_refit"]["checks"].values()))
            check(f"{condition}:training_only_fit_count", all(layer["images"] == 48000 for layer in row["bn_refit"]["layers"]))
            check(f"{condition}:expected_bn_layers", [layer["name"] for layer in row["bn_refit"]["layers"]] == study["bn_layers"])
            check(f"{condition}:no_labels_in_fit", not row["bn_refit"]["labels_used"])
        with np.load(folder / "predictions.npz") as saved:
            for split in study["evaluation_splits"]:
                x = torch.as_tensor(images[indices[split]], device=device)
                y = torch.as_tensor(labels[indices[split]], device=device)
                metrics, logits = evaluate(model, x, y, mean, std)
                check(f"{condition}:{split}:reload_logits", np.array_equal(logits, saved[f"{split}_logits"]))
                check(f"{condition}:{split}:metrics", all(abs(metrics[key]-row["evaluation"][split][key]) < 1e-10 for key in metrics))
        del model
    a, b = [summary["conditions"][name]["evaluation"] for name in ("learned_recalibrated", "random_recalibrated")]
    for split in study["evaluation_splits"]:
        check(f"{split}:effect", abs(summary["effect"][f"{split}_accuracy_difference_pp"]-100*(a[split]["accuracy"]-b[split]["accuracy"])) < 1e-10)
    gate = {"calibration_gain": a["calibration"]["accuracy"] >= b["calibration"]["accuracy"]+study["minimum_calibration_gain_for_replication"]-1e-12,
            "validation_quality": a["validation"]["accuracy"] >= b["validation"]["accuracy"]-study["maximum_validation_deficit_for_replication"]-1e-12,
            "membership_changed": len(member_sets(chosen)-member_sets(initial)) > 0}
    gate["promising_for_replication"] = all(gate.values())
    check("replication_gate_recomputed", gate == summary["replication_gate"])
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
              "check_count": len(checks), "verifier_sha256": file_sha256(Path(__file__)), "official_test_evaluated": False}
    write_json(output / "verification.json", result)
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed": [key for key, value in checks.items() if not value]}, indent=2), flush=True)
    if result["status"] != "passed":
        raise AssertionError("Discrete-forward artifact checks failed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/hard_forward_grouping")
    args = parser.parse_args()
    verify(ROOT / args.output)
