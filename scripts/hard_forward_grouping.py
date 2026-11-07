"""One-seed discrete-forward grouping pilot, with a reused paired control."""

import argparse
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import platform
import sys
import traceback
from unittest.mock import patch

import numpy as np
import torch

from scripts import learned_grouping as pilot
from scripts.cifar_cnn import write_json
from src.cnn_training import evaluate, first_update_audit, prepare_batch, seed_everything
from src.data.cifar_cached import file_sha256, load_cached_cifar
from src.evaluation.routing_hardening import recalibrate_bn_layers
from src.models.channel_routing import RoutedCifarCNN
from src.models.hard_forward_routing import HardForwardChannelPermutation, make_model

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
SOURCES = ["scripts/hard_forward_grouping.py", "src/models/hard_forward_routing.py", "src/evaluation/routing_hardening.py"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def groups(indices):
    values = np.asarray(indices)
    return {tuple(sorted(values[unit+np.arange(4)*(len(values)//4)].tolist())) for unit in range(len(values)//4)}


def train(config, tensors, labels, mean, std, folder, hashes):
    """Reuse the frozen numerical trainer; hooks only add captures/metadata.

    Its legacy 'soft' phase and 'soft_validation' keys denote adaptive HARD
    routing here. Canonical names are in training_discrete.csv/freeze.json.
    """
    folder = Path(folder)
    original_save, original_harden = pilot.save_last, HardForwardChannelPermutation.harden
    live, capture, assignment_trace = {}, {}, []
    if (folder / "last.pt").exists():
        assignment_trace = torch.load(folder / "last.pt", map_location="cpu", weights_only=True).get("assignment_trace", [])

    def construct(configuration, mode):
        model = make_model(configuration, mode)
        if mode == "learned":
            live["model"] = model
        return model

    def harden(router):
        capture["before"] = {name: value.detach().cpu().clone() for name, value in live["model"].state_dict().items()}
        original_harden(router)

    def save(path, state):
        model = state["model"]
        indices = model["router.indices"].detach().cpu().tolist()
        initial = model["router.initial_indices"].detach().cpu().tolist()
        assignment_trace.append({"epoch": state["epoch"], "matching": indices,
                                 "changed_positions_from_initial": sum(a != b for a, b in zip(initial, indices)),
                                 "new_unordered_groups": len(groups(indices)-groups(initial)),
                                 "assignment_switch_events": int(model["router.assignment_switch_events"]),
                                 "assignment_switched_positions": int(model["router.assignment_switched_positions"])})
        state["assignment_trace"] = deepcopy(assignment_trace)
        if state["epoch"] == config["soft_epochs"]:
            if "before" not in capture:
                raise RuntimeError("Missing pre-freeze model capture")
            before = {**state, "model": capture["before"], "best_state": None,
                      "best_validation": None, "best_epoch": None,
                      "boundary_note": "Adaptive discrete-forward model before assignment freeze; scheduler is in the ordinary post-epoch state."}
            original_save(folder / "freeze_before.pt", before)
            original_save(folder / "freeze_after.pt", state)
        original_save(path, state)

    def progress(*args, **kwargs):
        if args and isinstance(args[0], str):
            args = (args[0].replace("(soft)", "(discrete, adaptive)").replace("(hard)", "(discrete, fixed)")
                    .replace("HARDENING:", "FREEZING ASSIGNMENT:").replace("soft val=", "adaptive hard val="), *args[1:])
        print(*args, **kwargs)

    with patch.object(pilot, "make_model", construct), patch.object(pilot, "save_last", save), \
            patch.object(HardForwardChannelPermutation, "harden", harden), patch.object(pilot, "print", progress, create=True):
        result = pilot.train_mode(config, "learned", tensors, labels, mean, std, folder, hashes)
    last = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
    history = [{**row, "phase_after_epoch": "adaptive_hard" if row["epoch"] < config["soft_epochs"] else "fixed_hard"} for row in last["history"]]
    with (folder / "training_discrete.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    write_json(folder / "assignment_trace.json", last["assignment_trace"])
    event = result["hardening"]
    write_json(folder / "freeze.json", {"epoch": event["epoch"], "before_freeze_validation": event["soft_validation"],
               "after_freeze_validation": event["immediate_hard_validation"], "max_abs_logit_shift": event["max_abs_logit_shift"],
               "forward_was_always_discrete": True, "legacy_soft_keys_are_schema_aliases": True})
    if event["max_abs_logit_shift"] != 0 or event["accuracy_drop"] != 0:
        raise AssertionError("Freezing an already-discrete assignment changed predictions")
    return result


def smoke(config):
    device = torch.device(config["device"])
    seed_everything(config["seed"])
    model = make_model(config, "learned").to(device).eval()
    seed_everything(config["seed"])
    control = make_model(config, "random").to(device).eval()
    checks = {"paired_backbone_initialization": all(torch.equal(value, dict(control.named_parameters())[name]) for name, value in model.named_parameters() if name != "router.logits"),
              "paired_initial_permutation": torch.equal(model.router.indices, control.router.indices)}
    inputs = torch.randn(8, 3, 32, 32, device=device)
    errors = {}
    for enabled in (False, True):
        with torch.no_grad(), torch.amp.autocast(device.type, enabled=enabled and device.type == "cuda"):
            error = float((model(inputs)-control(inputs)).abs().max())
        errors["amp" if enabled else "fp32"] = error
    checks["initial_control_forward_equivalence"] = all(error == 0 for error in errors.values())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    audit = first_update_audit(model, inputs, torch.arange(8, device=device)%10, optimizer)
    checks["all_tracked_variables_and_quaternion_components_updated"] = all(audit["checks"].values())
    route = next(row for row in audit["parameters"] if row["name"] == "router.logits")
    checks["surrogate_gradient_and_logit_update"] = route["gradient_norm"] > 0 and route["max_abs_update"] > 0
    model.eval()
    with torch.no_grad():
        before = model(inputs)
        model.router.harden()
        after = model(inputs)
        exported = model.export_hard()
        final = exported(inputs)
    checks["freeze_and_export_do_not_change_forward"] = torch.equal(before, after) and torch.equal(after, final)
    checks["training_parameter_count"] = sum(p.numel() for p in model.parameters()) == 74922
    checks["export_parameter_count"] = sum(p.numel() for p in exported.parameters()) == 73898
    if not all(checks.values()):
        raise AssertionError(f"Hard-forward smoke failed: {checks}; errors={errors}")
    return {"status": "passed", "checks": checks, "initial_forward_errors": errors, "first_update": audit,
            "scope": "Synthetic GPU integrity check, not CIFAR quality"}


def write_report(output, summary):
    rows = ["# Discrete-forward learned grouping: one-seed pilot", "", "## Hypotheses", "",
            "Using a discrete permutation throughout training may avoid the soft-to-hard operator gap. "
            "Whether learned group membership improves held-out quality over paired fixed random membership is unconfirmed.", "",
            "## Protocol", "",
            "Seed 2026; unchanged CIFAR backbone, split, normalization, augmentation, optimizer and 50-epoch schedule. "
            "Exact matching of the routing logits at every adaptive forward; Sinkhorn straight-through surrogate in backward. "
            "This is a biased surrogate, not the true derivative of the discrete assignment. "
            "Learn assignments for 40 epochs, freeze for 10. Choose minimum original-BN validation CE in epochs 40-50. "
            "Reuse the paired fixed-random control's completed 50 epochs, selected by the same rule. "
            "Refit the same four downstream BNs, layer-wise on 48,000 training images, in both selected models. "
            "Validation/calibration each have 1,000 held-out images; no new official-test inference.", "",
            "## Measured results", "",
            "| Condition | Validation accuracy | Validation CE | Calibration accuracy | Calibration CE |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for name, item in summary["conditions"].items():
        v, c = item["evaluation"]["validation"], item["evaluation"]["calibration"]
        rows.append(f"| {name} | {100*v['accuracy']:.2f}% | {v['cross_entropy']:.6f} | {100*c['accuracy']:.2f}% | {c['cross_entropy']:.6f} |")
    m, e = summary["membership"], summary["effect"]
    rows += ["", f"Selected learned checkpoint: epoch {summary['learned_training']['best_epoch']}. "
             f"Final exported assignment changes {m['changed_positions']}/32 positions and creates {m['new_unordered_groups']}/8 new unordered groups.",
             f"BN-refitted learned-minus-random effect: validation {e['validation_accuracy_difference_pp']:+.2f} pp; "
             f"calibration {e['calibration_accuracy_difference_pp']:+.2f} pp. "
             f"Prespecified replication gate: {summary['replication_gate']['promising_for_replication']}.", "",
             "## Interpretation and limits", "",
             "This tests one deterministic surrogate, seed, schedule and initialization. "
             "An unchanged final assignment is not evidence of useful learned membership. "
             "A positive gate only motivates replication; it is not statistical significance, a general gain, "
             "a novel method, or a compression result. A negative outcome does not rule out all grouping methods. "
             "Previously explored held-out data are not a new blind benchmark.", "",
             "The learned training adds 1,024 real logit parameters: 74,922 vs 73,898. "
             "It also adds matching, Sinkhorn and surrogate-backward compute. Final fixed-index exports "
             "both have 73,898 real parameters and 934 buffer elements. No backbone channel is removed.", "",
             "The unmodified trainer's legacy 'soft' keys mean adaptive DISCRETE routing in this run. "
             "Canonical phases and freeze metrics are in learned/training_discrete.csv and learned/freeze.json. "
             "Both complete freeze-boundary checkpoints and all resume RNG/optimizer states are retained.", "",
             "Prior art: Bengio2013STE, Mena2018, Wang2023PSTE, Wang2019FLGC, Zhang2021StructuredShuffle "
             "in references.bib. This is not a faithful reproduction of those complete methods.", ""]
    (output / "REPORT.md").write_text("\n".join(rows), encoding="utf-8")


def plot(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, split in zip(axes, ("validation", "calibration")):
        values = [100*row["evaluation"][split]["accuracy"] for row in summary["conditions"].values()]
        bars = ax.bar(["Fixed\noriginal BN", "Learned\noriginal BN", "Fixed\nrefitted BN", "Learned\nrefitted BN"], values,
                      color=["#326ea8", "#bb5545", "#709cc1", "#55a077"])
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set_ylim(0, 100)
        ax.set_title(f"{split.capitalize()} (n=1,000)")
        ax.set_ylabel("Accuracy (%)")
    fig.suptitle("Discrete-forward grouping: one seed; equal 50-epoch budgets; fixed random control reused")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)


def run(study):
    reference, output = ROOT / study["pilot_reference"], ROOT / study["output"]
    config, prior = read(ROOT / study["training_config"]), read(reference / "protocol.json")
    if config != prior["config"] or study["official_test_evaluation"]:
        raise ValueError("Require the exact paired pilot configuration and no test inference")
    for name, digest in prior["source_hashes"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f"Frozen core changed: {name}")
    hashes = {**prior["source_hashes"], **{name: file_sha256(ROOT / name) for name in SOURCES}}
    protected_paths = [reference / "protocol.json", reference / "summary.json", reference / "split_indices.npz", reference / "data.json",
                       reference / "random" / "model.pt", reference / "random" / "last.pt", reference / "random" / "metrics.json", reference / "random" / "training.csv"]
    protected = {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in protected_paths}
    protocol = {"study": study, "training_config": config, "source_hashes": hashes, "protected_inputs": protected,
                "seed": config["seed"], "environment": {"python": sys.version, "torch": str(torch.__version__), "cuda": torch.version.cuda,
                                                         "numpy": np.__version__, "platform": platform.platform()},
                "official_test_evaluated": False, "trainable_parameter_budgets_equal": False,
                "control_training_reused": True}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists() and read(output / "protocol.json") != protocol:
        raise ValueError("Hard-forward study changed; preserve this run and use a new output folder")
    write_json(output / "protocol.json", protocol)
    if (output / "summary.json").exists():
        summary = read(output / "summary.json")
        write_report(output, summary)
        plot(output, summary)
        return summary
    torch.set_num_threads(config["cpu_threads"])
    if not (output / "smoke.json").exists():
        write_json(output / "smoke.json", smoke(config))
    if read(output / "smoke.json")["status"] != "passed":
        raise ValueError("Synthetic tracking/update gate failed")
    device = torch.device(config["device"])
    (images, targets), unused_test = load_cached_cifar(ROOT / config["archive"])
    del unused_test
    with np.load(reference / "split_indices.npz") as split:
        indices = {key: split[key] for key in ("train", "validation", "calibration")}
    if len(indices["train"]) != 48000 or len(np.unique(np.concatenate(list(indices.values())))) != 50000:
        raise ValueError("Unexpected split or overlap")
    statistics = read(reference / "data.json")["statistics"]
    mean = torch.tensor(statistics["mean"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(statistics["std"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    tensors = {key: torch.as_tensor(images[index], device=device) for key, index in indices.items()}
    labels = {key: torch.as_tensor(targets[index], device=device) for key, index in indices.items()}
    learned = train(config, tensors, labels, mean, std, output / "learned", hashes)
    selected_paths = {"random": reference / "random" / "model.pt", "learned": output / "learned" / "model.pt"}
    source_metrics = {"random": read(reference / "random" / "metrics.json"), "learned": learned}
    write_json(output / "selection_frozen.json", {"selected_checkpoint_hashes": {key: file_sha256(path) for key, path in selected_paths.items()},
               "selection_rule": study["selection"], "selected_epochs": {key: value["best_epoch"] for key, value in source_metrics.items()},
               "all_selected_before_refit_comparison": True, "official_test_evaluated": False})
    np.savez_compressed(output / "heldout_targets.npz", **{f"{key}_labels": targets[indices[key]] for key in study["evaluation_splits"]},
                        **{f"{key}_indices": indices[key] for key in study["evaluation_splits"]})
    records, loaded = {}, {}
    for key, path in selected_paths.items():
        loaded[key] = pilot.load_export(path, device)
    for condition, key, refit in (("random_original", "random", False), ("learned_original", "learned", False),
                                  ("random_recalibrated", "random", True), ("learned_recalibrated", "learned", True)):
        model = deepcopy(loaded[key]).eval()
        calibration = recalibrate_bn_layers(model, tensors["train"], mean, std, study["bn_layers"], study["bn_batch_size"]) if refit else None
        metrics, predictions = {}, {}
        for split in study["evaluation_splits"]:
            metrics[split], predictions[f"{split}_logits"] = evaluate(model, tensors[split], labels[split], mean, std)
        if not refit and any(abs(metrics["validation"][name]-source_metrics[key]["selected_validation"][name]) > 1e-8 for name in metrics["validation"]):
            raise AssertionError("Selected-model validation mismatch")
        folder = output / condition
        folder.mkdir(exist_ok=True)
        torch.save({"model": {name: value.cpu() for name, value in model.state_dict().items()}, "config": config,
                    "condition": condition, "source_epoch": source_metrics[key]["best_epoch"]}, folder / "model.pt")
        np.savez_compressed(folder / "predictions.npz", **predictions)
        row = {"condition": condition, "evaluation": metrics, "bn_refit": calibration,
               "real_parameters": sum(p.numel() for p in model.parameters()), "buffer_elements": sum(b.numel() for b in model.buffers()),
               "checkpoint_sha256": file_sha256(folder / "model.pt"), "predictions_sha256": file_sha256(folder / "predictions.npz")}
        write_json(folder / "metrics.json", row)
        records[condition] = row
        print(f"{condition}: val={100*metrics['validation']['accuracy']:.2f}%, calibration={100*metrics['calibration']['accuracy']:.2f}%", flush=True)
    final_indices = loaded["learned"].router.indices.cpu().tolist()
    initial_indices = loaded["random"].router.indices.cpu().tolist()
    membership = {"initial_matching": initial_indices, "exported_matching": final_indices,
                  "changed_positions": sum(a != b for a, b in zip(initial_indices, final_indices)),
                  "new_unordered_groups": len(groups(final_indices)-groups(initial_indices))}
    a, b = records["learned_recalibrated"]["evaluation"], records["random_recalibrated"]["evaluation"]
    effect = {f"{key}_accuracy_difference_pp": 100*(a[key]["accuracy"]-b[key]["accuracy"]) for key in study["evaluation_splits"]}
    gate = {"calibration_gain": a["calibration"]["accuracy"] >= b["calibration"]["accuracy"]+study["minimum_calibration_gain_for_replication"]-1e-12,
            "validation_quality": a["validation"]["accuracy"] >= b["validation"]["accuracy"]-study["maximum_validation_deficit_for_replication"]-1e-12,
            "membership_changed": membership["new_unordered_groups"] > 0}
    gate["promising_for_replication"] = all(gate.values())
    if any(file_sha256(ROOT / name) != digest for name, digest in protected.items()):
        raise AssertionError("Frozen reference artifacts changed")
    summary = {"status": "completed", "protocol": protocol, "learned_training": learned, "conditions": records,
               "membership": membership, "effect": effect, "replication_gate": gate,
               "selection_freeze_sha256": file_sha256(output / "selection_frozen.json"),
               "official_test_evaluated": False, "physical_channel_savings": 0}
    write_json(output / "summary.json", summary)
    write_report(output, summary)
    plot(output, summary)
    print(json.dumps({"effect": effect, "membership": membership, "replication_gate": gate}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hard_forward_grouping.json")
    args = parser.parse_args()
    study = read(ROOT / args.config)
    try:
        run(study)
    except Exception:
        output = ROOT / study["output"]
        output.mkdir(parents=True, exist_ok=True)
        (output / "FAILED_CHECKS.md").write_text("# Failed execution/check\n\n```text\n"+traceback.format_exc()+"\n```\nNo gain is claimed from this failure.\n", encoding="utf-8")
        raise
