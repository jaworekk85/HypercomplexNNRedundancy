"""Independently verify diagnostic weights, data boundaries, and saved logits."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.cifar_cnn import write_json
from scripts.learned_grouping import make_model
from src.cnn_training import evaluate, seed_everything
from src.data.cifar_cached import file_sha256, load_cached_cifar

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(output):
    output = Path(output)
    protocol, summary = read(output / "intervention_protocol.json"), read(output / "summary.json")
    diagnostic = protocol["diagnostic"]
    reference = ROOT / diagnostic["pilot"]
    frozen = read(reference / "protocol.json")
    config = frozen["config"]
    checks = {}

    def check(name, value):
        checks[name] = bool(value)

    for name, digest in {**protocol["source_hashes"], **protocol["input_hashes"]}.items():
        check(f"hash:{name}", file_sha256(ROOT / name) == digest)
    replay_protocol = read(output / "replay_protocol.json")
    check("replay_wrapper_unchanged", file_sha256(ROOT / "scripts/replay_routing_boundary.py") == replay_protocol["replay_script_sha256"])
    for name, digest in replay_protocol["protected_inputs"].items():
        check(f"original_preserved:{name}", file_sha256(ROOT / name) == digest)
    check("replay_passed", read(output / "replay_verification.json")["status"] == "passed")
    check("no_official_test", not protocol["official_test_evaluated"] and not summary["official_test_evaluated"])
    check("no_optimizer_updates", protocol["weight_updates_after_boundary"] == summary["weight_updates_after_boundary"] == 0)
    check("not_claimed_as_compression", summary["compact_parameter_savings"] == 0)
    with np.load(reference / "split_indices.npz") as split:
        indices = {key: split[key] for key in ("train", "validation", "calibration")}
    check("48000_training_images", len(indices["train"]) == diagnostic["bn_fit_samples"] == 48000)
    check("held_out_disjoint", not any(np.intersect1d(indices[a], indices[b]).size for a, b in (("train", "validation"), ("train", "calibration"), ("validation", "calibration"))))
    torch.set_num_threads(config["cpu_threads"])
    seed_everything(config["seed"])
    device = torch.device(config["device"])
    (images, labels), unused_test = load_cached_cifar(ROOT / config["archive"])
    del unused_test
    stats = read(reference / "data.json")["statistics"]
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    boundary = {phase: torch.load(output / "replay" / f"boundary_{phase}.pt", map_location="cpu", weights_only=True) for phase in ("soft", "hard")}
    check("boundary_update_count", boundary["soft"]["updates"] == boundary["hard"]["updates"] == 15000)
    check("tracked_variables_gradients_and_updates", all(boundary["hard"]["audit"]["checks"].values()))
    router_audit = next(row for row in boundary["hard"]["audit"]["parameters"] if row["name"] == "router.logits")
    check("router_gradient_and_update", router_audit["gradient_norm"] > 0 and router_audit["max_abs_update"] > 0)
    original_hardening = read(reference / "learned" / "hardening.json")
    allowed_bn = {f"{name}.{suffix}" for name in diagnostic["bn_layers"] for suffix in ("running_mean", "running_var")}
    with np.load(output / "evaluation_targets.npz") as targets:
        for split in diagnostic["evaluation_splits"]:
            check(f"{split}:indices", np.array_equal(targets[f"{split}_indices"], indices[split]))
            check(f"{split}:labels", np.array_equal(targets[f"{split}_labels"], labels[indices[split]]))
    for condition in diagnostic["conditions"]:
        folder = output / condition
        row = read(folder / "metrics.json")
        checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        phase = condition.split("_")[0]
        base = boundary[phase]["model"]
        state = checkpoint["model"]
        changed = {name for name in state if not torch.equal(state[name], base[name])}
        recalibrated = condition.endswith("recalibrated")
        check(f"{condition}:only_allowed_buffers_changed", changed <= allowed_bn if recalibrated else not changed)
        check(f"{condition}:buffers_did_change", bool(changed) if recalibrated else not changed)
        model = make_model(config, "learned").to(device).eval()
        model.load_state_dict(state)
        check(f"{condition}:weights_unchanged", all(torch.equal(value.detach().cpu(), base[name]) for name, value in model.named_parameters()))
        check(f"{condition}:routing_phase", bool(model.router.hard_enabled) == (phase == "hard"))
        check(f"{condition}:valid_permutation", torch.equal(model.router.indices.sort().values.cpu(), torch.arange(32)))
        check(f"{condition}:parameter_accounting", sum(p.numel() for p in model.parameters()) == row["retained_real_parameter_elements"] == 74922)
        check(f"{condition}:checkpoint_hash", file_sha256(folder / "model.pt") == row["checkpoint_sha256"])
        check(f"{condition}:prediction_hash", file_sha256(folder / "predictions.npz") == row["predictions_sha256"])
        check(f"{condition}:summary_matches_record", summary["conditions"][condition] == row)
        check(f"{condition}:no_training_or_test", row["weight_updates_after_boundary"] == 0 and not row["official_test_evaluated"])
        if recalibrated:
            check(f"{condition}:refit_invariants", all(row["refit"]["checks"].values()))
            check(f"{condition}:refit_layers", [item["name"] for item in row["refit"]["layers"]] == diagnostic["bn_layers"])
            check(f"{condition}:refit_samples", all(item["images"] == 48000 for item in row["refit"]["layers"]))
            check(f"{condition}:no_refit_labels", not row["refit"]["labels_used"])
        with np.load(folder / "predictions.npz") as predictions:
            for split in diagnostic["evaluation_splits"]:
                x = torch.as_tensor(images[indices[split]], device=device)
                y = torch.as_tensor(labels[indices[split]], device=device)
                metrics, logits = evaluate(model, x, y, mean, std)
                saved = predictions[f"{split}_logits"]
                check(f"{condition}:{split}:reload_logits", np.array_equal(logits, saved))
                check(f"{condition}:{split}:metrics", all(abs(metrics[key]-row["evaluation"][split][key]) < 1e-10 for key in metrics))
                if split == "validation" and not recalibrated:
                    expected = original_hardening["soft_validation" if phase == "soft" else "immediate_hard_validation"]
                    check(f"{condition}:original_boundary_reproduced", all(abs(metrics[key]-expected[key]) <= diagnostic["replay_metric_tolerance"] for key in metrics))
        del model
    for split in diagnostic["evaluation_splits"]:
        rows = summary["conditions"]
        s, h, sr, hr = [rows[name]["evaluation"][split]["accuracy"] for name in diagnostic["conditions"]]
        contrast = summary["contrasts"][split]
        check(f"{split}:original_gap", abs(contrast["original_soft_minus_hard_accuracy_pp"] - 100*(s-h)) < 1e-10)
        check(f"{split}:refitted_gap", abs(contrast["refitted_soft_minus_hard_accuracy_pp"] - 100*(sr-hr)) < 1e-10)
        check(f"{split}:hard_recovery", abs(contrast["hard_refit_accuracy_gain_pp"] - 100*(hr-h)) < 1e-10)
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
              "check_count": len(checks), "official_test_evaluated": False,
              "verifier_sha256": file_sha256(Path(__file__))}
    write_json(output / "verification.json", result)
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed": [name for name, value in checks.items() if not value]}, indent=2), flush=True)
    if result["status"] != "passed":
        raise AssertionError("BN intervention artifact verification failed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/routing_bn_diagnostic")
    arguments = parser.parse_args()
    verify(ROOT / arguments.output)
