"""Independently check reconstruction fits, selection, surgery and predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.cifar_cnn import write_json
from scripts.quaternion_merge_compression import load_case
from src.cnn_training import evaluate, seed_everything
from src.data.cifar_cached import file_sha256, load_cached_cifar

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def centered_mse(values):
    return float(np.square(values-values.mean(1, keepdims=True)).mean())


def independent_errors(values, weights, pairs):
    total = {name: np.zeros((len(values), len(weights))) for name in ("merged", "mean_only", "pruned")}
    for row in pairs:
        x, y = values[:, row["keeper"]], values[:, row["victim"]]
        w = weights[:, row["victim"]]
        total["merged"] += (y-x@np.asarray(row["matrix"]).T-row["bias"])@w.T
        total["mean_only"] += (y-row["target_fit_mean"])@w.T
        total["pruned"] += y@w.T
    errors = {name: centered_mse(value) for name, value in total.items()}
    for name, key in (("ratio_vs_mean_only", "mean_only"), ("ratio_vs_pruned", "pruned")):
        errors[name] = errors["merged"]/errors[key] if errors[key] > 1e-12 else None
    return errors


def close_dict(a, b, tolerance=1e-9):
    return all((a[key] is None and b[key] is None) or
               (a[key] is not None and b[key] is not None and abs(a[key]-b[key]) <= tolerance*(1+abs(b[key]))) for key in a)


def verify(output):
    output = Path(output)
    diagnostic = read(output / "diagnostic_summary.json")
    protocol, config = diagnostic["protocol"], diagnostic["protocol"]["config"]
    checks = {}

    def check(name, value):
        checks[name] = bool(value)

    for name, digest in {**protocol["source_hashes"], **protocol["protected_inputs"]}.items():
        check(f"frozen:{name}", file_sha256(ROOT / name) == digest)
    gate_manifest = read(output / "gate_frozen.json")
    check("diagnostic_gate_frozen", file_sha256(output / "diagnostic_summary.json") == gate_manifest["diagnostic_summary_sha256"])
    check("diagnostic_no_labels_or_training", not protocol["fit_labels_used"] and protocol["new_cnn_training_updates"] == 0)
    passing = {mode: 0 for mode in config["compression_maps"]}
    source_data = {}
    for record in diagnostic["records"]:
        seed, source = record["seed"], ROOT / record["source"]
        with np.load(source / "quaternion" / "calibration_activations.npz") as cached:
            pooled = cached["trained"].astype(np.float64).mean(1)
        checkpoint = torch.load(source / "quaternion" / "model.pt", map_location="cpu", weights_only=True)
        state = checkpoint["model"]
        weights = state["readout.weight"].double().numpy().reshape(10, 4, 32).transpose(0, 2, 1)
        with np.load(source / "split_indices.npz") as split:
            indices = {key: split[key] for key in ("train", "calibration", "validation")}
        check(f"{seed}:disjoint_original_splits", len(np.unique(np.concatenate(list(indices.values())))) == 50000)
        with np.load(output / f"seed_{seed}" / "calibration_partition.npz") as partition:
            check(f"{seed}:whole_image_fit_check_partition", np.array_equal(partition["fit_positions"], np.arange(0, 1000, 2)) and np.array_equal(partition["check_positions"], np.arange(1, 1000, 2)))
            check(f"{seed}:original_ids_preserved", np.array_equal(partition["fit_original_indices"], indices["calibration"][::2]) and np.array_equal(partition["check_original_indices"], indices["calibration"][1::2]))
        fit, holdout = pooled[::2], pooled[1::2]
        all_fits = {}
        for mode in config["diagnostic_maps"]:
            candidates = read(output / f"seed_{seed}" / f"{mode}_all_pairs.json")
            all_fits[mode] = candidates
            check(f"{seed}:{mode}:all_directed_pairs", {(row["keeper"], row["victim"]) for row in candidates} == {(a, b) for a in range(32) for b in range(32) if a != b})
            errors, normal_error, family_error = [], 0., 0.
            for row in candidates:
                a, b = np.asarray(row["matrix"]), np.asarray(row["bias"])
                x, y = fit[:, row["keeper"]], fit[:, row["victim"]]
                xm, ym = x.mean(0), y.mean(0)
                errors.extend(np.abs(np.asarray(row["target_fit_mean"])-ym).tolist())
                errors.extend(np.abs(b-(ym-a@xm)).tolist())
                residual = y-x@a.T-b
                xc = x-xm
                if mode == "full_real":
                    normal = xc.T@residual
                elif mode == "diagonal":
                    family_error = max(family_error, float(np.max(np.abs(a-np.diag(np.diag(a))))))
                    normal = (xc*residual).sum(0)
                else:
                    r, i, j, k = a[:, 0]
                    expected = np.array([[r, -i, -j, -k], [i, r, k, -j], [j, -k, r, i], [k, j, -i, r]])
                    family_error = max(family_error, float(np.max(np.abs(a-expected))))
                    xr, xi, xj, xk = xc.T
                    design = np.stack([xc, np.stack([-xi, xr, xk, -xj], 1), np.stack([-xj, -xk, xr, xi], 1), np.stack([-xk, xj, -xi, xr], 1)], axis=2)
                    normal = np.einsum("noc,no->c", design, residual)
                normal_error = max(normal_error, float(np.max(np.abs(normal))))
                for name, values in (("fit", fit), ("check", holdout)):
                    residual = values[:, row["victim"]]-values[:, row["keeper"]]@a.T-b
                    feature_mse = float(np.square(residual).mean())
                    score = centered_mse(residual@weights[:, row["victim"]].T)
                    errors.append(abs(feature_mse-row[f"{name}_feature_mse"]))
                    errors.append(abs(score-row[f"{name}_centered_logit_mse"]))
            check(f"{seed}:{mode}:fit_intercept_and_saved_errors", max(errors) < 1e-9)
            check(f"{seed}:{mode}:least_squares_stationarity", normal_error < 1e-7)
            check(f"{seed}:{mode}:declared_family", family_error < 1e-12)
            if mode in config["compression_maps"]:
                selected, used = [], set()
                for row in sorted((row for row in candidates if row["nonconstant_fit_target"]), key=lambda row: (row["fit_centered_logit_mse"], row["keeper"], row["victim"])):
                    if row["keeper"] not in used and row["victim"] not in used:
                        selected.append(row)
                        used.update((row["keeper"], row["victim"]))
                        if len(selected) == 8:
                            break
                stored = record["selected"][mode]
                check(f"{seed}:{mode}:fit_only_disjoint_selection", selected == stored["pairs"] and len(used) == 16)
                actual = independent_errors(holdout, weights, selected)
                check(f"{seed}:{mode}:heldout_aggregate_errors", close_dict(actual, stored["check_aggregate_errors"]))
                signal = actual["ratio_vs_mean_only"] is not None and actual["ratio_vs_pruned"] is not None and actual["ratio_vs_mean_only"] <= config["gate_maximum_check_mse_ratio_vs_mean_only"]+1e-12 and actual["ratio_vs_pruned"] <= config["gate_maximum_check_mse_ratio_vs_pruning"]+1e-12
                check(f"{seed}:{mode}:signal_gate", signal == stored["signal_passes"])
                passing[mode] += int(signal)
        source_data[seed] = {"source": source, "state": state, "pooled": pooled, "indices": indices, "config": checkpoint["config"]}
    check("cross_seed_gate_counts", diagnostic["gate"]["passing_seeds"] == passing)
    check("cross_seed_gate_decision", diagnostic["gate"]["supports_physical_comparison"] == any(value >= config["gate_minimum_passing_seeds"] for value in passing.values()))
    summary = read(output / "summary.json")
    if summary["status"] == "completed":
        physical = read(output / "physical_protocol.json")
        check("physical_runner_preserved", file_sha256(ROOT / "scripts/quaternion_merge_compression.py") == physical["physical_script_sha256"])
        for name, digest in physical["additional_protected_inputs"].items():
            check(f"baseline_preserved:{name}", file_sha256(ROOT / name) == digest)
        check("gate_manifest_preserved", file_sha256(output / "gate_frozen.json") == physical["gate_frozen_sha256"])
        frozen = read(output / "selection_frozen.json")
        check("all_choices_frozen_before_validation", frozen["all_choices_frozen_before_validation"] and file_sha256(output / "selection_frozen.json") == summary["selection_frozen_sha256"])
        check("expected_case_count", len(summary["rows"]) == len(frozen["manifests"]) == 51)
        check("no_training_bn_refit_or_test", summary["optimizer_updates"] == 0 and not summary["bn_recalibration"] and not summary["official_test_evaluated"])
        torch.set_num_threads(config["cpu_threads"])
        seed_everything(config["seeds"][0])
        device = torch.device(config["device"])
        sample = source_data[config["seeds"][0]]
        (images, labels), unused_test = load_cached_cifar(ROOT / sample["config"]["archive"])
        del unused_test
        stats = read(sample["source"] / "data.json")["statistics"]
        mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
        std = torch.tensor(stats["std"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
        for row, manifest in zip(summary["rows"], frozen["manifests"]):
            name = f"{row['seed']}:{row['case']}"
            folder = output / f"seed_{row['seed']}" / row["case"]
            check(f"{name}:manifest", read(folder / "manifest.json") == manifest and file_sha256(folder / "manifest.json") == row["manifest_sha256"])
            check(f"{name}:checkpoint_hash", file_sha256(ROOT / manifest["checkpoint_path"]) == row["checkpoint_sha256"] == manifest["checkpoint_sha256"])
            check(f"{name}:predictions_hash", file_sha256(folder / "predictions.npz") == row["predictions_sha256"])
            original = source_data[row["seed"]]["state"]
            kept = manifest["kept_units"]
            channels = [component*32+unit for component in range(4) for unit in kept]
            current = torch.load(ROOT / manifest["checkpoint_path"], map_location="cpu", weights_only=True)["model"]
            unchanged = set(original)-{"blocks.2.3.kernel", "blocks.2.4.weight", "blocks.2.4.bias", "blocks.2.4.running_mean", "blocks.2.4.running_var", "readout.weight", "readout.bias"}
            surgery_ok = all(torch.equal(original[key], current[key]) for key in unchanged)
            surgery_ok &= torch.equal(current["blocks.2.3.kernel"], original["blocks.2.3.kernel"][:, kept])
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                surgery_ok &= torch.equal(current[f"blocks.2.4.{suffix}"], original[f"blocks.2.4.{suffix}"][channels])
            check(f"{name}:only_exact_output_slices", surgery_ok and set(current) == set(original))
            expected_w = original["readout.weight"][:, channels].double().numpy().copy()
            expected_b = original["readout.bias"].double().numpy().copy()
            for pair in manifest["pairs"]:
                victim = pair["victim"]
                wv = original["readout.weight"][:, [component*32+victim for component in range(4)]].double().numpy()
                if manifest["compensation"] == "affine":
                    positions = [component*len(kept)+kept.index(pair["keeper"]) for component in range(4)]
                    expected_w[:, positions] += wv@np.asarray(pair["matrix"])
                    expected_b += wv@np.asarray(pair["bias"])
                elif manifest["compensation"] == "mean_only":
                    expected_b += wv@np.asarray(pair["target_fit_mean"])
            check(f"{name}:readout_fold_formula", np.allclose(current["readout.weight"].numpy(), expected_w, rtol=1e-5, atol=1e-6) and np.allclose(current["readout.bias"].numpy(), expected_b, rtol=1e-5, atol=1e-6))
            model = load_case(manifest, device)
            full = manifest["kind"] == "baseline"
            check(f"{name}:physical_counts", sum(p.numel() for p in model.parameters()) == row["real_parameters"] == (73898 if full else 64298) and sum(b.numel() for b in model.buffers()) == row["buffer_elements"] == (902 if full else 838))
            check(f"{name}:actual_compact_dimensions", model.blocks[2][3].out_channels == (32 if full else 24) and model.readout.in_features == (128 if full else 96))
            check(f"{name}:byte_count", (ROOT / manifest["checkpoint_path"]).stat().st_size == row["checkpoint_bytes"])
            with np.load(folder / "predictions.npz") as saved:
                original_ids = source_data[row["seed"]]["indices"]
                for split, ids in (("check_calibration", original_ids["calibration"][1::2]), ("validation", original_ids["validation"])):
                    check(f"{name}:{split}:ids_labels", np.array_equal(saved[f"{split}_indices"], ids) and np.array_equal(saved[f"{split}_labels"], labels[ids]))
                    metrics, logits = evaluate(model, torch.as_tensor(images[ids], device=device), torch.as_tensor(labels[ids], device=device), mean, std, config["batch_size"])
                    check(f"{name}:{split}:reload_logits", np.array_equal(logits, saved[f"{split}_logits"]))
                    check(f"{name}:{split}:metrics", all(abs(metrics[key]-row["evaluation"][split][key]) <= 1e-10 for key in metrics))
            del model
        aggregates = read(output / "aggregated.json")
        for entry in aggregates:
            values = []
            for seed in config["seeds"]:
                cases = [row for row in summary["rows"] if row["seed"] == seed and row["policy"] == entry["policy"]]
                values.append(float(np.mean([row["evaluation"]["validation"]["accuracy"] for row in cases])))
            check(f"aggregate:{entry['policy']}", abs(np.mean(values)-entry["mean_accuracy"]) < 1e-12 and abs(np.std(values, ddof=1)-entry["sample_sd_accuracy"]) < 1e-12)
    else:
        check("honest_skipped_stage", summary["status"] == "not_run_gate_failed" and not diagnostic["gate"]["supports_physical_comparison"] and not summary["physical_cases"])
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "check_count": len(checks),
              "verifier_sha256": file_sha256(Path(__file__)), "official_test_evaluated": False}
    write_json(output / "verification.json", result)
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed": [key for key, value in checks.items() if not value]}, indent=2), flush=True)
    if result["status"] != "passed":
        raise AssertionError("Merging verification failed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/quaternion_merging")
    args = parser.parse_args()
    verify(ROOT / args.output)
