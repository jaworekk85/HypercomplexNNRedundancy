"""Fit and independently check affine reconstruction of cached GAP units."""

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import torch

from scripts.cifar_cnn import write_json
from src.compression.quaternion_merging import all_pair_fits, aggregate_errors, choose_disjoint_pairs
from src.data.cifar_cached import file_sha256

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
SOURCES = ["scripts/quaternion_merge_diagnostic.py", "src/compression/quaternion_merging.py"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def signal_passes(errors, config):
    a, b = errors["ratio_vs_mean_only"], errors["ratio_vs_pruned"]
    return a is not None and b is not None and a <= config["gate_maximum_check_mse_ratio_vs_mean_only"]+1e-12 and b <= config["gate_maximum_check_mse_ratio_vs_pruning"]+1e-12


def render(output, result):
    config = result["protocol"]["config"]
    rows = ["# Affine quaternion-unit reconstruction and merging diagnostic", "", "## Hypotheses", "",
            "A unit with weak fixed-coordinate similarity might still be reconstructible from another unit. "
            "A held-out affine relation may preserve classifier logits when the victim is removed and its contribution "
            "is folded into the keeper/readout bias. Whether quaternion constraints help is unconfirmed.", "",
            "## Protocol", "",
            "Three existing seeds, final block only. Average the 16 spatial positions to obtain one four-component "
            "vector per unit/image. Fit on even calibration positions (500 images); evaluate on odd positions (500). "
            "These are relations of GAP features, not an asserted spatial or ReLU equivariance. "
            "Fit right-Hamilton affine (4+4), diagonal affine (4+4, matched coefficient count), and full-real affine "
            "(16+4) maps in float64. All 992 directed pairs per family/seed are retained. "
            "Choose eight disjoint pairs using fit-only class-centered single-pair logit residual MSE. "
            "Check the combined eight-pair residual independently, including pair interactions. "
            "Compare to zeroing the same victims and to folding only their FIT-half means into the readout bias. "
            "No labels, new CNN training, validation inference or official-test inference are used in this diagnostic.", "",
            "The gate requires aggregate held-out logit-MSE ratio <=0.90 vs mean-only and <=0.75 vs pruning "
            "in at least two of three seeds. If either map passes, physically evaluate both maps and all controls. "
            "This is a gate for a small compression test, not evidence of accuracy gains.", "",
            "## Measured diagnostic results", "",
            "| Seed | Selection/map | Check MSE vs mean-only | Check MSE vs pruning | Signal gate |",
            "| ---: | --- | ---: | ---: | --- |"]
    for record in result["records"]:
        for mode in config["compression_maps"]:
            item = record["selected"][mode]
            errors = item["check_aggregate_errors"]
            a, b = errors["ratio_vs_mean_only"], errors["ratio_vs_pruned"]
            rows.append(f"| {record['seed']} | {mode} | {a:.4f} | {b:.4f} | {item['signal_passes']} |" if a is not None and b is not None else f"| {record['seed']} | {mode} | {a} | {b} | {item['signal_passes']} |")
    rows += ["", "| Seed | Map | Median check feature NMSE vs fit-half mean, all directed pairs |",
             "| ---: | --- | ---: |"]
    for record in result["records"]:
        for mode, metrics in record["all_pair_summaries"].items():
            rows.append(f"| {record['seed']} | {mode} | {metrics['median_check_feature_nmse']:.4f} |")
    rows += ["", f"Physical compression supported by the prespecified diagnostic gate: **{result['gate']['supports_physical_comparison']}**.", "",
             "## Interpretation and limits", "",
             "Each method selects its own pairs; between-policy effects include selection and map family. "
             "The common-full-selected-pair diagnostic additionally compares all three map families on the same "
             "pairs, but that selection favors the full fit. The diagonal diagnostic has the same number of fitted "
             "coefficients as the quaternion fit. No unique quaternion benefit follows from reconstruction alone. "
             "The gate uses held-out class-centered logit error, not labels or accuracy; a positive gate still needs "
             "physical evaluation on disjoint validation images. A simple quaternion least-squares fit is not "
             "a methodological novelty (PROJECT.md; prior context Kim2020, Ebid2025 in references.bib).", "",
             "Reconstruction coefficients are calibration estimates. After folding into the existing readout they "
             "are not additional inference parameters. If physical evaluation proceeds, removing 8/32 final-block "
             "units retains 64,298 of 73,898 real parameters (about 13% saving), with no optimizer updates or BN refit.", ""]
    (output / "DIAGNOSTIC_REPORT.md").write_text("\n".join(rows), encoding="utf-8")


def run(config):
    output = ROOT / config["output"]
    multiseed = read(ROOT / config["source_multiseed"] / "summary.json")
    compression = read(ROOT / config["source_random_compression"] / "summary.json")
    if multiseed["status"] != "completed" or [row["seed"] for row in multiseed["runs"]] != config["seeds"]:
        raise ValueError("Require three completed baseline seeds")
    if config["optimizer_updates"] or config["bn_recalibration"] or config["official_test_evaluation"] or config["block"] != 3 or config["removed_units"] != 8:
        raise ValueError("This implements only the frozen last-block, eight-unit, no-training/no-test study")
    old_hashes = compression["protocol"]["source_hashes"]
    for name, digest in old_hashes.items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f"Frozen numerical source changed: {name}")
    paths = [ROOT / config["source_multiseed"] / "summary.json", ROOT / config["source_random_compression"] / "summary.json"]
    for record in multiseed["runs"]:
        source = ROOT / record["path"]
        paths.extend(source / name for name in ("summary.json", "data.json", "split_indices.npz", "quaternion/model.pt", "quaternion/metrics.json", "quaternion/calibration_activations.npz"))
        audit = next(row for row in record["models"] if row["kind"] == "quaternion")["first_update"]["checks"]
        if not all(audit.values()):
            raise ValueError("Original quaternion gradient/update gate failed")
        for repetition in range(config["random_repetitions"]):
            folder = ROOT / config["source_random_compression"] / f"seed_{record['seed']}" / f"random_r{repetition:02d}_removed08"
            paths.extend(folder / name for name in ("manifest.json", "model.pt", "validation.json"))
    protected = {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in paths}
    hashes = {**old_hashes, **{name: file_sha256(ROOT / name) for name in SOURCES}}
    protocol = {"config": config, "source_hashes": hashes, "protected_inputs": protected,
                "environment": {"python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__), "platform": platform.platform()},
                "fit_labels_used": False, "new_cnn_training_updates": 0, "official_test_evaluated": False}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "diagnostic_protocol.json").exists() and read(output / "diagnostic_protocol.json") != protocol:
        raise ValueError("Diagnostic protocol changed; preserve this run and use a different output")
    write_json(output / "diagnostic_protocol.json", protocol)
    if (output / "diagnostic_summary.json").exists():
        result = read(output / "diagnostic_summary.json")
        render(output, result)
        return result
    torch.set_num_threads(config["cpu_threads"])
    records = []
    for record in multiseed["runs"]:
        source, seed = ROOT / record["path"], record["seed"]
        folder = output / f"seed_{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = torch.load(source / "quaternion" / "model.pt", map_location="cpu", weights_only=True)
        weight, bias = [checkpoint["model"][f"readout.{key}"].double().numpy() for key in ("weight", "bias")]
        with np.load(source / "quaternion" / "calibration_activations.npz") as saved:
            spatial = saved["trained"]
        if spatial.shape != (1000, 16, 32, 4) or not np.isfinite(spatial).all():
            raise ValueError("Unexpected cached activation shape or nonfinite values")
        pooled = spatial.astype(np.float64).mean(axis=1)
        fit_positions, check_positions = np.arange(0, len(pooled), 2), np.arange(1, len(pooled), 2)
        with np.load(source / "split_indices.npz") as split:
            calibration_ids, validation_ids, train_ids = split["calibration"], split["validation"], split["train"]
        if len(fit_positions) != config["fit_images"] or len(check_positions) != config["check_images"] or np.intersect1d(calibration_ids, validation_ids).size or np.intersect1d(calibration_ids, train_ids).size:
            raise ValueError("Calibration partition overlap or wrong image count")
        np.savez_compressed(folder / "calibration_partition.npz", fit_positions=fit_positions, check_positions=check_positions,
                            fit_original_indices=calibration_ids[fit_positions], check_original_indices=calibration_ids[check_positions])
        all_fits, selected, distributions = {}, {}, {}
        for mode in config["diagnostic_maps"]:
            candidates = all_pair_fits(pooled[fit_positions], pooled[check_positions], weight, mode)
            write_json(folder / f"{mode}_all_pairs.json", candidates)
            all_fits[mode] = candidates
            valid = [row["check_feature_nmse_vs_fit_mean"] for row in candidates if row["check_feature_nmse_vs_fit_mean"] is not None]
            distributions[mode] = {"directed_pair_count": len(candidates), "valid_check_nmse_count": len(valid),
                                   "median_check_feature_nmse": float(np.median(valid)), "mean_check_feature_nmse": float(np.mean(valid)),
                                   "fitted_coefficients_per_pair": 20 if mode == "full_real" else 8}
            if mode in config["compression_maps"]:
                pairs = choose_disjoint_pairs(candidates, config["removed_units"])
                fit_errors = aggregate_errors(pooled[fit_positions], weight, bias, pairs)
                check_errors = aggregate_errors(pooled[check_positions], weight, bias, pairs)
                selected[mode] = {"pairs": pairs, "fit_aggregate_errors": fit_errors, "check_aggregate_errors": check_errors,
                                  "signal_passes": signal_passes(check_errors, config)}
                print(f"Seed {seed}, {mode}: held-out MSE ratio vs mean-only={check_errors['ratio_vs_mean_only']:.4f}, vs pruning={check_errors['ratio_vs_pruned']:.4f}; signal={selected[mode]['signal_passes']}", flush=True)
        common_indices = [(row["keeper"], row["victim"]) for row in selected["full_real"]["pairs"]]
        common = {}
        for mode, candidates in all_fits.items():
            lookup = {(row["keeper"], row["victim"]): row for row in candidates}
            pairs = [lookup[key] for key in common_indices]
            common[mode] = aggregate_errors(pooled[check_positions], weight, bias, pairs)
        result = {"seed": seed, "source": str(source.relative_to(ROOT)).replace("\\", "/"),
                  "selected": selected, "all_pair_summaries": distributions,
                  "same_full_selected_pairs_check_errors": common, "coefficients_fitted_without_labels": True,
                  "partition_sha256": file_sha256(folder / "calibration_partition.npz")}
        write_json(folder / "diagnostic.json", result)
        records.append(result)
    passes = {mode: sum(row["selected"][mode]["signal_passes"] for row in records) for mode in config["compression_maps"]}
    gate = {"passing_seeds": passes, "minimum_passing_seeds": config["gate_minimum_passing_seeds"],
            "supports_physical_comparison": any(count >= config["gate_minimum_passing_seeds"] for count in passes.values())}
    result = {"status": "completed", "protocol": protocol, "records": records, "gate": gate,
              "official_test_evaluated": False, "new_cnn_training_updates": 0}
    write_json(output / "diagnostic_summary.json", result)
    write_json(output / "gate_frozen.json", {"gate": gate, "diagnostic_summary_sha256": file_sha256(output / "diagnostic_summary.json"),
               "frozen_before_any_new_validation_inference": True, "new_validation_inference": False, "official_test_evaluated": False})
    render(output, result)
    if not gate["supports_physical_comparison"]:
        write_json(output / "summary.json", {"status": "not_run_gate_failed", "diagnostic": result, "physical_cases": [], "official_test_evaluated": False})
        (output / "REPORT.md").write_text((output / "DIAGNOSTIC_REPORT.md").read_text(encoding="utf-8") + "\n## Decision\n\nThe held-out diagnostic gate failed. No physical models, validation inference, optimizer updates or official-test inference were created by this stage.\n", encoding="utf-8")
    if any(file_sha256(ROOT / name) != digest for name, digest in protected.items()):
        raise AssertionError("An original input artifact changed")
    print(json.dumps(gate, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/quaternion_merging.json")
    args = parser.parse_args()
    config = read(ROOT / args.config)
    try:
        run(config)
    except Exception:
        output = ROOT / config["output"]
        output.mkdir(parents=True, exist_ok=True)
        (output / "FAILED_CHECKS.md").write_text("# Failed execution/check\n\n```text\n"+traceback.format_exc()+"\n```\nNo gain is claimed from this failure.\n", encoding="utf-8")
        raise
