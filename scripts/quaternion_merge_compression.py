"""Physically evaluate the frozen held-out merging gate, with matched controls."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import traceback

import numpy as np
import torch

from scripts.cifar_cnn import make_model, write_json
from src.cnn_training import evaluate, prepare_batch, seed_everything
from src.compression.quaternion_channels import compact_block, component_indices
from src.compression.quaternion_merging import logits_from_units, merged_last_block, predicted_logits
from src.data.cifar_cached import file_sha256, load_cached_cifar

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_case(manifest, device):
    model = compact_block(make_model(manifest["base_config"], "quaternion"), 2, manifest["kept_units"]).to(device).eval()
    model.load_state_dict(torch.load(ROOT / manifest["checkpoint_path"], map_location=device, weights_only=True)["model"])
    return model


def expected_logits(pooled, weight, bias, manifest):
    if manifest["kind"] == "baseline":
        return logits_from_units(pooled, weight, bias)
    if manifest["kind"] == "random":
        modified = np.asarray(pooled).copy()
        modified[:, manifest["removed_units"]] = 0
        return logits_from_units(modified, weight, bias)
    return predicted_logits(pooled, weight, bias, manifest["pairs"], manifest["compensation"])


def render(output, summary):
    policies = ["baseline", "random", "quaternion_right_pruned", "quaternion_right_mean_only", "quaternion_right_affine",
                "full_real_pruned", "full_real_mean_only", "full_real_affine"]
    aggregate = []
    for policy in policies:
        seed_rows = []
        for seed in summary["protocol"]["config"]["seeds"]:
            rows = [row for row in summary["rows"] if row["seed"] == seed and row["policy"] == policy]
            seed_rows.append({"seed": seed, "accuracy": float(np.mean([row["evaluation"]["validation"]["accuracy"] for row in rows])),
                              "cross_entropy": float(np.mean([row["evaluation"]["validation"]["cross_entropy"] for row in rows]))})
        values = [row["accuracy"] for row in seed_rows]
        aggregate.append({"policy": policy, "seed_values": seed_rows, "mean_accuracy": float(np.mean(values)),
                          "sample_sd_accuracy": float(np.std(values, ddof=1)), "mean_cross_entropy": float(np.mean([row["cross_entropy"] for row in seed_rows])),
                          "real_parameters": 73898 if policy == "baseline" else 64298})
    write_json(output / "aggregated.json", aggregate)
    baseline, random = (next(row for row in aggregate if row["policy"] == name) for name in ("baseline", "random"))
    effects = {}
    for mode in summary["protocol"]["config"]["compression_maps"]:
        merged, pruned, mean = [next(row for row in aggregate if row["policy"] == f"{mode}_{kind}") for kind in ("affine", "pruned", "mean_only")]
        effects[mode] = {}
        for name, reference in (("random", random), ("same_victims_pruned", pruned), ("same_victims_mean_only", mean), ("unpruned", baseline)):
            differences = [100*(a["accuracy"]-b["accuracy"]) for a, b in zip(merged["seed_values"], reference["seed_values"])]
            effects[mode][name] = {"paired_accuracy_differences_pp": differences, "mean_difference_pp": float(np.mean(differences)),
                                   "sample_sd_difference_pp": float(np.std(differences, ddof=1))}
    write_json(output / "paired_effects.json", effects)
    rows = ["# Physical last-block merging with readout compensation", "", "## Hypotheses", "",
            "Held-out reconstructible units may be removed while folding their contribution into retained readout "
            "weights/bias. Quaternion-restricted compensation may or may not retain quality as well as an unrestricted "
            "real 4x4 fit. No novel method or special quaternion advantage is assumed.", "",
            "## Protocol", "",
            "The diagnostic gate passed for both map families in all three existing baseline seeds. Fit/choose pairs "
            "using only 500 even-position calibration images; gate on the 500 disjoint odd-position images. "
            "Freeze every pair set and control manifest before new validation inference. Physically remove eight "
            "of 32 final-block quaternion units (73,898 to 64,298 real parameters, 12.99% saving). "
            "Evaluate affine compensation, pruning the exact same victims, and mean-only correction for each family. "
            "Reuse ten existing random-pruning selections/checkpoints per seed and the standardized baseline checkpoints. "
            "No CNN optimizer updates, BN refit, validation-based policy/size tuning or official-test inference. "
            "All 1,000 validation images are disjoint from fitting and the diagnostic gate, but served earlier checkpoint "
            "selection, so this is not a new blind benchmark.", "",
            "## Measured validation results", "",
            "| Policy | Real parameters | Mean accuracy ± seed SD | Mean CE |", "| --- | ---: | ---: | ---: |"]
    for row in aggregate:
        rows.append(f"| {row['policy']} | {row['real_parameters']} | {100*row['mean_accuracy']:.2f}% ± {100*row['sample_sd_accuracy']:.2f} pp | {row['mean_cross_entropy']:.6f} |")
    rows += ["", "Random repetitions are averaged within each training seed before computing the sample SD across "
             "three seeds. Parameters count actual compact tensors; no masks or extra inference mapping layers remain.", "",
             "| Map | Comparison | Paired seed accuracy differences (pp) | Mean difference (pp) |", "| --- | --- | --- | ---: |"]
    for mode, comparisons in effects.items():
        for name, value in comparisons.items():
            rows.append(f"| {mode} | {name} | {', '.join(f'{x:+.2f}' for x in value['paired_accuracy_differences_pp'])} | {value['mean_difference_pp']:+.2f} |")
    rows += ["", "## Interpretation and limits", "",
             "Accuracy effects against the same-victim pruning and mean-only controls separate compensation from "
             "selection and trivial average-offset correction. Between-family comparisons still combine each family's "
             "own pair choices and fitted map; the diagnostic also records same-pair comparisons. "
             "Diagonal fitting is a coefficient-matched diagnostic, not another physical compression policy. "
             "Affine fit coefficients (8 vs 20 per pair, including offsets) are calibration estimates, not new "
             "retained neural weights: both final compact methods have exactly the same parameter budget. "
             "A better reconstruction ratio alone is not proof of a better classifier.", "",
             "This is one architecture, dataset, selected block and removal level, with three existing training seeds "
             "and no fine-tuning. Report negative results and worse CE alongside accuracy. No general compression "
             "superiority, latency/FLOP gain, novelty or publication readiness follows automatically. Prior art keys: "
             "Kim2020, Ebid2025, Vecchi2020, Singh2026 in references.bib. No faithful full-method reproduction is claimed.", "",
             "## Artifacts", "",
             "DIAGNOSTIC_REPORT.md, diagnostic_protocol.json, all-pair fits, calibration partitions, gate_frozen.json, "
             "physical_protocol.json, selection_frozen.json, per-case manifests/checkpoints/predictions/metrics, "
             "aggregated.json, paired_effects.json and comparison.png.", ""]
    (output / "REPORT.md").write_text("\n".join(rows), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.5))
    labels = ["Unpruned", "Random", "Q: prune", "Q: mean", "Q: merge", "Full: prune", "Full: mean", "Full: merge"]
    ax.bar(labels, [100*row["mean_accuracy"] for row in aggregate], yerr=[100*row["sample_sd_accuracy"] for row in aggregate],
           color=["#777777", "#999999", "#a5bed6", "#7f9ebd", "#326ea8", "#bad9c6", "#83b798", "#429266"], capsize=4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Validation accuracy (%), mean ± training-seed SD")
    ax.set_title("Last-block merging: 64,298 parameters in every compact condition; 3 seeds; no fine-tuning")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)
    return aggregate, effects


def run(config):
    output = ROOT / config["output"]
    diagnostic = read(output / "diagnostic_summary.json")
    gate = read(output / "gate_frozen.json")
    if file_sha256(output / "diagnostic_summary.json") != gate["diagnostic_summary_sha256"] or not diagnostic["gate"]["supports_physical_comparison"]:
        raise ValueError("The frozen held-out diagnostic gate must pass before physical evaluation")
    if diagnostic["protocol"]["config"] != config:
        raise ValueError("Diagnostic and physical configurations differ")
    for name, digest in {**diagnostic["protocol"]["source_hashes"], **diagnostic["protocol"]["protected_inputs"]}.items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f"A frozen source or input changed: {name}")
    multiseed = read(ROOT / config["source_multiseed"] / "summary.json")
    additional = {}
    for run_record in multiseed["runs"]:
        path = ROOT / config["source_random_compression"] / f"seed_{run_record['seed']}" / "baseline_r00_removed00" / "model.pt"
        additional[str(path.relative_to(ROOT)).replace("\\", "/")] = file_sha256(path)
    protocol = {"config": config, "diagnostic_protocol_sha256": file_sha256(output / "diagnostic_protocol.json"),
                "gate_frozen_sha256": file_sha256(output / "gate_frozen.json"), "physical_script_sha256": file_sha256(Path(__file__)),
                "additional_protected_inputs": additional, "optimizer_updates": 0, "bn_recalibration": False, "official_test_evaluated": False}
    if (output / "physical_protocol.json").exists() and read(output / "physical_protocol.json") != protocol:
        raise ValueError("Physical runner/protocol changed; use a separate output folder")
    write_json(output / "physical_protocol.json", protocol)
    if (output / "summary.json").exists():
        summary = read(output / "summary.json")
        if summary["status"] == "completed":
            render(output, summary)
            return summary
    torch.set_num_threads(config["cpu_threads"])
    source0 = ROOT / multiseed["runs"][0]["path"]
    model_config = read(source0 / "summary.json")["config"]
    (images, labels), unused_test = load_cached_cifar(ROOT / model_config["archive"])
    del unused_test
    device = torch.device(config["device"])
    statistics = read(source0 / "data.json")["statistics"]
    mean = torch.tensor(statistics["mean"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(statistics["std"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    manifests = []
    # Prepare ALL models and manifests before any new validation forward.
    for run_record, diag in zip(multiseed["runs"], diagnostic["records"]):
        seed, source = run_record["seed"], ROOT / run_record["path"]
        seed_everything(seed)
        current_config = read(source / "summary.json")["config"]
        model = make_model(current_config, "quaternion").to(device).eval()
        model.load_state_dict(torch.load(source / "quaternion" / "model.pt", map_location=device, weights_only=True)["model"])
        original = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        with np.load(source / "split_indices.npz") as split:
            cal_ids = split["calibration"]
        with np.load(source / "quaternion" / "calibration_activations.npz") as cached:
            pooled = cached["trained"].astype(np.float64).mean(axis=1)
        probe_positions = np.arange(0, 64, 2)
        probe = prepare_batch(torch.as_tensor(images[cal_ids[probe_positions]], device=device), mean, std)
        weight, bias = model.readout.weight.detach().cpu().double().numpy(), model.readout.bias.detach().cpu().double().numpy()
        specs = [("baseline", "baseline", [], "none", None)]
        for mode in config["compression_maps"]:
            specs.extend((f"{mode}_{kind}", "merge", diag["selected"][mode]["pairs"], kind, None) for kind in ("none", "mean_only", "affine"))
        for rep in range(config["random_repetitions"]):
            specs.append((f"random_r{rep:02d}", "random", [], "none", rep))
        for case_name, kind, pairs, compensation, rep in specs:
            if kind == "baseline":
                old_folder = ROOT / config["source_random_compression"] / f"seed_{seed}" / "baseline_r00_removed00"
                kept, victims, compact = list(range(32)), [], deepcopy(model)
                checkpoint = old_folder / "model.pt"
                policy = "baseline"
            elif kind == "random":
                old_folder = ROOT / config["source_random_compression"] / f"seed_{seed}" / f"random_r{rep:02d}_removed08"
                old_manifest = read(old_folder / "manifest.json")
                kept, victims = old_manifest["kept_units"], old_manifest["removed_units"]
                compact = compact_block(model, 2, kept).eval()
                checkpoint, policy = old_folder / "model.pt", "random"
            else:
                compact, kept = merged_last_block(model, pairs, compensation)
                victims = sorted(row["victim"] for row in pairs)
                checkpoint = output / f"seed_{seed}" / case_name / "model.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": {name: value.detach().cpu() for name, value in compact.state_dict().items()}}, checkpoint)
                policy = case_name.replace("_none", "_pruned")
            # Existing standardized baseline/random checkpoints must match surgery.
            if kind in ("baseline", "random"):
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
                if not all(torch.equal(value.cpu(), saved[name]) for name, value in compact.state_dict().items()):
                    raise AssertionError("Reused baseline/random state differs from exact source compaction")
            manifest = {"seed": seed, "case": case_name, "policy": policy, "kind": kind, "pairs": pairs,
                        "compensation": compensation, "kept_units": kept, "removed_units": victims, "base_config": current_config,
                        "checkpoint_path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"), "checkpoint_sha256": file_sha256(checkpoint),
                        "source_checkpoint_sha256": file_sha256(source / "quaternion" / "model.pt"),
                        "random_repetition": rep, "checkpoint_reused": kind in ("baseline", "random"), "optimizer_updates": 0}
            with torch.no_grad():
                actual = compact(probe).cpu().numpy()
            predicted = expected_logits(pooled[probe_positions], weight, bias, manifest)
            error = float(np.max(np.abs(actual-predicted)))
            if error > config["equivalence_tolerance"]:
                raise AssertionError(f"Cached/physical logit formula failed at {seed}/{case_name}: {error}")
            manifest["cached_probe_max_abs_error"] = error
            case_folder = output / f"seed_{seed}" / case_name
            write_json(case_folder / "manifest.json", manifest)
            manifests.append(manifest)
            del compact
        if not all(torch.equal(value.cpu(), original[name]) for name, value in model.state_dict().items()):
            raise AssertionError("The source model changed during merging")
        del model
        print(f"Seed {seed}: 17 physical/reused cases frozen from calibration only", flush=True)
    frozen = {"manifests": manifests, "all_choices_frozen_before_validation": True, "official_test_evaluated": False}
    write_json(output / "selection_frozen.json", frozen)
    rows = []
    for manifest in manifests:
        seed, case_name = manifest["seed"], manifest["case"]
        case_folder = output / f"seed_{seed}" / case_name
        if (case_folder / "metrics.json").exists():
            result = read(case_folder / "metrics.json")
            if result["manifest_sha256"] != file_sha256(case_folder / "manifest.json"):
                raise ValueError("A scored case manifest changed")
            rows.append(result)
            continue
        run_record = next(row for row in multiseed["runs"] if row["seed"] == seed)
        source = ROOT / run_record["path"]
        with np.load(source / "split_indices.npz") as split:
            split_ids = {"check_calibration": split["calibration"][1::2], "validation": split["validation"]}
        model = load_case(manifest, device)
        metrics, predictions = {}, {}
        for name, ids in split_ids.items():
            metrics[name], predictions[f"{name}_logits"] = evaluate(model, torch.as_tensor(images[ids], device=device), torch.as_tensor(labels[ids], device=device), mean, std, config["batch_size"])
        with np.load(source / "quaternion" / "calibration_activations.npz") as cached:
            pooled = cached["trained"][1::2].astype(np.float64).mean(axis=1)
        source_state = torch.load(source / "quaternion" / "model.pt", map_location="cpu", weights_only=True)["model"]
        predicted = expected_logits(pooled, source_state["readout.weight"].double().numpy(), source_state["readout.bias"].double().numpy(), manifest)
        error = float(np.max(np.abs(predicted-predictions["check_calibration_logits"])))
        if error > config["equivalence_tolerance"]:
            raise AssertionError(f"Held-out cached/physical equivalence failed: {seed}/{case_name}: {error}")
        np.savez_compressed(case_folder / "predictions.npz", **predictions,
                            **{f"{name}_labels": labels[ids] for name, ids in split_ids.items()},
                            **{f"{name}_indices": ids for name, ids in split_ids.items()})
        result = {"seed": seed, "case": case_name, "policy": manifest["policy"], "evaluation": metrics,
                  "manifest_sha256": file_sha256(case_folder / "manifest.json"), "predictions_sha256": file_sha256(case_folder / "predictions.npz"),
                  "checkpoint_sha256": file_sha256(ROOT / manifest["checkpoint_path"]), "checkpoint_bytes": (ROOT / manifest["checkpoint_path"]).stat().st_size,
                  "real_parameters": sum(p.numel() for p in model.parameters()), "buffer_elements": sum(b.numel() for b in model.buffers()),
                  "removed_quaternion_units": len(manifest["removed_units"]), "check_cached_max_abs_error": error,
                  "optimizer_updates": 0, "bn_recalibration": False, "official_test_evaluated": False}
        write_json(case_folder / "metrics.json", result)
        rows.append(result)
        del model
    summary = {"status": "completed", "protocol": protocol, "rows": rows,
               "selection_frozen_sha256": file_sha256(output / "selection_frozen.json"),
               "optimizer_updates": 0, "bn_recalibration": False, "official_test_evaluated": False}
    write_json(output / "summary.json", summary)
    aggregate, effects = render(output, summary)
    print(json.dumps({"aggregated_accuracy": {row["policy"]: row["mean_accuracy"] for row in aggregate}, "paired_effects": effects}, indent=2), flush=True)
    return summary


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
