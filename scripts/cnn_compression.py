"""All-block seed confirmation and a frozen three-policy physical compression pilot."""

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch

from scripts.analyze_cnn_blocks import analyze_model
from scripts.cifar_cnn import make_model, source_hashes, write_json
from src.cnn_training import evaluate, prepare_batch, seed_everything
from src.compression.quaternion_channels import compact_block, removal_order
from src.data.cifar_cached import file_sha256, load_cached_cifar
from src.evaluation.block_ablation import capture_blocks, incoming_norms, predict_from_block
from src.evaluation.thresholds import at_most, correct_ablation_labels

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["scripts/cnn_compression.py", "scripts/analyze_cnn_blocks.py", "src/evaluation/block_ablation.py", "src/compression/quaternion_channels.py", "src/evaluation/thresholds.py"]


def confirm_blocks(runs, folder, config, images, labels, indices, statistics):
    analysis_config = json.loads((ROOT / config["source_block_analysis"] / "config.json").read_text(encoding="utf-8"))
    original = json.loads((ROOT / config["source_block_analysis"] / "summary.json").read_text(encoding="utf-8"))
    records = []
    for run in runs:
        seed = run["seed"]
        if seed == 2026:
            records.extend(original["models"])
            continue
        source = ROOT / run["path"]
        training = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        for kind in ("quaternion", "real"):
            target = folder / "block_analysis" / f"seed_{seed}" / kind
            if (target / "summary.json").exists():
                record = json.loads((target / "summary.json").read_text(encoding="utf-8"))
                if record["checkpoint_sha256"] != file_sha256(source / kind / "model.pt"):
                    raise ValueError("Analysis checkpoint changed")
            else:
                record = analyze_model(kind, analysis_config, training["config"], source, images, labels, indices, statistics, target)
            records.append(record)
    for record in records:
        correct_ablation_labels(record, analysis_config)
        corrected_folder = folder / "block_analysis" / f"seed_{record['seed']}" / record["kind"]
        write_json(corrected_folder / "summary.json", record)
        for block in record["blocks"]:
            block_folder = corrected_folder / f"block_{block['block']}"
            write_json(block_folder / "summary.json", block)
            with (block_folder / "ablations.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(block["ablations"][0]))
                writer.writeheader()
                writer.writerows(block["ablations"])
    selected = [next(block for block in record["blocks"] if block["block"] == config["block"])
                for record in records if record["kind"] == "quaternion"]
    gate = all(block["candidate_for_controlled_removal"] for block in selected)
    result = {"records": records, "selected_block": config["block"], "all_three_seeds_pass_selected_block_gate": gate}
    write_json(folder / "block_analysis" / "summary.json", result)
    lines = ["# All-block diagnostics across three seeds", "", "## Hypothesis", "",
             "Similarity may predict low-impact whole-unit removal. All negative correlations are retained.", "",
             "## Measured results", "", "| Seed | Model | Block | Low-impact units | Similarity vs negative loss impact (Spearman) | Image-half stability |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for record in records:
        for block in record["blocks"]:
            stability_key = "grouped_stability_across_disjoint_image_halves" if record["kind"] == "quaternion" else "scalar_stability_across_disjoint_image_halves"
            lines.append(f"| {record['seed']} | {record['kind']} | {block['block']} | {len(block['individual_low_impact_units'])}/{block['units']} | {block['similarity_vs_negative_loss_impact_spearman']} | {block['trained'][stability_key]['pair_value_pearson']:.4f} |")
    lines.extend(["", "## Interpretation", "", f"All three seeds pass the fixed block-{config['block']} diagnostic gate: {gate}.",
                  "The gate uses individual ablations and within-model image-half stability. It does not prove cumulative safe removal or neuron identity across seeds. Neurons are not aligned between independent trainings. Quaternion and scalar real ablations have different granularity. No novel score or established method superiority is claimed."])
    (folder / "block_analysis" / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def load_compact(case_dir, model_config, device):
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    base = make_model(model_config, "quaternion")
    model = compact_block(base, manifest["block"] - 1, manifest["kept_units"]).to(device).eval()
    model.load_state_dict(torch.load(case_dir / "model.pt", map_location=device, weights_only=True)["model"], strict=True)
    return model


def prepare_cases(run, config, folder, images, labels, indices, statistics, diagnostics):
    seed, source = run["seed"], ROOT / run["path"]
    training = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    model_config = training["config"]
    seed_everything(seed)
    device = torch.device(config["device"])
    model = make_model(model_config, "quaternion").to(device).eval()
    checkpoint_path = source / "quaternion" / "model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)["model"])
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    index = config["block"] - 1
    units = model.blocks[index][3].out_channels
    baseline_params = sum(p.numel() for p in model.parameters())
    val_x = torch.as_tensor(images[indices["validation"]], device=device)
    val_y = torch.as_tensor(labels[indices["validation"]], device=device)
    mean = torch.tensor(statistics["mean"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(statistics["std"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    normalized = prepare_batch(val_x[:32], mean, std)
    features = capture_blocks(model, normalized, 32)[index]
    with torch.no_grad():
        reference = model(normalized).cpu().numpy()
    # All rankings are derived from the frozen calibration matrix and incoming kernels.
    if seed == 2026:
        matrix_path = ROOT / config["source_block_analysis"] / "quaternion" / f"block_{config['block']}" / "calibration_similarity.npz"
    else:
        matrix_path = folder / "block_analysis" / f"seed_{seed}" / "quaternion" / f"block_{config['block']}" / "calibration_similarity.npz"
    with np.load(matrix_path) as saved:
        similarity = saved["trained_grouped"]
    norms = incoming_norms(model, index)
    specs = [("baseline", 0, 0.0)]
    for policy in config["policies"]:
        for repetition in range(config["random_repetitions"] if policy == "random" else 1):
            specs.extend((policy, repetition, fraction) for fraction in config["removed_fractions"] if fraction > 0)
    rows = []
    for policy, repetition, fraction in specs:
        random_seed = config["random_seed_offset"] + seed * 100 + repetition
        order = removal_order(policy, norms, similarity, random_seed) if policy != "baseline" else list(range(units))
        removed = order[:round(units * fraction)]
        kept = sorted(set(range(units)) - set(removed))
        case_id = f"{policy}_r{repetition:02d}_removed{len(removed):02d}"
        case_dir = folder / f"seed_{seed}" / case_id
        manifest = {"seed": seed, "policy": policy, "repetition": repetition, "random_seed": random_seed if policy == "random" else None,
                    "block": config["block"], "original_units": units, "removed_fraction": fraction, "removed_units": removed,
                    "kept_units": kept, "base_config": model_config, "source_checkpoint_sha256": file_sha256(checkpoint_path),
                    "calibration_matrix_sha256": file_sha256(matrix_path), "fine_tuning_updates": 0}
        if (case_dir / "validation.json").exists():
            if json.loads((case_dir / "manifest.json").read_text(encoding="utf-8")) != manifest:
                raise ValueError("Compression manifest changed")
            row = json.loads((case_dir / "validation.json").read_text(encoding="utf-8"))
            if row["checkpoint_sha256"] != file_sha256(case_dir / "model.pt"):
                raise ValueError("Compact checkpoint changed")
            rows.append(row)
            continue
        compact = compact_block(model, index, kept).eval()
        with torch.no_grad():
            actual = compact(normalized).cpu().numpy()
        expected = predict_from_block(model, index, features, removed, 32) if removed else reference
        error = float(np.max(np.abs(actual - expected)))
        if error > 1e-4 or not removed and error != 0:
            raise AssertionError(f"Compact/masked equivalence failed: {error}")
        write_json(case_dir / "manifest.json", manifest)
        torch.save({"model": {name: value.detach().cpu() for name, value in compact.state_dict().items()}}, case_dir / "model.pt")
        restored = load_compact(case_dir, model_config, device)
        with torch.no_grad():
            reload_error = float((restored(normalized) - compact(normalized)).abs().max())
        if reload_error != 0:
            raise AssertionError("Compact checkpoint reload failed")
        metrics, _ = evaluate(compact, val_x, val_y, mean, std, config["batch_size"])
        params = sum(p.numel() for p in compact.parameters() if p.requires_grad)
        baseline_val = next(r for r in training["models"] if r["kind"] == "quaternion")["selected_validation"]
        fraction_saved = (baseline_params - params) / baseline_params
        drop = baseline_val["accuracy"] - metrics["accuracy"]
        row = {"seed": seed, "case": case_id, "policy": policy, "repetition": repetition, "removed_fraction": fraction,
               "manifest_sha256": file_sha256(case_dir / "manifest.json"),
               "removed_quaternion_units": len(removed), "retained_quaternion_units": len(kept), "real_trainable_parameters": params,
               "real_parameter_fraction_saved": fraction_saved, "nontrainable_buffer_elements": sum(b.numel() for b in compact.buffers()),
               "checkpoint_bytes": (case_dir / "model.pt").stat().st_size, "checkpoint_sha256": file_sha256(case_dir / "model.pt"),
               "compact_vs_masked_max_abs_logit_error": error, "checkpoint_reload_max_abs_error": reload_error,
               "validation": metrics, "validation_accuracy_drop": drop,
               "validation_cross_entropy_increase": metrics["cross_entropy"] - baseline_val["cross_entropy"],
               "meets_prespecified_validation_usefulness_threshold": fraction_saved >= config["minimum_real_parameter_fraction_saved"] and at_most(drop, config["maximum_validation_accuracy_drop_for_useful_compression"])}
        write_json(case_dir / "validation.json", row)
        rows.append(row)
        del compact, restored
    if not all(torch.equal(value.cpu(), before[name]) for name, value in model.state_dict().items()):
        raise AssertionError("Source model mutated during compression")
    write_json(folder / f"seed_{seed}" / "validation_summary.json", {"rows": rows, "source_model_unchanged": True})
    print(f"Seed {seed}: {len(rows)} physical architectures frozen and validated", flush=True)
    del model, val_x, val_y, features
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def run(config, folder):
    torch.set_num_threads(4)
    multiseed = json.loads((ROOT / config["source_multiseed"] / "summary.json").read_text(encoding="utf-8"))
    if multiseed["status"] != "completed" or [run["seed"] for run in multiseed["runs"]] != config["seeds"]:
        raise ValueError("Three-seed training is incomplete")
    if config["fine_tuning_updates"] != 0:
        raise ValueError("Only the frozen no-fine-tuning protocol is implemented")
    hashes = {**source_hashes(), **{name: file_sha256(ROOT / name) for name in SOURCES}}
    protocol = {"config": config, "source_hashes": hashes, "multiseed_summary_sha256": file_sha256(ROOT / config["source_multiseed"] / "summary.json")}
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "protocol.json").exists() and json.loads((folder / "protocol.json").read_text(encoding="utf-8")) != protocol:
        raise ValueError("Compression protocol changed; use another output directory")
    write_json(folder / "protocol.json", protocol)
    source = ROOT / multiseed["runs"][0]["path"]
    original = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if original["source_hashes"] != source_hashes():
        raise ValueError("Frozen training sources changed")
    model_config = original["config"]
    (images, labels), (test_images, test_labels) = load_cached_cifar(ROOT / model_config["archive"])
    with np.load(source / "split_indices.npz") as split:
        indices = {key: split[key] for key in ("calibration", "validation")}
    statistics = json.loads((source / "data.json").read_text(encoding="utf-8"))["statistics"]
    diagnostics = confirm_blocks(multiseed["runs"], folder, config, images, labels, indices, statistics)
    if not diagnostics["all_three_seeds_pass_selected_block_gate"]:
        write_json(folder / "summary.json", {"status": "gate_not_confirmed", "diagnostics": diagnostics, "config": config, "test_inference": False})
        message = "# Compression diagnostic gate not confirmed\n\n## Measured result\n\nThe prespecified selected-block gate did not pass in all three training seeds. See block_analysis/REPORT.md for every seed, layer and negative predictor. No compact-model test evaluation or optimizer updates were performed.\n\n## Interpretation\n\nThis does not establish absence of redundancy generally. The fixed pilot did not justify proceeding under its own gate; no alternative layer or score was selected to force a positive result.\n"
        (folder / "REPORT.md").write_text(message, encoding="utf-8")
        print("Selected-block gate failed across seeds; compression experiment stopped", flush=True)
        return
    rows = []
    for run in multiseed["runs"]:
        rows.extend(prepare_cases(run, config, folder, images, labels, indices, statistics, diagnostics))
    write_json(folder / "validation_frozen.json", {"rows": rows, "all_manifests_frozen_before_test": True,
               "selection": "No policy or size discarded; evaluate every frozen case on test."})
    device = torch.device(config["device"])
    test_x, test_y = torch.as_tensor(test_images, device=device), torch.as_tensor(test_labels, device=device)
    mean = torch.tensor(statistics["mean"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(statistics["std"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    results = []
    for row in rows:
        case_dir = folder / f"seed_{row['seed']}" / row["case"]
        if (case_dir / "metrics.json").exists():
            result = json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))
        else:
            current = {**model_config, "seed": row["seed"]}
            model = load_compact(case_dir, current, device)
            metrics, logits = evaluate(model, test_x, test_y, mean, std, config["batch_size"])
            np.savez_compressed(case_dir / "test_predictions.npz", logits=logits, labels=test_labels)
            result = {**row, "test": metrics, "test_evaluated_after_all_validation_cases_frozen": True}
            write_json(case_dir / "metrics.json", result)
            del model
        results.append(result)
    write_json(folder / "summary.json", {"status": "completed", "config": config, "rows": results, "protocol": protocol,
               "validation_freeze_sha256": file_sha256(folder / "validation_frozen.json"), "fine_tuning_updates": 0})
    render_report(results, config, folder)
    print("Physical compression pilot completed", flush=True)


def render_report(rows, config, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = []
    for fraction in config["removed_fractions"]:
        for policy in (["baseline"] if fraction == 0 else config["policies"]):
            subset = [r for r in rows if r["removed_fraction"] == fraction and r["policy"] == policy]
            means = [np.mean([r["test"]["accuracy"] for r in subset if r["seed"] == seed]) for seed in config["seeds"]]
            groups.append({"policy": policy, "removed_fraction": fraction, "real_trainable_parameters": subset[0]["real_trainable_parameters"],
                           "checkpoint_bytes": subset[0]["checkpoint_bytes"], "mean_test_accuracy": float(np.mean(means)),
                           "sample_sd_across_training_seed_means": float(np.std(means, ddof=1)), "training_seed_means": [float(v) for v in means],
                           "mean_within_seed_random_sd": float(np.mean([np.std([r["test"]["accuracy"] for r in subset if r["seed"] == seed], ddof=1) for seed in config["seeds"]])) if policy == "random" else None,
                           "validation_usefulness_passes": sum(r["meets_prespecified_validation_usefulness_threshold"] for r in subset), "cases": len(subset)})
    write_json(folder / "aggregated.json", groups)
    with (folder / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
        flattened = [{**{k: v for k, v in row.items() if k not in ("validation", "test")}, **{f"{stage}_{metric}": value for stage in ("validation", "test") for metric, value in row[stage].items()}} for row in rows]
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    base = next(g for g in groups if g["policy"] == "baseline")
    for policy in config["policies"]:
        selected = [base] + [g for g in groups if g["policy"] == policy]
        y = [100*g["mean_test_accuracy"] for g in selected]
        error = [100*g["sample_sd_across_training_seed_means"] for g in selected]
        axes[0].errorbar([g["real_trainable_parameters"] for g in selected], y, yerr=error, marker="o", capsize=3, label=policy)
        axes[1].errorbar([100*g["removed_fraction"] for g in selected], y, yerr=error, marker="o", capsize=3, label=policy)
    axes[0].set(xlabel="Actual retained real trainable parameters", ylabel="Test accuracy (%)")
    axes[1].set(xlabel="Removed quaternion units in block 3 (%)", ylabel="Test accuracy (%)")
    for axis in axes:
        axis.legend(fontsize=8)
    fig.savefig(folder / "compression_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    lines = ["# Whole-quaternion physical compression pilot", "", "## Hypotheses", "",
             "A conventional activation-similarity policy may preserve quality better than raw incoming magnitude and random removal. No new algorithm, general superiority or success is assumed.", "",
             "## Fixed protocol", "", "Block 3 was selected on seed-2026 calibration/validation before new training outcomes. Three seeds; 0/25/50% whole-unit removal; no fine-tuning, BN re-calibration or weight compensation. Scores use calibration; all cases were frozen and validated before final test. The official test set was previously used in the pilots. See PROTOCOL.md and protocol.json for exact choices and hashes.",
             "Physical removal slices the second convolution output, matching BN channels and readout inputs. The block's first convolution remains unchanged. Unit-removal percentage therefore differs from total-model parameter savings. State files use the same serialization envelope for every case; manifests are separate. No measured latency or FLOP savings are claimed.", "",
             "## Measured results", "", "| Removal | Policy | Real trainable parameters | State bytes | Test accuracy mean +/- training-seed SD | Validation usefulness passes |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for g in groups:
        lines.append(f"| {100*g['removed_fraction']:.0f}% | {g['policy']} | {g['real_trainable_parameters']} | {g['checkpoint_bytes']} | {100*g['mean_test_accuracy']:.2f} +/- {100*g['sample_sd_across_training_seed_means']:.2f}% | {g['validation_usefulness_passes']}/{g['cases']} |")
    lines.extend(["", "Error bars: sample SD across three training-seed means; random first averages ten removal repetitions within each seed. Random repetitions are not 30 independent training seeds. All individual cases and within-seed random SD are retained in cases.csv / aggregated.json.",
                  "Validation usefulness was prespecified as <=2 pp accuracy drop with >=10% total real-parameter reduction. Every policy and size is reported regardless of this threshold. These practical thresholds are not evidence of novelty.", "", "## Interpretation", "",
                  "Single-unit dispensability need not survive simultaneous removal. Weak or negative similarity results and worse-than-random policies are valid outcomes. Raw kernel norms ignore BN scaling; this limits conclusions against stronger weight-based methods. The similarity policy has a norm tie/member rule and is not a faithful full reproduction of Ayinde2019. Weight-baseline context: Vecchi2020 and Singh2026 (references.bib).",
                  "H2's direct grouped-vs-scalar policy comparison, fine-tuning, matched pre-pruning quality, broader architectures/datasets, compute and latency remain untested. Three seeds in one small architecture support only a limited empirical conclusion."])
    (folder / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "cnn_compression.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "cnn_compression")
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")), args.output_dir)
