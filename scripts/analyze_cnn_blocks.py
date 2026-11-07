"""Analyze all frozen CNN blocks and individual validation ablations; no test scoring."""

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch

from scripts.cifar_cnn import make_model, write_json
from src.cnn_training import prepare_batch, seed_everything
from src.data.cifar_cached import file_sha256, load_cached_cifar
from src.evaluation.block_ablation import capture_blocks, classification_metrics, grouped_spatial, incoming_norms, nearest_neighbor_scores, predict_from_block, spearman
from src.evaluation.cnn_report import spatial_diagnostics

ROOT = Path(__file__).resolve().parents[1]


def analyze_model(kind, config, model_config, source, raw_images, raw_labels, indices, statistics, folder):
    device = torch.device(config["device"])
    seed_everything(model_config["seed"])
    model = make_model(model_config, kind).to(device).eval()
    mean = torch.tensor(statistics["mean"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(statistics["std"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    calibration = prepare_batch(torch.as_tensor(raw_images[indices["calibration"]], device=device), mean, std)
    validation = prepare_batch(torch.as_tensor(raw_images[indices["validation"]], device=device), mean, std)
    labels = raw_labels[indices["validation"]]
    initial = capture_blocks(model, calibration, config["batch_size"])
    checkpoint_path = source / kind / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["config"] != model_config:
        raise ValueError("Checkpoint configuration mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    trained = capture_blocks(model, calibration, config["batch_size"])
    with np.load(source / kind / "calibration_activations.npz") as saved:
        snapshot_errors = {stage: float(np.max(np.abs(grouped_spatial(values[-1], kind == "quaternion") - saved[stage])))
                           for stage, values in (("initial", initial), ("trained", trained))}
    val_features = capture_blocks(model, validation, config["batch_size"])
    with torch.no_grad():
        baseline_logits = np.concatenate([model(validation[start:start + config["batch_size"]]).cpu().numpy()
                                          for start in range(0, len(validation), config["batch_size"])])
    baseline = classification_metrics(baseline_logits, labels)
    prior = json.loads((source / kind / "metrics.json").read_text(encoding="utf-8"))
    checks = {"validation_accuracy_matches_frozen_model": abs(baseline["accuracy"] - prior["selected_validation"]["accuracy"]) < 1e-12,
              "validation_loss_matches_frozen_model": abs(baseline["cross_entropy"] - prior["selected_validation"]["cross_entropy"]) < 1e-5}
    checks.update({f"{stage}_last_block_snapshot_matches": error < 1e-4 for stage, error in snapshot_errors.items()})
    records = []
    for index in range(len(model.blocks)):
        block_dir = folder / f"block_{index + 1}"
        block_dir.mkdir(parents=True, exist_ok=True)
        before_values = grouped_spatial(initial[index], kind == "quaternion")
        values = grouped_spatial(trained[index], kind == "quaternion")
        before_report, before_matrices = spatial_diagnostics(before_values, kind == "quaternion")
        diagnostics, matrices = spatial_diagnostics(values, kind == "quaternion")
        np.savez_compressed(block_dir / "calibration_activations.npz", initial=before_values, trained=values)
        np.savez_compressed(block_dir / "calibration_similarity.npz",
                            **{f"initial_{key}": val for key, val in before_matrices.items()},
                            **{f"trained_{key}": val for key, val in matrices.items()})
        matrix = matrices["grouped"] if kind == "quaternion" else matrices["scalar"]
        scores, norms = nearest_neighbor_scores(matrix), incoming_norms(model, index)
        zero_logits = predict_from_block(model, index, val_features[index], batch_size=config["batch_size"])
        error = float(np.max(np.abs(zero_logits - baseline_logits)))
        checks[f"block_{index + 1}/zero_ablation_equivalence"] = error <= 1e-4
        rows = []
        for unit in range(len(scores)):
            logits = predict_from_block(model, index, val_features[index], [unit], config["batch_size"])
            metrics = classification_metrics(logits, labels)
            increase = metrics["cross_entropy"] - baseline["cross_entropy"]
            drop = baseline["accuracy"] - metrics["accuracy"]
            low_impact = increase <= config["maximum_loss_increase_for_individual_low_impact"] and drop <= config["maximum_accuracy_drop_for_individual_low_impact"]
            rows.append({"unit": unit, "nearest_neighbor_similarity": scores[unit], "incoming_weight_norm": norms[unit],
                         "validation_loss_increase": increase, "validation_accuracy_drop": drop,
                         "validation_accuracy_after_ablation": metrics["accuracy"], "individual_low_impact": low_impact})
        with (block_dir / "ablations.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        low = [row["unit"] for row in rows if row["individual_low_impact"]]
        stability_key = "grouped_stability_across_disjoint_image_halves" if kind == "quaternion" else "scalar_stability_across_disjoint_image_halves"
        stability = diagnostics[stability_key]["pair_value_pearson"]
        record = {"block": index + 1, "kind": kind, "units": len(rows), "baseline_validation": baseline,
                  "initial": before_report, "trained": diagnostics, "zero_ablation_max_abs_logit_error": error,
                  "individual_low_impact_units": low, "individual_low_impact_fraction": len(low) / len(rows),
                  "similarity_vs_negative_loss_impact_spearman": spearman(scores, [-row["validation_loss_increase"] for row in rows]),
                  "weight_norm_vs_loss_impact_spearman": spearman(norms, [row["validation_loss_increase"] for row in rows]),
                  "candidate_for_controlled_removal": kind == "quaternion" and len(low) >= config["minimum_individual_low_impact_quaternion_units"] and stability is not None and stability >= config["minimum_pair_stability_for_candidate_block"],
                  "ablations": rows}
        write_json(block_dir / "summary.json", record)
        records.append(record)
        print(f"{kind} block {index + 1}: {len(low)}/{len(rows)} individual low-impact ablations; score-impact rho={record['similarity_vs_negative_loss_impact_spearman']}", flush=True)
        del before_values, values
        gc.collect()
    checks["model_parameters_and_buffers_unchanged"] = all(torch.equal(value.detach().cpu(), before[name]) for name, value in model.state_dict().items())
    if not all(checks.values()):
        raise AssertionError([name for name, passed in checks.items() if not passed])
    result = {"kind": kind, "seed": model_config["seed"], "checkpoint_sha256": file_sha256(checkpoint_path), "blocks": records, "snapshot_max_abs_errors": snapshot_errors, "checks": checks}
    write_json(folder / "summary.json", result)
    del model, initial, trained, val_features, calibration, validation
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def render_report(records, config, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = next(record for record in records if record["kind"] == "quaternion")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for axis, block in zip(axes, q["blocks"]):
        rows = block["ablations"]
        axis.scatter([row["nearest_neighbor_similarity"] for row in rows], [row["validation_loss_increase"] for row in rows])
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.axhline(config["maximum_loss_increase_for_individual_low_impact"], color="orange", linestyle=":")
        axis.set(title=f"Quaternion block {block['block']}", xlabel="Calibration nearest-neighbor cosine", ylabel="Validation cross-entropy increase")
    fig.savefig(folder / "similarity_vs_ablation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    candidates = [block for block in q["blocks"] if block["candidate_for_controlled_removal"]]
    selected = max(candidates, key=lambda block: (block["individual_low_impact_fraction"], block["units"], -block["block"])) if candidates else None
    lines = ["# All-block activation and ablation pilot", "", "## Hypotheses", "",
             "Calibration similarity may identify individually removable quaternion channels. This is a diagnostic hypothesis, not a new method or proof of useful compression.", "",
             "## Fixed protocol", "",
             "Frozen selected checkpoints only. No optimizer updates, data augmentation or test inference. Initial activations are reconstructed using the checkpoint's recorded initialization seed. Capture outputs after each complete block (including pooling, with dropout disabled).",
             "Scores use 1000 calibration images. Functional ablations use the disjoint 1000 validation images. Stability splits calibration by whole images before pooling positions. Centered group cosine, scalar Pearson and spectral rank reuse the existing definitions.",
             f"An individually low-impact removal has validation cross-entropy increase at most {config['maximum_loss_increase_for_individual_low_impact']} and accuracy decrease at most {100*config['maximum_accuracy_drop_for_individual_low_impact']:.1f} percentage points. This threshold is prespecified, descriptive and allows removal to improve validation scores.",
             "A candidate block has at least two such individual quaternion removals and calibration pair-value stability at least 0.9. Select the largest individual low-impact fraction; ties favor more units, then the earlier block. This gate motivates an experiment; it does not establish safe simultaneous removal.", "",
             "## Measured results", "",
             "| Model | Block | Units | Maximum similarity | Effective scalar rank | Individual low-impact removals | Similarity vs negative loss impact (Spearman) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in records:
        for block in model["blocks"]:
            diag = block["trained"]
            key = "grouped_pairs" if model["kind"] == "quaternion" else "scalar_pairs"
            maximum = diag[key]["maximum"]
            rho = block["similarity_vs_negative_loss_impact_spearman"]
            lines.append(f"| {model['kind']} | {block['block']} | {block['units']} | {maximum:.3f} | {diag['rank']['effective_rank']:.2f} | {len(block['individual_low_impact_units'])}/{block['units']} | {rho:.3f} |" if maximum is not None and rho is not None else f"| {model['kind']} | {block['block']} | {block['units']} | {maximum} | {diag['rank']['effective_rank']:.2f} | {len(block['individual_low_impact_units'])}/{block['units']} | undefined |")
    lines.extend(["", "## Interpretation and decision", "",
                  f"Candidate block for the next controlled experiment: **{selected['block'] if selected else 'none'}**.",
                  "Positive Spearman means greater nearest-neighbor similarity is associated with smaller loss harm. A small or negative coefficient is a failure of that predictor in this model/layer and must be retained. Individual units can be individually dispensable while their simultaneous removal is harmful.",
                  "Temporary activation zeroing measures functional effects. No architecture was made smaller; parameter savings are zero. Real ablations remove one scalar channel, whereas quaternion ablations remove all four components; counts are not directly comparable across these granularities.",
                  "Weight norms describe raw incoming kernels and ignore BatchNorm re-scaling. These are reference measurements, not an optimized importance criterion. Threshold 0.9 for near-duplicate pairs is not a universal gate for all forms of redundancy.",
                  "Only seed 2026 is available in this report. Three-seed confirmation and physically compact 0/25/50% removal remain separate stages. Literature context uses existing references.bib keys Ayinde2019, Vecchi2020, Singh2026; no faithful reproduction or novel pruning algorithm is claimed."])
    (folder / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"candidate_blocks": [block["block"] for block in candidates], "selected_block": selected["block"] if selected else None,
            "controlled_removal_experiment_supported": selected is not None,
            "criterion": "At least two individually low-impact quaternion removals and calibration pair stability >=0.9; largest low-impact fraction."}


def run(config, output_dir):
    if (output_dir / "summary.json").exists():
        raise FileExistsError("Completed analysis exists; choose a new output directory")
    source = ROOT / config["source_run"]
    original = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if not all(file_sha256(ROOT / name) == digest for name, digest in original["source_hashes"].items()):
        raise ValueError("Frozen CNN source files changed")
    model_config = original["config"]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    (images, labels), unused_test = load_cached_cifar(ROOT / model_config["archive"])
    del unused_test
    with np.load(source / "split_indices.npz") as split:
        indices = {key: split[key] for key in ("validation", "calibration")}
    if np.intersect1d(indices["validation"], indices["calibration"]).size:
        raise AssertionError("Validation/calibration overlap")
    statistics = json.loads((source / "data.json").read_text(encoding="utf-8"))["statistics"]
    records = [analyze_model(kind, config, model_config, source, images, labels, indices, statistics, output_dir / kind) for kind in config["models"]]
    gate = render_report(records, config, output_dir)
    write_json(output_dir / "summary.json", {"status": "passed", "config": config, "models": records, "gate": gate,
               "test_used_for_scoring_or_selection": False, "optimizer_updates": 0, "compact_architecture_savings": 0,
               "source_run_summary_sha256": file_sha256(source / "summary.json"),
               "source_hashes": {name: file_sha256(ROOT / name) for name in ("scripts/analyze_cnn_blocks.py", "src/evaluation/block_ablation.py", "src/evaluation/cnn_report.py")}})
    print(f"Analysis finished; candidate block: {gate['selected_block']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "block_analysis.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "block_analysis")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(json.loads(args.config.read_text(encoding="utf-8")), args.output_dir)
