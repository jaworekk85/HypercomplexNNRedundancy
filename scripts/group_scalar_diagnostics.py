"""Compare fixed grouped/scalar scores using saved calibration and validation only."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch

from scripts.cifar_cnn import write_json
from src.activations.diagnostics import scalar_pearson
from src.activations.similarity import grouped_cosine
from src.data.cifar_cached import file_sha256
from src.evaluation.block_ablation import spearman
from src.evaluation.group_scalar import compare_with_ablations, score_pair

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["scripts/group_scalar_diagnostics.py", "src/evaluation/group_scalar.py"]
GROUP = "grouped_centered_cosine"
SCALAR = "scalar_pearson_mean_external_nn"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def matrices(values):
    flat = values.reshape(-1, values.shape[2], 4)
    scalar, _ = scalar_pearson(flat)
    grouped, _ = grouped_cosine(flat)
    return grouped, scalar


def analyze_case(seed, block, source_record, source, config, folder):
    with np.load(source / "calibration_similarity.npz") as saved:
        grouped, scalar = saved["trained_grouped"], saved["trained_scalar"]
    with np.load(source / "calibration_activations.npz") as saved:
        values = saved["trained"]
    recomputed_group, recomputed_scalar = matrices(values)
    checks = {"saved_grouped_matrix_matches_snapshots": bool(np.allclose(grouped, recomputed_group, atol=1e-10, rtol=1e-10, equal_nan=True)),
              "saved_scalar_matrix_matches_snapshots": bool(np.allclose(scalar, recomputed_scalar, atol=1e-10, rtol=1e-10, equal_nan=True)),
              "calibration_snapshot_shape": values.shape == (1000, (32 // (2**block))**2, source_record["units"], 4)}
    scores = score_pair(grouped, scalar)
    previous = [row["nearest_neighbor_similarity"] for row in source_record["ablations"]]
    checks["grouped_scores_match_previous_analysis"] = all((a is None and b is None) or (a is not None and b is not None and abs(a-b) <= 1e-12) for a,b in zip(scores[GROUP], previous))
    original_rho = spearman(scores[GROUP], [-row["validation_loss_increase"] for row in source_record["ablations"]])
    expected_rho = source_record["similarity_vs_negative_loss_impact_spearman"]
    checks["grouped_spearman_matches_previous_analysis"] = original_rho == expected_rho or (original_rho is not None and expected_rho is not None and abs(original_rho-expected_rho) <= 1e-12)
    first = score_pair(*matrices(values[::2]))
    second = score_pair(*matrices(values[1::2]))
    stability = {name: spearman(first[name], second[name]) for name in scores}
    report = compare_with_ablations(scores, source_record["ablations"], config["top_fraction"])
    group, independent = (report["criteria"][name] for name in (GROUP, SCALAR))
    rho = [r["spearman_score_vs_negative_loss_harm"] for r in (group, independent)]
    record = {"seed": seed, "block": block, "units": source_record["units"], "scores": scores,
              "calibration_image_half_scores": {"even_images": first, "odd_images": second},
              "calibration_image_half_score_spearman": stability, "target_ablations": source_record["ablations"], **report,
              "group_minus_scalar_spearman": rho[0]-rho[1] if all(value is not None for value in rho) else None,
              "group_minus_scalar_top_individual_loss_harm": group["top_mean_individual_loss_harm"]-independent["top_mean_individual_loss_harm"],
              "checks": checks,
              "source_artifacts": {str((source / name).relative_to(ROOT)).replace("\\", "/"): file_sha256(source / name)
                                   for name in ("calibration_similarity.npz", "calibration_activations.npz")}}
    if not all(checks.values()):
        raise AssertionError([key for key, passed in checks.items() if not passed])
    write_json(folder / f"seed_{seed}" / f"block_{block}" / "summary.json", record)
    print(f"seed {seed}, block {block}: grouped rho={rho[0]:.3f}, scalar rho={rho[1]:.3f}, paired delta={record['group_minus_scalar_spearman']:+.3f}", flush=True)
    return record


def primary_summary(records, config):
    primary = [record for record in records if record["block"] == config["primary_block"]]
    if [r["seed"] for r in primary] != config["seeds"]:
        raise ValueError("Missing primary seed")
    delta = [r["group_minus_scalar_spearman"] for r in primary]
    losses = [r["group_minus_scalar_top_individual_loss_harm"] for r in primary]
    complete = all(value is not None for value in delta)
    return {"block": config["primary_block"], "seeds": config["seeds"], "paired_spearman_differences": delta,
            "mean_paired_spearman_difference": float(np.mean(delta)) if complete else None,
            "sample_sd_paired_spearman_difference": float(np.std(delta, ddof=1)) if complete else None,
            "paired_top_individual_loss_harm_differences": losses,
            "repeatable_grouped_advantage_under_prespecified_rule": complete and all(value > 0 for value in delta) and all(value <= 1e-12 for value in losses),
            "rule_is_descriptive_not_significance_test": True}


def render(records, primary, config, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [r for r in records if r["block"] == config["primary_block"]]
    x = np.arange(len(selected))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for offset, name, label in ((-.18, GROUP, "Grouped centered cosine"), (.18, SCALAR, "Independent scalar Pearson")):
        axes[0].bar(x+offset, [r["criteria"][name]["spearman_score_vs_negative_loss_harm"] for r in selected], width=.34, label=label)
    for offset, name, label in ((-.24, GROUP, "Grouped"), (0, SCALAR, "Scalar")):
        axes[1].bar(x+offset, [r["criteria"][name]["top_mean_individual_loss_harm"] for r in selected], width=.23, label=label)
    axes[1].bar(x+.24, [r["uniform_random_expected_individual_loss_harm"] for r in selected], width=.23, label="Uniform random expectation")
    for axis in axes:
        axis.set_xticks(x, [str(r["seed"]) for r in selected])
        axis.set_xlabel("Training seed (primary block 3)")
        axis.axhline(0, color="gray", linewidth=.7)
        axis.legend(fontsize=8)
    axes[0].set(ylabel="Spearman(score, negative individual CE harm)", title="Higher predicts lower ablation harm")
    axes[1].set(ylabel="Mean CE increase from separate individual removals", title="Top-quarter ranking diagnostic; lower is better")
    fig.savefig(folder / "primary_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    with (folder / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        flat = [{"seed": r["seed"], "block": r["block"], "units": r["units"], "eligible_units": len(r["eligible_units"]),
                 **{f"{name}_{key}": value for name, report in r["criteria"].items() for key,value in report.items() if key != "selected_top_units"},
                 "group_minus_scalar_spearman": r["group_minus_scalar_spearman"],
                 "group_minus_scalar_top_individual_loss_harm": r["group_minus_scalar_top_individual_loss_harm"]} for r in records]
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ["# Grouped versus independent scalar activation diagnostics", "", "## Hypotheses", "",
             "Whole-quaternion grouping may help predict individual whole-unit removal harm. The user's broader hypothesis is that structural weight constraints may explain benefits without any unique need for quaternion algebra. This within-model diagnostic cannot identify that causal explanation; see docs/STRUCTURE_HYPOTHESIS.md. No new method or general advantage is assumed.", "",
             "## Fixed protocol", "",
             "Saved calibration matrices and trained activations only; existing single-whole-unit validation ablations. Three seeds; block 3 is primary, blocks 1/2 secondary. No new inference, training or test access. The calibration/validation data were already explored; this is a descriptive follow-up, not an independent blind confirmation. See PROTOCOL.md for exact definitions and input hashes in summary.json.",
             "Grouped score = maximum signed centered four-component cosine to another unit. Scalar score = mean of four maximum signed scalar Pearson correlations outside the entire current unit. Scalar features are unit-major in saved matrices. Both are judged on the identical four-channel ablation targets and shared valid-unit set. Scalar matching has more candidate partners, independent coordinates and per-component scale normalization. A difference does not isolate the algebra, matching restriction or scale effects.", "",
             "## Measured results", "", "| Seed | Block | Units | Grouped Spearman | Scalar Spearman | Group-minus-scalar | Grouped / scalar / random expected mean individual CE harm, top quarter |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in records:
        group, scalar = (r["criteria"][name] for name in (GROUP, SCALAR))
        lines.append(f"| {r['seed']} | {r['block']} | {len(r['eligible_units'])}/{r['units']} | {group['spearman_score_vs_negative_loss_harm']:.3f} | {scalar['spearman_score_vs_negative_loss_harm']:.3f} | {r['group_minus_scalar_spearman']:+.3f} | {group['top_mean_individual_loss_harm']:.5f} / {scalar['top_mean_individual_loss_harm']:.5f} / {r['uniform_random_expected_individual_loss_harm']:.5f} |")
    lines.extend(["", f"Primary block: paired Spearman differences {primary['paired_spearman_differences']}; mean {primary['mean_paired_spearman_difference']:+.3f}, sample SD {primary['sample_sd_paired_spearman_difference']:.3f} across three training seeds.",
                  f"Prespecified descriptive grouped-advantage rule passes: **{primary['repeatable_grouped_advantage_under_prespecified_rule']}**. It requires higher grouped Spearman and no higher top-quarter individual loss harm in every primary seed. It is not a significance test.", "",
                  "| Seed | Block | Grouped image-half score-rank stability | Scalar image-half score-rank stability |", "| ---: | ---: | ---: | ---: |"])
    for r in records:
        lines.append(f"| {r['seed']} | {r['block']} | {r['calibration_image_half_score_spearman'][GROUP]:.3f} | {r['calibration_image_half_score_spearman'][SCALAR]:.3f} |")
    lines.extend(["", "## Interpretation and scope", "",
                  "A positive within-model rank association need not produce a better simultaneous-removal policy. The top-quarter diagnostic averages separate single-unit removals; it is not an observed quarter-size compact-model loss. The previous greedy-pruning policy includes a norm member rule and differs from either nearest-neighbor ranking here. No old test accuracies are relabeled as new policy results.",
                  "Undefined scores are excluded from both criteria together. Whole images, not pooled positions, are separated for stability. Neurons are not aligned across independently trained seeds. Secondary blocks cannot rescue a failed primary criterion; no score variant is selected from the outcomes.",
                  "No Hamilton products are evaluated in either similarity score: grouped cosine and Pearson are ordinary real-vector statistics. Even a grouped advantage would not establish that quaternion algebra itself caused the training advantage. Literature context: Ayinde2019, Vecchi2020, ZhangEtAl2021 (references.bib). No novelty or broad causal claim follows."])
    (folder / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(config, folder):
    if (folder / "summary.json").exists():
        raise FileExistsError("Completed grouped/scalar results exist; choose another output directory")
    original_folder = ROOT / config["source_compression"]
    original = read(original_folder / "summary.json")
    if read(original_folder / "verification.json")["status"] != "passed":
        raise ValueError("Original artifacts are not verified")
    old_sources = original["protocol"]["source_hashes"]
    if not all(file_sha256(ROOT / name) == digest for name,digest in old_sources.items()):
        raise ValueError("Frozen numerical sources changed")
    folder.mkdir(parents=True, exist_ok=True)
    protocol = {"config": config, "source_hashes": {**old_sources, **{name:file_sha256(ROOT / name) for name in SOURCES}},
                "original_compression_summary_sha256": file_sha256(original_folder / "summary.json")}
    write_json(folder / "protocol.json", protocol)
    combined_path = original_folder / "block_analysis" / "summary.json"
    combined = read(combined_path)
    input_hashes = {str(combined_path.relative_to(ROOT)).replace("\\", "/"): file_sha256(combined_path)}
    split_path = ROOT / config["source_baseline"] / "split_indices.npz"
    with np.load(split_path) as split:
        validation, calibration = split["validation"], split["calibration"]
    input_hashes[str(split_path.relative_to(ROOT)).replace("\\", "/")] = file_sha256(split_path)
    checks = {"calibration_validation_disjoint": np.intersect1d(validation, calibration).size == 0,
              "held_out_sample_counts": len(validation) == len(calibration) == 1000,
              "no_inference_training_or_test": config["inference"] is False and config["training_updates"] == 0 and config["test_used"] is False}
    records = []
    for seed in config["seeds"]:
        source_record = next(r for r in combined["records"] if r["kind"] == "quaternion" and r["seed"] == seed)
        checks[f"seed_{seed}/previous_model_integrity_checks"] = all(source_record["checks"].values())
        for block in config["blocks"]:
            source = (ROOT / config["source_initial_analysis"] / "quaternion" / f"block_{block}" if seed == 2026 else
                      original_folder / "block_analysis" / f"seed_{seed}" / "quaternion" / f"block_{block}")
            prior_block = next(b for b in source_record["blocks"] if b["block"] == block)
            record = analyze_case(seed, block, prior_block, source, config, folder)
            records.append(record)
            input_hashes.update(record["source_artifacts"])
    if not all(checks.values()):
        raise AssertionError([name for name,passed in checks.items() if not passed])
    primary = primary_summary(records, config)
    summary = {"status": "completed", "config": config, "records": records, "primary": primary, "protocol": protocol,
               "input_hashes": input_hashes, "checks": checks, "model_inference_calls": 0, "optimizer_updates": 0,
               "test_used": False, "compact_architecture_savings": 0,
               "environment": {"python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__), "device": "CPU", "numeric_dtype": "float64"}}
    write_json(folder / "summary.json", summary)
    render(records, primary, config, folder)
    print(f"Grouped/scalar diagnostic complete; repeatable primary advantage: {primary['repeatable_grouped_advantage_under_prespecified_rule']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "group_scalar.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "group_scalar")
    args = parser.parse_args()
    run(read(args.config), args.output_dir)
