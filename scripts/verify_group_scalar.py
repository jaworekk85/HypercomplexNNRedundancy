"""Verify the paired diagnostic directly from saved correlation matrices/targets."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.cifar_cnn import write_json
from src.data.cifar_cached import file_sha256
from src.evaluation.block_ablation import spearman

ROOT = Path(__file__).resolve().parents[1]
GROUP = "grouped_centered_cosine"
SCALAR = "scalar_pearson_mean_external_nn"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def same_values(a, b):
    return len(a) == len(b) and all((x is None and y is None) or (x is not None and y is not None and abs(x-y) <= 1e-12) for x,y in zip(a,b))


def verify(folder):
    summary = read(folder / "summary.json")
    config = summary["config"]
    checks = {"completed": summary["status"] == "completed", "all_nine_seed_blocks": len(summary["records"]) == 9,
              "no_model_inference": summary["model_inference_calls"] == 0, "no_updates": summary["optimizer_updates"] == 0,
              "no_test_use": summary["test_used"] is False, "no_physical_savings_claim": summary["compact_architecture_savings"] == 0,
              "numerical_sources_unchanged": all(file_sha256(ROOT / name) == digest for name,digest in summary["protocol"]["source_hashes"].items()),
              "all_input_hashes_unchanged": all(file_sha256(ROOT / name) == digest for name,digest in summary["input_hashes"].items()),
              "failed_shape_check_and_protocol_retained": (folder / "FAILED_CHECKS.md").exists() and (folder / "failed_run_protocol.json").exists()}
    checks.update({f"source/{key}": value for key,value in summary["checks"].items()})
    previous = read(ROOT / config["source_compression"] / "block_analysis" / "summary.json")
    for record in summary["records"]:
        seed, block, units = record["seed"], record["block"], record["units"]
        prefix = f"{seed}/block_{block}"
        checks.update({f"{prefix}/{key}": value for key,value in record["checks"].items()})
        source = next(ROOT / name for name in record["source_artifacts"] if name.endswith("calibration_similarity.npz"))
        with np.load(source) as data:
            scalar, grouped = data["trained_scalar"], data["trained_grouped"]
        # Independent vectorized mask: no channel from the removed unit can be a partner.
        ids = np.repeat(np.arange(units), 4)
        allowed = (ids[:, None] != ids[None, :]) & np.isfinite(scalar)
        component_max = np.where(allowed, scalar, -np.inf).max(1).reshape(units, 4)
        scalar_scores = [float(row.mean()) if np.isfinite(row).all() else None for row in component_max]
        group_matrix = np.where(~np.eye(units, dtype=bool) & np.isfinite(grouped), grouped, -np.inf)
        group_scores = [float(value) if np.isfinite(value) else None for value in group_matrix.max(1)]
        checks[prefix + "/scalar_scores_recomputed_with_external_mask"] = same_values(scalar_scores, record["scores"][SCALAR])
        checks[prefix + "/grouped_scores_recomputed"] = same_values(group_scores, record["scores"][GROUP])
        old_model = next(r for r in previous["records"] if r["kind"] == "quaternion" and r["seed"] == seed)
        old_block = next(b for b in old_model["blocks"] if b["block"] == block)
        checks[prefix + "/same_whole_unit_targets_unchanged"] = old_block["ablations"] == record["target_ablations"]
        eligible = [u for u in range(units) if scalar_scores[u] is not None and group_scores[u] is not None]
        checks[prefix + "/shared_valid_unit_set"] = eligible == record["eligible_units"]
        targets = record["target_ablations"]
        harm = np.array([row["validation_loss_increase"] for row in targets])
        k = max(1, int(len(eligible)*config["top_fraction"]))
        for name, scores in ((GROUP, group_scores), (SCALAR, scalar_scores)):
            details = record["criteria"][name]
            rho = spearman([scores[u] for u in eligible], [-harm[u] for u in eligible])
            checks[prefix + f"/{name}/spearman_recomputed"] = same_values([rho], [details["spearman_score_vs_negative_loss_harm"]])
            order = np.lexsort((np.array(eligible), -np.array([scores[u] for u in eligible])))
            selected = [eligible[i] for i in order[:k]]
            checks[prefix + f"/{name}/top_quarter_label_free_selection"] = selected == details["selected_top_units"]
            checks[prefix + f"/{name}/individual_harm_recomputed"] = abs(float(harm[selected].mean()) - details["top_mean_individual_loss_harm"]) <= 1e-12
            low = np.mean([targets[u]["individual_low_impact"] for u in selected])
            checks[prefix + f"/{name}/individual_low_impact_fraction_recomputed"] = abs(float(low)-details["top_individual_low_impact_fraction"]) <= 1e-12
            stability = spearman(record["calibration_image_half_scores"]["even_images"][name], record["calibration_image_half_scores"]["odd_images"][name])
            checks[prefix + f"/{name}/score_rank_stability_recomputed"] = same_values([stability], [record["calibration_image_half_score_spearman"][name]])
        checks[prefix + "/uniform_random_expectation_recomputed"] = abs(float(harm[eligible].mean()) - record["uniform_random_expected_individual_loss_harm"]) <= 1e-12
    primary = [r for r in summary["records"] if r["block"] == config["primary_block"]]
    delta = [r["criteria"][GROUP]["spearman_score_vs_negative_loss_harm"] - r["criteria"][SCALAR]["spearman_score_vs_negative_loss_harm"] for r in primary]
    effects = summary["primary"]
    checks["primary_seed_order_preserved"] = [r["seed"] for r in primary] == config["seeds"]
    checks["primary_paired_effects_recomputed"] = same_values(delta, effects["paired_spearman_differences"])
    checks["primary_mean_recomputed"] = abs(float(np.mean(delta))-effects["mean_paired_spearman_difference"]) <= 1e-12
    checks["primary_seed_sd_recomputed"] = abs(float(np.std(delta, ddof=1))-effects["sample_sd_paired_spearman_difference"]) <= 1e-12
    passed = all(value > 0 for value in delta) and all(r["criteria"][GROUP]["top_mean_individual_loss_harm"] <= r["criteria"][SCALAR]["top_mean_individual_loss_harm"] + 1e-12 for r in primary)
    checks["honest_prespecified_primary_decision"] = passed == effects["repeatable_grouped_advantage_under_prespecified_rule"]
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "verifier_sha256": file_sha256(Path(__file__)),
              "scope": "Saved matrices, targets and hashes only; no training, inference or test data."}
    write_json(folder / "verification.json", result)
    print(f"Grouped/scalar artifact verification: {result['status']}; {len(checks)} checks", flush=True)
    if not all(checks.values()):
        raise AssertionError([key for key,passed in checks.items() if not passed])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?", default=ROOT / "results" / "group_scalar")
    verify(parser.parse_args().folder)
