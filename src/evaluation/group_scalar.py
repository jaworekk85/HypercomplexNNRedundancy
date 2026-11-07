"""Two fixed activation references at the same whole-unit ablation granularity."""

import numpy as np

from src.evaluation.block_ablation import nearest_neighbor_scores, spearman


def scalar_external_scores(matrix, components=4):
    """Mean of component-wise maximum signed Pearson, outside the entire unit."""
    values = np.asarray(matrix, dtype=float).copy()
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] % components:
        raise ValueError("Expected a square unit-major component correlation matrix")
    units = len(values) // components
    if units < 2:
        raise ValueError("At least two units are required")
    for unit in range(units):
        block = slice(unit*components, (unit+1)*components)
        values[block, block] = np.nan
    component_scores = []
    for row in values:
        valid = row[np.isfinite(row)]
        component_scores.append(float(valid.max()) if len(valid) else None)
    unit_scores = []
    for unit in range(units):
        part = component_scores[unit*components:(unit+1)*components]
        unit_scores.append(float(np.mean(part)) if all(value is not None for value in part) else None)
    return unit_scores


def score_pair(grouped, scalar):
    group = nearest_neighbor_scores(grouped)
    independent = scalar_external_scores(scalar)
    if len(group) != len(independent):
        raise ValueError("Grouped and scalar widths differ")
    return {"grouped_centered_cosine": group, "scalar_pearson_mean_external_nn": independent}


def compare_with_ablations(scores, rows, top_fraction=.25):
    if any(row["unit"] != unit for unit, row in enumerate(rows)) or any(len(value) != len(rows) for value in scores.values()):
        raise ValueError("Scores and ablation unit identities do not match")
    eligible = [unit for unit in range(len(rows)) if all(value[unit] is not None and np.isfinite(value[unit]) for value in scores.values())]
    if not eligible or not 0 < top_fraction <= 1:
        raise ValueError("No shared valid units or invalid top fraction")
    k = max(1, int(len(eligible)*top_fraction))
    harm = [row["validation_loss_increase"] for row in rows]
    reports = {}
    for name, values in scores.items():
        selected = sorted(eligible, key=lambda unit: (-values[unit], unit))[:k]
        reports[name] = {"spearman_score_vs_negative_loss_harm": spearman([values[u] for u in eligible], [-harm[u] for u in eligible]),
                         "selected_top_units": selected, "top_k": k,
                         "top_mean_individual_loss_harm": float(np.mean([harm[u] for u in selected])),
                         "top_mean_individual_accuracy_drop": float(np.mean([rows[u]["validation_accuracy_drop"] for u in selected])),
                         "top_individual_low_impact_fraction": float(np.mean([rows[u]["individual_low_impact"] for u in selected]))}
    return {"eligible_units": eligible, "excluded_units": sorted(set(range(len(rows))) - set(eligible)), "criteria": reports,
            "uniform_random_expected_individual_loss_harm": float(np.mean([harm[u] for u in eligible])),
            "uniform_random_expected_individual_low_impact_fraction": float(np.mean([rows[u]["individual_low_impact"] for u in eligible]))}
