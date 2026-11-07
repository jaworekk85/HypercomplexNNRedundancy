"""Inclusive descriptive thresholds with machine-rounding tolerance."""


def at_most(value, limit):
    return value <= limit + 1e-12


def correct_ablation_labels(record, config):
    """Keep numerical measurements; correct strict floating-point boundary labels."""
    if "inclusive_threshold_correction" in record:
        return record
    corrections = []
    for block in record["blocks"]:
        previous = block["individual_low_impact_units"]
        for row in block["ablations"]:
            row["individual_low_impact"] = at_most(row["validation_loss_increase"], config["maximum_loss_increase_for_individual_low_impact"]) and at_most(row["validation_accuracy_drop"], config["maximum_accuracy_drop_for_individual_low_impact"])
        low = [row["unit"] for row in block["ablations"] if row["individual_low_impact"]]
        block["individual_low_impact_units"] = low
        block["individual_low_impact_fraction"] = len(low) / block["units"]
        stability = block["trained"].get("grouped_stability_across_disjoint_image_halves", {}).get("pair_value_pearson")
        block["candidate_for_controlled_removal"] = record["kind"] == "quaternion" and len(low) >= config["minimum_individual_low_impact_quaternion_units"] and stability is not None and stability >= config["minimum_pair_stability_for_candidate_block"]
        if previous != low:
            corrections.append({"block": block["block"], "previous_units": previous, "corrected_units": low})
    record["inclusive_threshold_correction"] = {"tolerance": 1e-12, "changed_blocks": corrections,
        "reason": "For example, 0.783-0.778 equals 0.0050000000000000044; the prespecified <=0.005 threshold includes this boundary."}
    return record
