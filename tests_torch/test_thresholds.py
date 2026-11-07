from copy import deepcopy

from src.evaluation.thresholds import at_most, correct_ablation_labels


def test_inclusive_accuracy_threshold_handles_subtraction_rounding():
    assert at_most(0.783 - 0.778, 0.005)
    assert at_most(0.783 - 0.763, 0.02)
    assert not at_most(0.783 - 0.777, 0.005)
    config = {"maximum_loss_increase_for_individual_low_impact": .01,
              "maximum_accuracy_drop_for_individual_low_impact": .005,
              "minimum_individual_low_impact_quaternion_units": 2,
              "minimum_pair_stability_for_candidate_block": .9}
    record = {"kind": "quaternion", "blocks": [{"block": 3, "units": 2,
              "individual_low_impact_units": [1], "trained": {"grouped_stability_across_disjoint_image_halves": {"pair_value_pearson": .99}},
              "ablations": [{"unit": 0, "validation_loss_increase": .009, "validation_accuracy_drop": .783-.778},
                            {"unit": 1, "validation_loss_increase": .003, "validation_accuracy_drop": 0}]}]}
    corrected = correct_ablation_labels(record, config)
    assert corrected["blocks"][0]["individual_low_impact_units"] == [0, 1]
    assert corrected["blocks"][0]["candidate_for_controlled_removal"]
    once = deepcopy(corrected)
    assert correct_ablation_labels(corrected, config) == once  # Resume preserves the original correction audit.
