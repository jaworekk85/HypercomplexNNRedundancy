import numpy as np
import pytest

from src.activations.diagnostics import scalar_pearson
from src.activations.similarity import grouped_cosine
from src.evaluation.group_scalar import compare_with_ablations, scalar_external_scores, score_pair


def test_scalar_scores_exclude_all_within_unit_matches():
    matrix = np.full((8, 8), .2)
    matrix[:4, :4] = .99
    matrix[4:, 4:] = .99
    np.fill_diagonal(matrix, 1)
    np.testing.assert_allclose(scalar_external_scores(matrix), [.2, .2])


def test_independent_scalar_matching_can_use_different_component_partners():
    rng = np.random.default_rng(471)
    first = rng.normal(size=(2000, 4))
    values = np.stack([first, first[:, [1, 2, 3, 0]]], axis=1)
    grouped, _ = grouped_cosine(values)
    scalar, _ = scalar_pearson(values)
    scores = score_pair(grouped, scalar)
    np.testing.assert_allclose(scores["scalar_pearson_mean_external_nn"], [1, 1], atol=1e-12)
    assert max(scores["grouped_centered_cosine"]) < .1


def test_undefined_components_use_shared_exclusion_and_ties_are_deterministic():
    matrix = np.eye(12)
    matrix[0] = matrix[:, 0] = np.nan
    assert scalar_external_scores(matrix)[0] is None
    scores = {"group": [None, .7, .7, .2], "scalar": [.9, .4, .8, .1]}
    rows = [{"unit": i, "validation_loss_increase": loss, "validation_accuracy_drop": loss,
             "individual_low_impact": loss <= .01} for i, loss in enumerate([.2, 0, .01, .03])]
    result = compare_with_ablations(scores, rows, .5)
    assert result["eligible_units"] == [1, 2, 3]
    assert result["criteria"]["group"]["selected_top_units"] == [1]
    assert result["criteria"]["scalar"]["selected_top_units"] == [2]
    assert result["uniform_random_expected_individual_loss_harm"] == pytest.approx(.04/3)
