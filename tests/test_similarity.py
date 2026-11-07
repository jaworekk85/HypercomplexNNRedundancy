import numpy as np
import pytest

from src.activations.similarity import grouped_cosine


def test_identical_scaled_and_opposite_quaternion_units():
    rng = np.random.default_rng(13)
    unit = rng.normal(size=(10, 4))
    activations = np.stack([unit, 2 * unit + 3, -unit], axis=1)
    similarity, valid = grouped_cosine(activations)
    assert valid.all()
    np.testing.assert_allclose(similarity, [[1, 1, -1], [1, 1, -1], [-1, -1, 1]], atol=1e-14)


def test_constant_and_zero_units_are_excluded():
    rng = np.random.default_rng(14)
    activations = rng.normal(size=(10, 3, 4))
    activations[:, 0] = 0
    activations[:, 1] = [1, 2, 3, 4]
    similarity, valid = grouped_cosine(activations)
    np.testing.assert_array_equal(valid, [False, False, True])
    assert np.isnan(similarity[:2]).all()
    assert similarity[2, 2] == pytest.approx(1)


def test_reject_nonfinite_or_lost_grouping():
    with pytest.raises(ValueError):
        grouped_cosine(np.zeros((10, 12)))
    with pytest.raises(ValueError):
        grouped_cosine(np.full((10, 3, 4), np.nan))
