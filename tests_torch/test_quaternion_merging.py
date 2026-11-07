from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(2)

from src.compression.quaternion_merging import (aggregate_errors, all_pair_fits, choose_disjoint_pairs,
    fit_affine, logits_from_units, merged_last_block, predicted_logits, right_hamilton)
from src.models.cifar_cnn_torch import CifarCNN
from src.models.quaternion_conv_torch import hamilton_kernel


def test_right_hamilton_orientation_matches_hand_product_and_frozen_kernel():
    a = np.array([.7, -.2, .3, .5])
    x = np.array([2., 3., -1., 4.])
    np.testing.assert_allclose(right_hamilton(a)@x, [.3, 0., -2.4, 4.5], atol=1e-14)
    expanded = hamilton_kernel(torch.tensor(a).reshape(4, 1, 1, 1, 1)).reshape(4, 4).numpy()
    np.testing.assert_array_equal(right_hamilton(a), expanded)


@pytest.mark.parametrize("mode", ["quaternion_right", "diagonal", "full_real"])
def test_known_affine_relation_is_recovered_on_unseen_examples(mode):
    rng = np.random.default_rng(31)
    matrix = right_hamilton([.7, -.2, .3, .5]) if mode == "quaternion_right" else np.diag([.3, -.4, .5, .2]) if mode == "diagonal" else rng.normal(size=(4, 4))
    bias = rng.normal(size=4)
    fit_x, check_x = rng.normal(size=(70, 4)), rng.normal(size=(30, 4))
    result = fit_affine(fit_x, fit_x@matrix.T+bias, mode)
    np.testing.assert_allclose(result["matrix"], matrix, atol=1e-12)
    np.testing.assert_allclose(check_x@np.asarray(result["matrix"]).T+result["bias"], check_x@matrix.T+bias, atol=1e-12)
    assert result["linear_coefficients"] == (16 if mode == "full_real" else 4)
    constant = fit_affine(np.ones((10, 4)), np.full((10, 4), 3.), mode)
    assert np.isfinite(constant["matrix"]).all()
    assert np.isfinite(constant["bias"]).all()


def test_fit_and_pair_selection_ignore_the_check_half():
    rng = np.random.default_rng(14)
    x, y, weight = rng.normal(size=(30, 4, 4)), rng.normal(size=(20, 4, 4)), rng.normal(size=(3, 16))
    a = all_pair_fits(x, y, weight, "quaternion_right")
    b = all_pair_fits(x, y+100., weight, "quaternion_right")
    for first, second in zip(a, b):
        for key in ("matrix", "bias", "target_fit_mean", "fit_centered_logit_mse"):
            assert first[key] == second[key]
    first, second = choose_disjoint_pairs(a, 2), choose_disjoint_pairs(b, 2)
    assert [(row["keeper"], row["victim"]) for row in first] == [(row["keeper"], row["victim"]) for row in second]
    assert len({row[key] for row in first for key in ("keeper", "victim")}) == 4


@pytest.mark.parametrize("compensation", ["affine", "mean_only", "none"])
def test_physical_merging_matches_cached_logit_formula_without_source_mutation(compensation):
    torch.manual_seed(43)
    model = CifarCNN(q_widths=(2, 3, 4)).eval()
    original = deepcopy(model.state_dict())
    pair = {"keeper": 0, "victim": 2, "matrix": right_hamilton([.6, -.1, .2, .3]).tolist(),
            "bias": [.2, -.3, .1, -.4], "target_fit_mean": [.5, .2, .3, .1]}
    images = torch.randn(5, 3, 32, 32)
    with torch.no_grad():
        values = model.forward_features(images).mean((2, 3)).reshape(5, 4, 4).transpose(1, 2).numpy()
        baseline = model(images).numpy()
    weight, bias = model.readout.weight.detach().numpy(), model.readout.bias.detach().numpy()
    np.testing.assert_allclose(logits_from_units(values, weight, bias), baseline, atol=1e-6)
    expected = predicted_logits(values, weight, bias, [pair], compensation)
    compact, kept = merged_last_block(model, [pair], compensation)
    assert kept == [0, 1, 3]
    with torch.no_grad():
        actual = compact(images).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
    assert sum(p.numel() for p in model.parameters())-sum(p.numel() for p in compact.parameters()) == 192
    restored, _ = merged_last_block(model, [pair], "none")
    restored.load_state_dict(compact.state_dict())
    with torch.no_grad():
        assert torch.equal(restored(images), compact(images))
    zero, _ = merged_last_block(model, [], "none")
    with torch.no_grad():
        assert torch.equal(model(images), zero(images))


def test_mean_only_control_prevents_trivial_constant_offsets_becoming_a_signal():
    rng = np.random.default_rng(3)
    values = rng.normal(size=(40, 2, 4))
    values[:, 1] = [1., 2., 3., 4.]
    fit = fit_affine(values[:, 0], values[:, 1], "full_real")
    pair = {"keeper": 0, "victim": 1, **fit}
    errors = aggregate_errors(values, rng.normal(size=(3, 8)), np.zeros(3), [pair])
    assert errors["mean_only"] < 1e-20
    assert errors["ratio_vs_mean_only"] is None
