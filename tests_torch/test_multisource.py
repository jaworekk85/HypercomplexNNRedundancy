from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(2)

from src.compression.internal_merging import expand_right
from src.compression.multisource import (fit_full_multi, fit_quaternion_multi, fit_real_multi, merge_quaternion_multi,
    merge_real, pair_maps, quaternion_maps, real_maps, real_pair_fits, select_disjoint, substitute_multi)
from src.models.cifar_cnn_torch import CifarCNN


def test_unregularized_multi_fits_recover_planted_maps():
    rng = np.random.default_rng(3)
    values = rng.normal(size=(40, 9, 5, 4))
    kept, victim = [0, 1, 3, 4], 2
    planted = rng.normal(size=(4, 4))
    values[..., victim, :] = np.einsum("koi,npki->npo", expand_right(planted), values[..., kept, :])
    np.testing.assert_allclose(fit_quaternion_multi(values, victim, kept, 0.), planted, atol=1e-10)
    general = rng.normal(size=(4, 4, 4))
    values[..., victim, :] = np.einsum("koi,npki->npo", general, values[..., kept, :])
    np.testing.assert_allclose(fit_full_multi(values, victim, kept, 0.), general, atol=1e-10)
    scalar = rng.normal(size=(30, 9, 6))
    scalar[..., 5] = scalar[..., :5]@np.array([.5, -.2, 0., .3, 1.])
    np.testing.assert_allclose(fit_real_multi(scalar, 5, [0, 1, 2, 3, 4], 0.), [.5, -.2, 0., .3, 1.], atol=1e-10)


def test_ridge_shrinks_coefficients():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(20, 4, 3, 4))
    small = np.linalg.norm(fit_quaternion_multi(values, 2, [0, 1], 1.))
    assert small < np.linalg.norm(fit_quaternion_multi(values, 2, [0, 1], 0.))


def test_quaternion_multi_fold_is_exact_and_full_projection_is_lossy():
    torch.manual_seed(5)
    model = CifarCNN(q_widths=(2, 5, 3)).eval()
    original = deepcopy(model.state_dict())
    fit = np.random.default_rng(6).normal(size=(12, 16, 5, 4))
    maps = quaternion_maps(fit, [1, 3], "quaternion_right", 1e-3)
    compact, kept, diagnostic = merge_quaternion_multi(model, 1, maps)
    features = torch.randn(4, 20, 8, 8)
    channels = [c*5+u for c in range(4) for u in kept]
    with torch.no_grad():
        expected = model.blocks[2](substitute_multi(features, maps))
        actual = compact.blocks[2](features[:, channels])
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    assert diagnostic["discarded_energy_fraction"] < 1e-25
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
    _, _, projected = merge_quaternion_multi(model, 1, quaternion_maps(fit, [1, 3], "full_real", 1e-3))
    assert projected["discarded_energy_fraction"] > 0
    assert sum(p.numel() for p in compact.parameters()) < sum(p.numel() for p in model.parameters())


def test_real_merges_are_exact_and_pairs_are_disjoint():
    torch.manual_seed(7)
    model = CifarCNN("real", real_widths=(4, 6, 5)).eval()
    fit = np.abs(np.random.default_rng(8).normal(size=(10, 16, 6)))
    pairs = select_disjoint(real_pair_fits(fit, model.blocks[2][0].weight.detach().numpy()), 3)
    assert len({row["keeper"] for row in pairs} | {row["victim"] for row in pairs}) == 6
    features = torch.randn(3, 6, 8, 8)
    for maps in (pair_maps(pairs[:2], 6), real_maps(fit, [0, 4], 1e-3)):
        compact, kept = merge_real(model, 1, maps, 6)
        with torch.no_grad():
            expected = model.blocks[2](substitute_multi(features, maps, quaternion=False))
            actual = compact.blocks[2](features[:, kept])
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        assert sum(p.numel() for p in compact.parameters()) < sum(p.numel() for p in model.parameters())
