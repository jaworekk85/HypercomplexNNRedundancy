from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(2)
from torch.nn import functional as F

from src.compression.internal_merging import (expand_right, fit_linear, grouped_to_tensor,
    merge_internal, outgoing_addition, project_right, substitute_features)
from src.compression.quaternion_merging import right_hamilton
from src.models.cifar_cnn_torch import CifarCNN


def test_projection_is_orthogonal_and_quaternion_composition_order_is_correct():
    rng = np.random.default_rng(8)
    a, w = rng.normal(size=4), rng.normal(size=4)
    product = right_hamilton(w)@right_hamilton(a)
    coefficients, projected = project_right(product)
    np.testing.assert_allclose(product, projected, atol=1e-14)
    # a*w, independently expanded by the scalar/vector Hamilton formula.
    expected = np.r_[a[0]*w[0]-a[1:]@w[1:], a[0]*w[1:]+w[0]*a[1:]+np.cross(a[1:], w[1:])]
    np.testing.assert_allclose(coefficients, expected, atol=1e-14)
    full = rng.normal(size=(4, 4))
    _, projected = project_right(full)
    for coefficient in np.eye(4):
        assert abs(np.sum((full-projected)*right_hamilton(coefficient))) < 1e-12
    assert np.linalg.norm(full-projected) > .1


@pytest.mark.parametrize("mode", ["quaternion_right", "full_real"])
def test_zero_offset_fit_recovers_known_mapping(mode):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(50, 4))
    a = right_hamilton([.3, -.4, .2, .7]) if mode == "quaternion_right" else rng.normal(size=(4, 4))
    fit = fit_linear(x, x@a.T, mode)
    np.testing.assert_allclose(fit["matrix"], a, atol=1e-12)
    assert not fit["intercept"]


def test_internal_folding_is_exact_for_substitution_and_does_not_mutate_source():
    torch.manual_seed(16)
    model = CifarCNN(q_widths=(2, 4, 5)).eval()
    original = deepcopy(model.state_dict())
    pair = {"keeper": 0, "victim": 2, "maps": {
        "quaternion_right": {"matrix": right_hamilton([.6, -.2, .1, .3]).tolist()},
        "full_real": {"matrix": [[.7,.1,.2,.1],[.1,.8,.2,-.1],[.1,.2,.6,.2],[.2,-.1,.1,.5]]}}}
    features = torch.randn(6, 16, 8, 8)
    compact, kept, diagnostic = merge_internal(model, 1, [pair], "quaternion_exact")
    with torch.no_grad():
        expected = model.blocks[2](substitute_features(features, [pair], "quaternion_right"))
        actual = compact.blocks[2](features[:, [c*4+u for c in range(4) for u in kept]])
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    assert diagnostic[0]["discarded_energy_fraction"] < 1e-25
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
    projected, _, diagnostics = merge_internal(model, 1, [pair], "full_projected")
    assert diagnostics[0]["discarded_energy_fraction"] > 0
    assert sum(p.numel() for p in compact.parameters()) == sum(p.numel() for p in projected.parameters())
    assert sum(p.numel() for p in compact.parameters()) < sum(p.numel() for p in model.parameters())


def test_spatial_layout_and_padding_offset_counterexample():
    values = np.arange(2*64*3*4).reshape(2, 64, 3, 4).astype(np.float32)
    features = grouped_to_tensor(values)
    restored = features.reshape(2, 4, 3, 8, 8).permute(0, 3, 4, 2, 1).reshape(2, 64, 3, 4)
    np.testing.assert_array_equal(restored.numpy(), values)
    # Constant replacement through zero-padded 3x3 convolution is not a
    # spatially constant offset, so folding it into one bias is not exact.
    output = F.conv2d(torch.ones(1, 1, 5, 5), torch.ones(1, 1, 3, 3), padding=1)
    assert output[0, 0, 0, 0] == 4 and output[0, 0, 2, 2] == 9
