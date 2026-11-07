from copy import deepcopy

import numpy as np
import pytest
import torch

from src.compression.direct_consumer import q_normal_equations, solve_centered, moments, theta, kernel_of, refit_consumer
from src.compression.internal_merging import BASIS, expand_right
from src.compression import multilayer as old
from src.compression.structure_cost import compact, make_model
from src.models.quaternion_conv_torch import hamilton_kernel

torch.set_num_threads(2)


def test_normal_equations_equal_explicit_four_component_design_and_augmented_ridge():
    rng = np.random.default_rng(25)
    x, y = rng.normal(size=(83, 12)), rng.normal(size=(83, 8))
    design = np.einsum("cai,nij->nacj", BASIS, x.reshape(83, 4, 3)).reshape(332, 12)
    targets = y.reshape(83, 4, 2).reshape(332, 2)
    gram, rhs = q_normal_equations(x.T@x, x.T@y, 2)
    np.testing.assert_allclose(gram, design.T@design, atol=1e-12)
    np.testing.assert_allclose(rhs, design.T@targets, atol=1e-12)
    prior = rng.normal(size=(12, 2))
    fitted, audit = solve_centered(gram, rhs, np.square(y).sum(), prior, .001)
    lam = audit["lambda"]
    expected = np.linalg.lstsq(np.vstack([design, np.sqrt(lam)*np.eye(12)]),
                               np.vstack([targets, np.sqrt(lam)*prior]), rcond=None)[0]
    np.testing.assert_allclose(fitted, expected, atol=1e-12)
    assert audit["fit_direct_regularized_objective"] <= audit["fit_prior_sse"]


@pytest.mark.parametrize("norm", ["bn", "none"])
def test_patch_moments_match_actual_spatial_convolution_and_only_target_kernel_changes(norm):
    torch.manual_seed(71)
    config = {"quaternion_widths": [3, 4, 5], "real_widths": [6, 8, 10], "dropout": [0., 0., 0.]}
    model = make_model(config, "cifar", norm, "quaternion").double().eval()
    site, kept = (0, 0), [0, 2]
    fit, check = torch.randn(13, 3, 16, 16, dtype=torch.float64), torch.randn(7, 3, 16, 16, dtype=torch.float64)
    maps = [{"victim": 1, "kept": kept, "matrices": expand_right(np.ones((2, 4))*.1).tolist()}]
    folded = compact(model, site, [1], maps)
    result, audit, arrays = refit_consumer(model, folded, site, kept, fit, check, .001, 4)
    assert audit["all_other_parameters_and_buffers_unchanged"]
    assert audit["fit_direct_mse"] <= audit["fit_prior_mse"]+1e-12
    observed = []
    consumer = old.consumer(model, site)
    new = old.consumer(result, site)
    def collect(module, inputs, output):
        h = inputs[0][:, [0, 2, 3, 5, 6, 8, 9, 11]]
        observed.append((new(h)-output).square().flatten())
    handle = consumer.register_forward_hook(collect)
    with torch.no_grad(): model(check)
    handle.remove()
    assert abs(float(torch.cat(observed).mean())-audit["check_direct_mse"]) < 1e-12
    restored = make_model(config, "cifar", norm, "quaternion").double().eval()
    # Intermediate widths are irregular until all sites are compacted; exact state copy is used here.
    restored = deepcopy(result)
    restored.load_state_dict(result.state_dict())
    torch.testing.assert_close(restored(check), result(check), atol=0, rtol=0)


def test_no_removal_keeps_exact_teacher_kernel_and_tensor_layout():
    torch.manual_seed(3)
    config = {"quaternion_widths": [2, 3, 4], "real_widths": [4, 6, 8], "dropout": [0., 0., 0.]}
    model = make_model(config, "cifar", "none", "quaternion").double().eval()
    fit = torch.randn(10, 3, 16, 16, dtype=torch.float64)
    result, audit, _ = refit_consumer(model, model, (0, 0), [0, 1], fit, fit[:4], .001, 5)
    old_kernel, new_kernel = old.consumer(model, (0, 0)).kernel, old.consumer(result, (0, 0)).kernel
    torch.testing.assert_close(new_kernel, old_kernel, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(kernel_of(theta(old_kernel.detach().numpy()), tuple(old_kernel.shape)), old_kernel.detach().numpy())
    assert audit["check_direct_mse"] < 1e-12
