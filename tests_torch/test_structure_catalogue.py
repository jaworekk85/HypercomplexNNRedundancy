from copy import deepcopy
import itertools

import pytest
import torch
from torch.nn import functional as F

from src.compression.internal_merging import BASIS
from src.compression.structure_catalogue import (architecture, candidates, collect, mutable_count,
    narrow_producer, parameter_count, restore, select)
from src.models.cifar_resnet import CifarResNet20
from src.models.phm_conv import PHMConv2d
from src.models.quaternion_conv_torch import hamilton_kernel


@pytest.mark.parametrize("n", [2, 4, 8])
def test_phm_svd_optimum_actual_storage_and_all_factor_updates(n):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    weight = torch.randn(16, 16, 3, 3, dtype=torch.float64)
    model = PHMConv2d.from_kernel(weight, n)
    assert parameter_count(model) == n**3+16*16*9//n
    arranged = weight.reshape(n, 16//n, n, 16//n, 3, 3).permute(0, 2, 1, 3, 4, 5).reshape(n*n, -1)
    optimum = torch.linalg.svdvals(arranged)[n:].square().sum()
    torch.testing.assert_close((model.effective_kernel()-weight).square().sum(), optimum, atol=1e-10, rtol=1e-10)
    x = torch.randn(3, 16, 8, 8, dtype=torch.float64)
    # Independent block assembly, including spatial sharing and scalar-channel order.
    literal = torch.zeros_like(weight)
    for a in range(n):
        for b in range(n):
            for j in range(n):
                literal[a*(16//n):(a+1)*(16//n), b*(16//n):(b+1)*(16//n)] += model.basis[j, a, b]*model.cores[j]
    torch.testing.assert_close(model(x), F.conv2d(x, literal, padding=1), atol=1e-11, rtol=1e-11)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    model(x).square().mean().backward()
    for p in model.parameters(): assert p.grad is not None and torch.isfinite(p.grad).all() and bool((p.grad != 0).all())
    torch.optim.SGD(model.parameters(), lr=.001).step()
    for name, p in model.named_parameters(): assert bool((p != before[name]).all())


def test_hamilton_is_exactly_contained_and_invalid_partition_rejected():
    torch.manual_seed(8)
    q = torch.randn(4, 3, 2, 3, 3, dtype=torch.float64)
    model = PHMConv2d(8, 12, n=4).double()
    with torch.no_grad(): model.basis.copy_(torch.from_numpy(BASIS)); model.cores.copy_(q)
    torch.testing.assert_close(model.effective_kernel(), hamilton_kernel(q), atol=0, rtol=0)
    fitted = PHMConv2d.from_kernel(hamilton_kernel(q), 4)
    torch.testing.assert_close(fitted.effective_kernel(), hamilton_kernel(q), atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError): PHMConv2d(8, 12, n=3)


def test_budget_selector_matches_exhaustive_small_problem_and_rejects_infeasible():
    catalogues = [[{"id": "a", "cost": 2, "error": .3}, {"id": "b", "cost": 4, "error": .1}],
                  [{"id": "c", "cost": 1, "error": .4}, {"id": "d", "cost": 3, "error": .2}]]
    actual = select(catalogues, 5)
    feasible = [(sum(c["error"] for c in choice), sum(c["cost"] for c in choice), [c["id"] for c in choice])
                for choice in itertools.product(*catalogues) if sum(c["cost"] for c in choice) <= 5]
    assert (actual["proxy_error"], actual["cost"], actual["choices"]) == min(feasible)
    with pytest.raises(ValueError): select(catalogues, 2)


def test_real_candidate_fit_physical_widths_q_residuals_and_mixed_checkpoint_reload():
    torch.set_num_threads(2)
    torch.manual_seed(12)
    config = {"real_widths": [4, 8, 16], "quaternion_widths": [2, 4, 8], "classes": 10}
    source = CifarResNet20("quaternion", **config).eval()
    original_state = {n: v.clone() for n, v in source.state_dict().items()}
    images = torch.randn(8, 3, 8, 8)
    catalogue = []
    for index in (0, 8):
        pack = collect(source, index, images, 4)
        width = source.blocks[index].conv1.out_channels
        options, _ = candidates(source.blocks[index], pack, [max(1, width//2)], [2, 4, 8], .001)
        catalogue.append(options)
        for item in options:
            assert item["cost"] == mutable_count(item["block"])
            kept = item["kept"]
            scalar = [c*width+k for c in range(4) for k in kept]
            with torch.no_grad(): direct_error = float((item["block"].conv2(pack[0][:, scalar])-pack[1]).double().square().mean())
            assert abs(direct_error-item["fit_mse"]) < 1e-5
        identity = narrow_producer(source.blocks[index], list(range(width)))
        assert identity.conv1.out_channels == width
        for name, value in identity.state_dict().items(): torch.testing.assert_close(value, source.blocks[index].state_dict()[name], atol=0, rtol=0)
    mixed = deepcopy(source)
    chosen = [next(c for c in catalogue[0] if c["family"] == "phm" and c["n"] == 4),
              next(c for c in catalogue[1] if c["family"] == "real")]
    for index, item in zip((0, 8), chosen): mixed.blocks[index] = deepcopy(item["block"])
    expected_count = parameter_count(source)-sum(mutable_count(source.blocks[i]) for i in (0, 8))+sum(c["cost"] for c in chosen)
    assert parameter_count(mixed) == expected_count
    restored = restore(architecture(mixed, config)).eval()
    restored.load_state_dict(mixed.state_dict())
    torch.testing.assert_close(mixed(images), restored(images), atol=0, rtol=0)
    assert all(torch.equal(v, original_state[n]) for n, v in source.state_dict().items())
