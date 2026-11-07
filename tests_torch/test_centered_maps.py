import torch
from torch.nn import functional as F

from src.compression.centered_maps import CenteredKernel, left_operator, operators, product, solve_cases
from src.models.quaternion_conv_torch import QuaternionConv2d, hamilton_kernel


def test_literal_product_tracking_gradient_update_and_q_fold():
    torch.set_num_threads(2)
    torch.manual_seed(44)
    prior = torch.randn(4, 3, 2, 3, 3, dtype=torch.float64)
    c = torch.randn(4, 3, 2, 3, 3, dtype=torch.float64)
    a0 = torch.randn(4, 2, 2, dtype=torch.float64)
    module = CenteredKernel(prior, c, a0)
    assert set(dict(module.named_parameters())) == {"a", "mu"}
    assert torch.equal(module(), prior)
    before = {n: p.detach().clone() for n, p in module.named_parameters()}
    module().square().mean().backward()
    for p in module.parameters(): assert (p.grad.flatten(1).norm(dim=1) > 0).all()
    torch.optim.Adam(module.parameters(), lr=.001).step()
    for n, p in module.named_parameters(): assert (p-before[n]).abs().max() > 0
    assert torch.equal(module.prior, prior) and torch.equal(module.c, c)
    expected = prior.clone()
    for k in range(2):
        for v in range(2):
            for o in range(3):
                expected[:, o, k] += product((module.a-module.a0)[:, k, v, None, None], c[:, o, v])
                expected[:, o, k] -= product(module.a[:, k, v], module.mu[:, v])[:, None, None]
    torch.testing.assert_close(module(), expected, atol=1e-12, rtol=1e-12)
    layer = QuaternionConv2d(2, 3).double()
    with torch.no_grad(): layer.kernel.copy_(module())
    x = torch.randn(5, 8, 7, 7, dtype=torch.float64)
    torch.testing.assert_close(layer(x), F.conv2d(x, hamilton_kernel(module()), padding=1), atol=1e-12, rtol=1e-12)
    d = product(module.a, module.mu[:, None, :]).sum(2)
    torch.testing.assert_close(left_operator(module.a)@module.mu.reshape(-1), d.reshape(-1))
    ta, td = operators(c, 2)
    theta = lambda w: w.permute(0, 2, 3, 4, 1).reshape(-1, 3)
    coefficients = torch.cat(((module.a-module.a0).reshape(-1), d.reshape(-1)))
    torch.testing.assert_close(theta(module())-theta(prior), torch.einsum("opv,v->po", torch.cat((ta, td), -1), coefficients), atol=1e-12, rtol=1e-12)


def test_joint_recovery_and_nested_common_objective():
    torch.manual_seed(32)
    k, v, o, spatial = 2, 2, 3, 9
    p = 4*k*spatial
    c = torch.randn(4, o, v, 3, 3, dtype=torch.float64)
    a0 = torch.randn(4, k, v, dtype=torch.float64)
    prior = torch.randn(p, o, dtype=torch.float64)
    x = torch.randn(200, p, dtype=torch.float64)
    target = torch.randn(200, o, dtype=torch.float64)
    gram, rhs = x.T@x, x.T@target
    penalty = .001*float(gram.diag().mean())
    results = solve_cases(gram, rhs, prior, c, a0, penalty)
    objective = lambda w: float((x@w-target).square().sum()+penalty*(w-prior).square().sum())
    values = {n: objective(r["theta"]) for n, r in results.items()}
    assert values["a_mu"] <= min(values["a_only"], values["mu_only"]) + 1e-7
    assert max(values["a_only"], values["mu_only"]) <= values["mapfold"] + 1e-7
    kernel = prior.reshape(4, k, 3, 3, o).permute(0, 4, 1, 2, 3)
    for result in results.values():
        model = CenteredKernel(kernel, c, a0)
        with torch.no_grad():
            model.a.copy_(result["a"])
            model.mu.copy_(result["mu"])
        actual = model().permute(0, 2, 3, 4, 1).reshape(p, o)
        torch.testing.assert_close(actual, result["theta"], atol=1e-10, rtol=1e-10)
    assert results["a_mu"]["audit"]["mu_recovery_max_abs_error"] < 1e-10
