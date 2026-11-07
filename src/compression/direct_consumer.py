"""Direct fixed-size quaternion consumer regression, not a new LS method.

Targets are pre-BN consumer outputs. All other parameters/buffers stay fixed.
Ridge is centered at the existing map-then-fold kernel, making it a feasible
zero-penalty reference. No labels or gradients enter this regression.
"""
from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

from src.compression.internal_merging import BASIS
from src.compression import multilayer as old
from src.compression.quaternion_channels import component_indices
from src.models.quaternion_conv_torch import QuaternionConv2d


def q_normal_equations(raw_gram, raw_cross, outputs):
    """Raw patch order [input component, unit, kernel offset]; Y=[component,unit]."""
    features = len(raw_gram)
    if features % 4 or raw_cross.shape != (features, 4*outputs):
        raise ValueError("Invalid component-blocked moment shapes")
    j = features // 4
    gram = np.einsum("cai,daj,ikjl->ckdl", BASIS, BASIS, raw_gram.reshape(4, j, 4, j)).reshape(features, features)
    rhs = np.einsum("cai,ikao->cko", BASIS, raw_cross.reshape(4, j, 4, outputs)).reshape(features, outputs)
    return (gram+gram.T)/2, rhs


def theta(kernel):
    return np.asarray(kernel, dtype=np.float64).transpose(0, 2, 3, 4, 1).reshape(-1, kernel.shape[1])


def kernel_of(values, shape):
    return values.reshape(shape[0], shape[2], shape[3], shape[4], shape[1]).transpose(0, 4, 1, 2, 3)


def sse(gram, rhs, target_squares, values):
    result = float(target_squares-2*np.sum(values*rhs)+np.sum(values*(gram@values)))
    if result < -1e-8*max(1., target_squares):
        raise AssertionError("Negative reconstruction SSE beyond rounding tolerance")
    return max(0., result)


def solve_centered(gram, rhs, target_squares, prior, alpha):
    if alpha <= 0: raise ValueError("Positive prespecified ridge required")
    penalty = alpha*float(np.mean(np.diag(gram)))
    if not np.isfinite(penalty) or penalty <= 0: raise ValueError("Degenerate/nonfinite patch energy")
    system = gram+penalty*np.eye(len(gram))
    residual_rhs = rhs-gram@prior
    delta = np.linalg.solve(system, residual_rhs)
    result = prior+delta
    normal_residual = float(np.linalg.norm(system@delta-residual_rhs)/max(1., np.linalg.norm(residual_rhs)))
    before = sse(gram, rhs, target_squares, prior)
    after = sse(gram, rhs, target_squares, result)
    objective = after+penalty*float(np.square(delta).sum())
    if not np.isfinite(result).all() or normal_residual > 1e-8:
        raise AssertionError("Direct Q ridge solve failed")
    if objective > before+1e-7*max(1., before):
        raise AssertionError("Centered fit is worse than its feasible prior")
    return result, {"lambda": penalty, "normal_equation_relative_residual": normal_residual,
                    "fit_prior_sse": before, "fit_direct_sse": after,
                    "fit_direct_regularized_objective": objective,
                    "coefficient_update_l2": float(np.linalg.norm(delta))}


@torch.no_grad()
def moments(model, site, kept, images, batch_size=128):
    """Collect paired kept-input patches and full consumer targets, all eval mode."""
    consumer = old.consumer(model, site)
    if not isinstance(consumer, QuaternionConv2d) or consumer.bias is not None:
        raise ValueError("Only bias-free quaternion convolution consumers are fitted")
    model.eval()
    channels = component_indices(kept, consumer.in_channels)
    state = {"gram": None, "cross": None, "squares": None, "rows": 0, "batches": 0}

    def collect(module, inputs, output):
        patches = F.unfold(inputs[0][:, channels], consumer.kernel.shape[-2:], padding=consumer.padding, stride=consumer.stride)
        x = patches.transpose(1, 2).reshape(-1, patches.shape[1]).double()
        y = output.flatten(2).transpose(1, 2).reshape(-1, output.shape[1]).double()
        if len(x) != len(y): raise AssertionError("Patch/target spatial mismatch")
        gram, cross, squares = x.T@x, x.T@y, y.square().sum()
        if state["gram"] is None:
            state.update(gram=gram, cross=cross, squares=squares)
        else:
            state["gram"].add_(gram); state["cross"].add_(cross); state["squares"].add_(squares)
        state["rows"] += len(x); state["batches"] += 1
    handle = consumer.register_forward_hook(collect)
    try:
        for start in range(0, len(images), batch_size): model(images[start:start+batch_size])
    finally:
        handle.remove()
    if state["rows"] == 0: raise ValueError("Empty fit/check images")
    gram, rhs = q_normal_equations(state["gram"].cpu().numpy(), state["cross"].cpu().numpy(), consumer.out_channels)
    return {"gram": gram, "rhs": rhs, "target_squares": float(state["squares"]),
            "rows": state["rows"], "outputs": consumer.out_channels, "images": len(images), "batches": state["batches"]}


def error(data, values):
    return sse(data["gram"], data["rhs"], data["target_squares"], values)/(data["rows"]*4*data["outputs"])


@torch.no_grad()
def refit_consumer(reference, folded, site, kept, fit_images, check_images, alpha=.001, batch_size=128):
    source, target = old.consumer(reference, site), old.consumer(folded, site)
    if not isinstance(target, QuaternionConv2d): raise ValueError("Consumer must remain quaternion")
    before_state = {n: v.detach().clone() for n, v in folded.state_dict().items()}
    fit, check = moments(reference, site, kept, fit_images, batch_size), moments(reference, site, kept, check_images, batch_size)
    prior = theta(target.kernel.detach().cpu().numpy())
    values, audit = solve_centered(fit["gram"], fit["rhs"], fit["target_squares"], prior, alpha)
    result = deepcopy(folded)
    new = old.consumer(result, site)
    new.kernel.copy_(torch.as_tensor(kernel_of(values, tuple(new.kernel.shape)), dtype=new.kernel.dtype, device=new.kernel.device))
    parameter_name = next(name for name, p in result.named_parameters() if p is new.kernel)
    for name, value in result.state_dict().items():
        if name != parameter_name and not torch.equal(value, before_state[name]):
            raise AssertionError(f"Direct fit changed another parameter/buffer: {name}")
    rounded = theta(new.kernel.detach().cpu().numpy())
    audit.update(fit_images=len(fit_images), check_images=len(check_images), fitted_parameter=parameter_name,
                 fit_prior_mse=error(fit, prior), fit_direct_mse=error(fit, rounded),
                 check_prior_mse=error(check, prior), check_direct_mse=error(check, rounded),
                 all_other_parameters_and_buffers_unchanged=True, labels_used=False,
                 fit_spatial_rows=fit["rows"], check_spatial_rows=check["rows"], output_groups=fit["outputs"],
                 target="Pre-BN output of the current reference consumer before this local removal",
                 estimator="Float64 moments and centered ridge; stored kernel rounded to model dtype")
    if audit["fit_direct_mse"] > audit["fit_prior_mse"]+1e-6*max(1., audit["fit_prior_mse"]):
        raise AssertionError("Rounding spoiled local training-error improvement")
    arrays = {"fit_gram": fit["gram"], "fit_rhs": fit["rhs"], "fit_target_squares": fit["target_squares"],
              "check_gram": check["gram"], "check_rhs": check["rhs"], "check_target_squares": check["target_squares"],
              "prior_theta": prior, "direct_theta_float64": values, "stored_theta": rounded}
    return result.eval(), audit, arrays
