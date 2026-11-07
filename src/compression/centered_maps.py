"""Calibrate shared Q maps and per-victim Q centers on cached output moments.

U=U0+(A-A0)C-A*mu. Joint fitting first solves the linear (deltaA,d)
space, d=A*mu, then recovers mu. Only an accurately recovered solution is
accepted as the original bilinear model. No novelty/optimizer claim.
"""
import torch
from torch import nn

from src.compression.internal_merging import BASIS, expand_right


def product(a, b):
    r, i, j, k = a.unbind(0)
    s, x, y, z = b.unbind(0)
    return torch.stack((r*s-i*x-j*y-k*z, r*x+i*s+j*z-k*y,
                        r*y-i*z+j*s+k*x, r*z+i*y-j*x+k*s))


def left_operator(a):
    """Component-blocked vec(d)=L(A)vec(mu), d[k]=sum_v A[k,v]*mu[v]."""
    r, i, j, k = a.unbind(0)
    rows = [[r, -i, -j, -k], [i, r, -k, j], [j, k, r, -i], [k, -j, i, r]]
    return torch.stack([torch.stack(row) for row in rows]).permute(0, 2, 1, 3).reshape(4*a.shape[1], 4*a.shape[2])


class CenteredKernel(nn.Module):
    """Tracked A/mu factors used for technical gradient/folding checks."""
    def __init__(self, prior, c, a0):
        super().__init__()
        self.register_buffer("prior", prior.clone())
        self.register_buffer("c", c.clone())
        self.register_buffer("a0", a0.clone())
        self.register_buffer("basis", torch.as_tensor(BASIS.copy()).to(prior))
        self.a = nn.Parameter(a0.clone())
        self.mu = nn.Parameter(a0.new_zeros(4, a0.shape[2]))

    def forward(self):
        update = torch.einsum("jci,ikv,jovhw->cokhw", self.basis, self.a-self.a0, self.c)
        d = product(self.a, self.mu[:, None, :]).sum(2)
        return self.prior + update - d[:, None, :, None, None]


def operators(c, kept):
    c_np = c.detach().cpu().numpy()
    _, outputs, removed, kh, kw = c_np.shape
    matrices = expand_right(c_np.transpose(1, 3, 4, 2, 0))
    block = torch.from_numpy(matrices.reshape(outputs, kh*kw, removed, 4, 4).transpose(0, 3, 1, 4, 2)).to(c)
    ta = c.new_zeros(outputs, 4, kept, kh*kw, 4, kept, removed)
    td = c.new_zeros(outputs, 4, kept, kh*kw, 4, kept)
    for k in range(kept):
        ta[:, :, k, :, :, k, :] = block
        for component in range(4): td[:, component, k, :, component, k] = -1
    return ta.reshape(outputs, 4*kept*kh*kw, 4*kept*removed), td.reshape(outputs, 4*kept*kh*kw, 4*kept)


def psd_solve(hessian, rhs, cutoff):
    values, vectors = torch.linalg.eigh((hessian+hessian.T)/2)
    threshold = float(values[-1])*cutoff
    if float(values[0]) < -max(1e-8, threshold): raise AssertionError("Indefinite calibration Hessian")
    retained = values > threshold
    solution = vectors[:, retained] @ ((vectors[:, retained].T@rhs)/values[retained])
    residual = float((hessian@solution-rhs).norm()/rhs.norm().clamp_min(1))
    if not torch.isfinite(solution).all() or residual > 1e-6: raise AssertionError("Calibration normal equations failed")
    return solution, {"rank": int(retained.sum()), "variables": len(rhs), "eigen_cutoff": threshold, "relative_normal_residual": residual}


def solve_cases(gram, rhs, prior, c, a0, penalty, cutoff=1e-10):
    """All arms minimize the SAME kernel-centered ridge objective.

    theta shape [4*K*kernel_area,O]. Fixed bases B,C and old masks; no labels.
    Joint solution is globally optimal in the relaxed linear space. If its
    recovered A*mu reproduces d, it also attains this bound in the bilinear space.
    """
    ta, td = operators(c, a0.shape[1])
    t = torch.cat((ta, td), dim=-1)
    n = ta.shape[-1]
    h = gram + penalty*torch.eye(len(gram), dtype=gram.dtype, device=gram.device)
    residual = rhs-gram@prior
    system = gram.new_zeros(t.shape[-1], t.shape[-1])
    vector = gram.new_zeros(t.shape[-1])
    for o in range(len(t)):
        system.add_(t[o].T@h@t[o])
        vector.add_(t[o].T@residual[:, o])
    da, audit_a = psd_solve(system[:n, :n], vector[:n], cutoff)
    l0 = left_operator(a0)
    mu, audit_mu = psd_solve(l0.T@system[n:, n:]@l0, l0.T@vector[n:], cutoff)
    joint, audit_joint = psd_solve(system, vector, cutoff)
    a = a0+joint[:n].reshape_as(a0)
    d = joint[n:]
    l = left_operator(a)
    sv = torch.linalg.svdvals(l)
    recovered = torch.linalg.lstsq(l, d[:, None], rcond=cutoff, driver="gelsd")
    joint_mu = recovered.solution[:, 0]
    recovery_error = float((l@joint_mu-d).abs().max())
    if recovery_error > 1e-8*max(1., float(d.abs().max())):
        raise AssertionError(f"Joint linear relaxation cannot be recovered as A*mu: {recovery_error}")
    audit_joint.update(mu_recovery_max_abs_error=recovery_error, recovery_rank=int(recovered.rank),
                       recovery_condition=float(sv[0]/sv[-1]) if sv[-1] > 0 else None)
    zero_a, zero_mu = torch.zeros_like(a0), torch.zeros_like(mu)
    definitions = [("mapfold", zero_a, zero_mu, {}), ("a_only", da.reshape_as(a0), zero_mu, audit_a),
                   ("mu_only", zero_a, mu, audit_mu), ("a_mu", a-a0, joint_mu, audit_joint)]
    results = {}
    for name, delta_a, center, audit in definitions:
        current_a = a0+delta_a
        current_d = left_operator(current_a)@center
        coefficients = torch.cat((delta_a.reshape(-1), current_d))
        theta = prior+torch.einsum("opv,v->po", t, coefficients)
        results[name] = {"theta": theta, "a": current_a, "mu": center.reshape(4, -1), "d": current_d.reshape(4, -1), "audit": audit}
    return results
