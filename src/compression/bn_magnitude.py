"""A fixed Conv+BN magnitude reference, grouped by whole quaternion outputs."""

import torch

from src.models.quaternion_conv_torch import QuaternionConv2d


@torch.no_grad()
def bn_group_scores(conv, bn):
    """L2 of four output rows in the BN-folded real Hamilton kernel.

    Every component row has the same pre-BN squared norm for a fixed
    quaternion output. Scale each row by gamma/sqrt(running_var+eps),
    then sum all four row energies. Bias terms are deliberately excluded.
    Evaluate on CPU float64 so policy ranking is device independent.
    """
    if not isinstance(conv, QuaternionConv2d) or bn.num_features != 4 * conv.out_channels:
        raise ValueError("Expected matching quaternion convolution and component BN")
    if bn.training or not bn.affine or not bn.track_running_stats:
        raise ValueError("Use evaluation-mode affine BN with saved running statistics")
    kernel = conv.kernel.detach().cpu().double()
    gamma = bn.weight.detach().cpu().double()
    variance = bn.running_var.detach().cpu().double()
    if not torch.isfinite(variance).all() or not torch.all(variance + bn.eps > 0):
        raise ValueError("Invalid running variance")
    scale = gamma * torch.rsqrt(variance + bn.eps)
    raw = kernel.square().sum(dim=(0, 2, 3, 4)).sqrt()
    folded = raw * scale.reshape(4, conv.out_channels).square().sum(0).sqrt()
    if not torch.isfinite(folded).all():
        raise ValueError("Nonfinite folded magnitude")
    return {"raw_kernel_l2": raw.tolist(), "bn_component_scales": scale.reshape(4, conv.out_channels).T.tolist(),
            "bn_folded_group_l2": folded.tolist()}


def bn_removal_order(scores):
    return sorted(range(len(scores)), key=lambda unit: (scores[unit], unit))
