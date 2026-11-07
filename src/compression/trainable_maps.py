"""Temporary Q map corrections, exactly folded away at deployment.

U = B + sum_v delta_a[k,v] * C[o,v,offset].  C is frozen.  Zero corrections
preserve the compact starting function.  This is a reparameterization of the
same Q family, not a new capacity, algebra or structural-reparameterization claim.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from src.compression.internal_merging import BASIS
from src.models.quaternion_conv_torch import QuaternionConv2d, hamilton_kernel


class MapQuaternionConv2d(QuaternionConv2d):
    def __init__(self, layer, removed_kernel, learn_maps):
        super().__init__(layer.in_channels, layer.out_channels, layer.kernel.shape[-1], layer.stride, layer.padding, layer.bias is not None)
        self.to(layer.kernel)
        with torch.no_grad():
            self.kernel.copy_(layer.kernel)
            if self.bias is not None: self.bias.copy_(layer.bias)
        c = removed_kernel.detach().clone().to(layer.kernel)
        if c.ndim != 5 or c.shape[:2] != layer.kernel.shape[:2] or c.shape[-2:] != layer.kernel.shape[-2:]:
            raise ValueError("Frozen C must match Q components, outputs and kernel offsets")
        self.register_buffer("removed_kernel", c)
        self.register_buffer("product_basis", torch.as_tensor(BASIS.copy(), dtype=c.dtype, device=c.device))
        delta = c.new_zeros((4, layer.in_channels, c.shape[2]))
        if learn_maps: self.map_delta = nn.Parameter(delta)
        else: self.register_buffer("map_delta", delta)

    def effective_kernel(self):
        update = torch.einsum("jci,ikv,jovhw->cokhw", self.product_basis, self.map_delta, self.removed_kernel)
        return self.kernel+update

    def forward(self, inputs):
        return F.conv2d(inputs, hamilton_kernel(self.effective_kernel()), self.bias, self.stride, self.padding)

    @torch.no_grad()
    def folded(self):
        # Constructor initialization must not perturb the next training epoch's RNG.
        with torch.random.fork_rng(devices=[]):
            result = QuaternionConv2d(self.in_channels, self.out_channels, self.kernel.shape[-1], self.stride, self.padding, self.bias is not None).to(self.kernel)
        result.kernel.copy_(self.effective_kernel())
        if result.bias is not None: result.bias.copy_(self.bias)
        return result.train(self.training)


def attach(model, removed_kernel, learn_maps):
    result = deepcopy(model)
    if not isinstance(result.blocks[2][3], QuaternionConv2d): raise ValueError("Expected the existing last Q convolution")
    result.blocks[2][3] = MapQuaternionConv2d(result.blocks[2][3], removed_kernel, learn_maps)
    return result


def deploy(model):
    result = deepcopy(model)
    if not isinstance(result.blocks[2][3], MapQuaternionConv2d): raise ValueError("Missing temporary map layer")
    result.blocks[2][3] = result.blocks[2][3].folded()
    return result
