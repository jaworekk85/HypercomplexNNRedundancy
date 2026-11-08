"""Quaternion input/stem and independently learned PHM residual convolutions.

PHM is established (ZhangEtAl2021; Grassucci2024). No algebra axioms are
imposed on learned bases. Stage widths retain the existing scalar layout.
"""
import math

import torch

from src.models.cifar_resnet import CifarResNet20
from src.models.phm_conv import PHMConv2d


class CifarPHMResNet20(CifarResNet20):
    def __init__(self, quaternion_widths=(8,16,32), classes=10,
                 dimensions=None, hidden_widths=None):
        super().__init__("quaternion", quaternion_widths=quaternion_widths,
                         classes=classes, hidden_widths=hidden_widths)
        dimensions = [4]*18 if dimensions is None else list(dimensions)
        if len(dimensions) != 18:
            raise ValueError("One PHM dimension required for each of 18 residual convolutions")
        self.phm_dimensions = dimensions
        for index, block in enumerate(self.blocks):
            for position, name in enumerate(("conv1","conv2")):
                old = getattr(block,name)
                n = dimensions[2*index+position]
                layer = PHMConv2d(4*old.in_channels,4*old.out_channels,
                                  kernel_size=3,n=n,stride=old.stride,padding=old.padding)
                # Default PHM factors give effective variance 2/(n*I*k^2).
                # Scale the random cores so E[var(W_eff)] = 2/(I*k^2).
                with torch.no_grad():
                    layer.cores.mul_(math.sqrt(n))
                setattr(block,name,layer)
