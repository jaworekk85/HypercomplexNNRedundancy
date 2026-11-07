"""Conventional 4x4 circulant weight sharing: a control, not a new algebra.

Prior structured-weight context: Ding2017CirCNN. We expand the small blocks
for ordinary Conv2d; no FFT speedup or CirCNN reproduction is claimed.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models.cifar_cnn_torch import CifarCNN


# C(a)[output,input] = a[(output-input) mod 4]; coefficients are the first column.
CIRCULANT_BASIS=np.stack([np.fromfunction(lambda o,i: (o-i)%4==c,(4,4)).astype(np.float64) for c in range(4)])


def circulant_kernel(kernel):
    return torch.cat([torch.cat([kernel[(o-i)%4] for i in range(4)],dim=1) for o in range(4)],dim=0)


class CirculantConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False,initialize=True):
        super().__init__()
        if min(in_channels,out_channels,kernel_size)<1: raise ValueError("Positive group dimensions required")
        self.in_channels,self.out_channels=in_channels,out_channels
        self.stride,self.padding=stride,padding
        self.kernel=nn.Parameter(torch.empty(4,out_channels,in_channels,kernel_size,kernel_size))
        self.bias=nn.Parameter(torch.zeros(4*out_channels)) if bias else None
        if initialize: nn.init.normal_(self.kernel,std=math.sqrt(2/(4*in_channels*kernel_size**2)))

    def forward(self,inputs):
        if inputs.ndim!=4 or inputs.shape[1]!=4*self.in_channels: raise ValueError("Expected component-blocked four-component groups")
        return F.conv2d(inputs,circulant_kernel(self.kernel),self.bias,stride=self.stride,padding=self.padding)


class CirculantCNN(CifarCNN):
    def __init__(self,config):
        # Same seeded coefficient tensors/BN/readout and RNG consumption as Q CNN6.
        super().__init__("quaternion",config["quaternion_widths"],config["real_widths"],config["dropout"])
        self.kind="circulant"
        for block in self.blocks:
            for index in (0,3):
                old=block[index]
                new=CirculantConv2d(old.in_channels,old.out_channels,old.kernel.shape[-1],old.stride,old.padding,initialize=False)
                with torch.no_grad(): new.kernel.copy_(old.kernel)
                block[index]=new

    def forward_features(self,rgb):
        # Identical four-channel INPUT INFORMATION; no quaternion interpretation.
        features=torch.cat((torch.zeros_like(rgb[:,:1]),rgb),dim=1)
        for block in self.blocks: features=block(features)
        return features


def make_circulant(config,kind="circulant"):
    if kind!="circulant": raise ValueError("This factory constructs only the circulant control")
    return CirculantCNN(config)
