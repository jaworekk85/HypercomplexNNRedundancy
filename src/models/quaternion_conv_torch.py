"""Minimal right-Hamilton Conv2d for component-blocked PyTorch images.

Input: (N,4*input_quaternion_channels,H,W), ordered [all r,all i,all j,all k].
Four real kernel tensors generate a structured real convolution matrix.
This is standard quaternion arithmetic, not a new convolution method.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def hamilton_kernel(kernel):
    wr, wi, wj, wk = kernel.unbind(0)
    return torch.cat((
        torch.cat((wr, -wi, -wj, -wk), dim=1),
        torch.cat((wi, wr, wk, -wj), dim=1),
        torch.cat((wj, -wk, wr, wi), dim=1),
        torch.cat((wk, wj, -wi, wr), dim=1),
    ), dim=0)


class QuaternionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        if min(in_channels, out_channels, kernel_size) < 1:
            raise ValueError("Quaternion channel counts and kernel size must be positive")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.stride, self.padding = stride, padding
        self.kernel = nn.Parameter(torch.empty(4, out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(4 * out_channels)) if bias else None
        nn.init.normal_(self.kernel, std=math.sqrt(2 / (4 * in_channels * kernel_size**2)))

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 4 * self.in_channels:
            raise ValueError("Expected component-blocked (N,4*quaternion_channels,H,W) input")
        return F.conv2d(inputs, hamilton_kernel(self.kernel), self.bias, stride=self.stride, padding=self.padding)
