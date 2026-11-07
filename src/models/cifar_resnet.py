"""CIFAR ResNet20, option-A shortcuts, ordinary or fixed-Hamilton convolutions.

He2016ResNet is the architecture reference. Q widths count quaternion units;
scalar BN/split ReLU and a real readout follow this project's conventions.
"""
import torch
from torch import nn
from torch.nn import functional as F

from src.models.quaternion_conv_torch import QuaternionConv2d


def convolution(kind, inputs, outputs, stride=1):
    if kind == "quaternion": return QuaternionConv2d(inputs, outputs, 3, stride, 1)
    result = nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False)
    nn.init.kaiming_normal_(result.weight, mode="fan_in", nonlinearity="relu")
    return result


class BasicBlock(nn.Module):
    def __init__(self, kind, inputs, outputs, stride=1, hidden=None):
        super().__init__()
        self.components = 4 if kind == "quaternion" else 1
        self.inputs, self.outputs, self.stride = inputs, outputs, stride
        hidden = outputs if hidden is None else hidden
        self.conv1 = convolution(kind, inputs, hidden, stride)
        self.bn1 = nn.BatchNorm2d(self.components*hidden)
        self.conv2 = convolution(kind, hidden, outputs)
        self.bn2 = nn.BatchNorm2d(self.components*outputs)

    def shortcut(self, x):
        x = x[:, :, ::self.stride, ::self.stride]
        if self.outputs == self.inputs: return x
        if self.outputs < self.inputs: raise ValueError("Option-A shortcut cannot reduce width")
        n, _, h, w = x.shape
        padding = self.outputs-self.inputs
        grouped = x.reshape(n, self.components, self.inputs, h, w)
        return F.pad(grouped, (0, 0, 0, 0, padding//2, padding-padding//2)).reshape(n, self.components*self.outputs, h, w)

    def forward(self, x):
        z = F.relu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        return F.relu(z+self.shortcut(x))


class CifarResNet20(nn.Module):
    def __init__(self, kind="real", real_widths=(16, 32, 64), quaternion_widths=(8, 16, 32), classes=10, hidden_widths=None):
        super().__init__()
        if kind not in ("real", "quaternion"): raise ValueError(kind)
        self.kind = kind
        widths = quaternion_widths if kind == "quaternion" else real_widths
        if len(widths) != 3 or min(widths) < 1: raise ValueError("Three positive stage widths required")
        if hidden_widths is not None and (len(hidden_widths) != 9 or min(hidden_widths) < 1): raise ValueError("Nine positive internal widths required")
        components = 4 if kind == "quaternion" else 1
        self.stem = convolution(kind, 1 if kind == "quaternion" else 3, widths[0])
        self.stem_bn = nn.BatchNorm2d(components*widths[0])
        blocks, inputs = [], widths[0]
        for stage, outputs in enumerate(widths):
            for index in range(3):
                blocks.append(BasicBlock(kind, inputs, outputs, 2 if stage and index == 0 else 1,
                                         None if hidden_widths is None else hidden_widths[len(blocks)]))
                inputs = outputs
        self.blocks = nn.ModuleList(blocks)
        self.readout = nn.Linear(components*widths[-1], classes)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward_features(self, rgb):
        x = torch.cat((torch.zeros_like(rgb[:, :1]), rgb), dim=1) if self.kind == "quaternion" else rgb
        x = F.relu(self.stem_bn(self.stem(x)))
        for block in self.blocks: x = block(x)
        return x

    def forward(self, rgb):
        return self.readout(self.forward_features(rgb).mean(dim=(2, 3)))
