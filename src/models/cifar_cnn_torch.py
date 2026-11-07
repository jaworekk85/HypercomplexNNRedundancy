"""A small conventional six-convolution CNN and its quaternion counterpart."""

import torch
from torch import nn

from src.models.quaternion_conv_torch import QuaternionConv2d


class CifarCNN(nn.Module):
    def __init__(self, kind="quaternion", q_widths=(8, 16, 32), real_widths=(16, 32, 64), dropout=(0.1, 0.2, 0.3)):
        super().__init__()
        if kind not in ("quaternion", "real"):
            raise ValueError("kind must be quaternion or real")
        self.kind = kind
        widths = q_widths if kind == "quaternion" else real_widths
        if len(widths) != 3 or len(dropout) != 3:
            raise ValueError("This small CNN has three convolution blocks")
        in_channels = 1 if kind == "quaternion" else 3
        blocks = []
        for width, probability in zip(widths, dropout):
            layers = []
            for _ in range(2):
                if kind == "quaternion":
                    conv = QuaternionConv2d(in_channels, width)
                    scalar_channels = 4 * width
                else:
                    conv = nn.Conv2d(in_channels, width, 3, padding=1, bias=False)
                    nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
                    scalar_channels = width
                layers.extend((conv, nn.BatchNorm2d(scalar_channels), nn.ReLU(inplace=False)))
                in_channels = width
            layers.extend((nn.MaxPool2d(2), nn.Dropout2d(probability)))
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(blocks)
        self.readout = nn.Linear(scalar_channels, 10)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward_features(self, rgb):
        # Same normalized RGB information for both models; alpha is exactly zero.
        features = torch.cat((torch.zeros_like(rgb[:, :1]), rgb), dim=1) if self.kind == "quaternion" else rgb
        for block in self.blocks:
            features = block(features)
        return features

    def forward(self, rgb):
        return self.readout(self.forward_features(rgb).mean(dim=(2, 3)))
