"""Eight-convolution/four-pooling-block comparator; no architectural novelty."""

import torch
from torch import nn

from src.models.cifar_cnn_torch import CifarCNN
from src.models.quaternion_conv_torch import QuaternionConv2d


class FourBlockCifarCNN(CifarCNN):
    def __init__(self, widths=(8,16,24,32), dropout=(.1,.2,.3,.3)):
        nn.Module.__init__(self)
        if len(widths)!=4 or len(dropout)!=4:
            raise ValueError("The comparator has exactly four blocks")
        self.kind="quaternion"
        blocks=[]
        incoming=1
        for width,probability in zip(widths,dropout):
            layers=[]
            for _ in range(2):
                layers.extend((QuaternionConv2d(incoming,width),nn.BatchNorm2d(4*width),nn.ReLU(inplace=False)))
                incoming=width
            layers.extend((nn.MaxPool2d(2),nn.Dropout2d(probability)))
            blocks.append(nn.Sequential(*layers))
        self.blocks=nn.ModuleList(blocks)
        self.readout=nn.Linear(4*widths[-1],10)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)


def make_fourblock(config,kind="quaternion"):
    if kind!="quaternion":
        raise ValueError("This controlled extension trains the quaternion comparator only")
    return FourBlockCifarCNN(config["quaternion_widths"],config["dropout"])
