"""Small controlled ablation: original CNN6 with every BatchNorm removed.

No replacement normalization, convolution bias, initialization adjustment or
learning-rate tuning. Remaining initialized parameters match the original model.
"""
from torch import nn

from src.models.cifar_cnn_torch import CifarCNN


def make_no_bn(config, kind):
    model = CifarCNN(kind, config["quaternion_widths"], config["real_widths"], config["dropout"])
    for block in model.blocks:
        for index in (1, 4):
            if not isinstance(block[index], nn.BatchNorm2d):
                raise ValueError("Unexpected original block layout")
            block[index] = nn.Identity()
    return model
