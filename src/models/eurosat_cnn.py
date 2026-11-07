"""The CIFAR CNN6 pair adapted to four-band EuroSAT input; no architectural novelty.

Quaternion: the input is already one full quaternion channel [NIR, R, G, B] in component-blocked
order, so no zero real part is padded. Real: the first convolution takes four input channels.
"""

from torch import nn

from src.models.cifar_cnn_torch import CifarCNN


class FourBandCNN(CifarCNN):
    def __init__(self, kind="quaternion", q_widths=(8, 16, 32), real_widths=(16, 32, 64), dropout=(0.1, 0.2, 0.3)):
        super().__init__(kind, q_widths, real_widths, dropout)
        if kind == "real":
            conv = nn.Conv2d(4, real_widths[0], 3, padding=1, bias=False)
            nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
            self.blocks[0][0] = conv

    def forward_features(self, bands):
        if bands.ndim != 4 or bands.shape[1] != 4:
            raise ValueError("Expected (N,4,H,W) [NIR,R,G,B] input")
        features = bands
        for block in self.blocks:
            features = block(features)
        return features


def make_fourband(config, kind):
    return FourBandCNN(kind, config["quaternion_widths"], config["real_widths"], config["dropout"])
