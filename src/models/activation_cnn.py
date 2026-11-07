"""Parameter-free activation controls for the unchanged quaternion CNN6.

MagnitudeTanh is Eq. 21 in PoppelbaumSchwung2024Activations, references.bib.
Only this nonlinearity is coupled; scalar BN/pooling are retained. No full-model
equivariance, new activation, or performance claim follows.
"""

import torch
from torch import nn

from src.models.cifar_cnn_torch import CifarCNN


class MagnitudeTanh(nn.Module):
    """q -> tanh(|q|) q/|q|, continuous at zero; component-blocked channels."""

    def __init__(self,epsilon=1e-6):
        super().__init__()
        if epsilon<=0:
            raise ValueError("Positive numerical epsilon required")
        self.epsilon=epsilon

    def forward(self,inputs):
        if inputs.ndim!=4 or inputs.shape[1]%4:
            raise ValueError("Expected N,4*units,H,W in component-blocked order")
        work=inputs if inputs.dtype==torch.float64 else inputs.float()
        grouped=work.reshape(len(work),4,work.shape[1]//4,*work.shape[2:])
        radius=torch.linalg.vector_norm(grouped,dim=1,keepdim=True)
        # The Taylor branch implements the unit limiting multiplier and finite
        # Jacobian at zero. Norms/ratios use FP32 under mixed precision.
        scale=torch.where(radius>self.epsilon,torch.tanh(radius)/radius.clamp_min(self.epsilon),1-radius.square()/3)
        return (grouped*scale).reshape_as(inputs).to(inputs.dtype)


class ActivationCNN(CifarCNN):
    def __init__(self,config):
        super().__init__("quaternion",config["quaternion_widths"],config["real_widths"],config["dropout"])
        self.activation=config["activation"]
        for block in self.blocks:
            for index in (2,5):
                if self.activation=="split_leaky_relu":
                    block[index]=nn.LeakyReLU(config["negative_slope"],inplace=False)
                elif self.activation=="magnitude_tanh":
                    block[index]=MagnitudeTanh(config["activation_epsilon"])
                else:
                    raise ValueError("Unknown registered activation")


def make_activation_model(config,kind="quaternion"):
    if kind!="quaternion":
        raise ValueError("This bounded activation sensitivity pilot uses quaternion CNN6")
    return ActivationCNN(config)
