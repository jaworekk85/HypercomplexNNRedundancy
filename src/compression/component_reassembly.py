"""Component selection compiled into a genuinely smaller Hamilton pair.

Projection onto fixed Hamilton blocks is elementary least squares. This module
tests realizability of scalar reassembly; it makes no novelty or accuracy claim.
The full teacher and the packing indices are never used by the student forward.
"""
import torch
from torch import nn

from src.models.quaternion_conv_torch import QuaternionConv2d, hamilton_kernel


def hamilton_basis(dtype=torch.float64, device="cpu"):
    eye = torch.eye(4, dtype=dtype, device=device)
    return torch.stack([hamilton_kernel(row.reshape(4,1,1,1,1))[:,:,0,0] for row in eye])


def packing_indices(quartets, source_units):
    groups = [tuple(int(v) for v in group) for group in quartets]
    flat = [v for group in groups for v in group]
    if not groups or any(len(group) != 4 for group in groups) or len(set(flat)) != len(flat):
        raise ValueError("Select distinct scalar components in nonempty ordered quartets")
    if min(flat) < 0 or max(flat) >= 4*source_units:
        raise ValueError("Component index outside producer output")
    # Original and new tensors are component-blocked, not unit-major.
    return [group[role] for role in range(4) for group in groups]


def project_hamilton_kernel(dense):
    if dense.ndim != 4 or dense.shape[0] % 4 or dense.shape[1] % 4:
        raise ValueError("Expected component-blocked convolution kernel")
    o, i, h, w = dense.shape
    blocks = dense.reshape(4,o//4,4,i//4,h,w)
    basis = hamilton_basis(dense.dtype,dense.device)
    return torch.einsum("cab,aubvxy->cuvxy",basis,blocks)/4


def decoded_producer_rows(dense_rows):
    """Pull each target row back through its assigned Hamilton output role.

    Input: four selected scalar output rows, each [4*inputs,H,W].
    Output: four implied coefficient tensors, each [4,inputs,H,W].
    Their average is the Frobenius-optimal shared quaternion producer kernel.
    """
    if dense_rows.shape[0] != 4 or dense_rows.shape[1] % 4:
        raise ValueError("Expected exactly one proposed quartet")
    rows = dense_rows.reshape(4,4,dense_rows.shape[1]//4,*dense_rows.shape[2:])
    return torch.einsum("cab,abvxy->acvxy",hamilton_basis(dense_rows.dtype,dense_rows.device),rows)


class TinyQuaternionPair(nn.Module):
    """1x1 Q producer -> scalar BN/ReLU -> 1x1 Q consumer."""
    def __init__(self, inputs=1, hidden=2, outputs=1):
        super().__init__()
        self.producer = QuaternionConv2d(inputs,hidden,1,padding=0)
        self.bn = nn.BatchNorm2d(4*hidden)
        self.consumer = QuaternionConv2d(hidden,outputs,1,padding=0)

    def features(self,x):
        return torch.relu(self.bn(self.producer(x)))

    def forward(self,x):
        return self.consumer(self.features(x))


@torch.no_grad()
def compiled_pair(teacher,quartets):
    source = teacher.producer
    indices = packing_indices(quartets,source.out_channels)
    if source.kernel.shape[-2:] != (1,1):
        raise ValueError("First diagnostic is limited to 1x1 convolutions")
    target_rows = hamilton_kernel(source.kernel)[indices]
    coefficients = project_hamilton_kernel(target_rows)
    result = TinyQuaternionPair(source.in_channels,len(quartets),teacher.consumer.out_channels).to(source.kernel).eval()
    result.producer.kernel.copy_(coefficients)
    for name in ("weight","bias","running_mean","running_var"):
        getattr(result.bn,name).copy_(getattr(teacher.bn,name)[indices])
    result.bn.eps, result.bn.momentum = teacher.bn.eps, teacher.bn.momentum
    result.bn.num_batches_tracked.copy_(teacher.bn.num_batches_tracked)
    error = float((target_rows-hamilton_kernel(coefficients)).square().sum())
    return result,{"producer_kernel_squared_error":error,"indices":indices}


def quaternion_design(features):
    if features.ndim != 2 or features.shape[1] % 4:
        raise ValueError("Expected [examples,4*input_units]")
    x = features.reshape(len(features),4,-1)
    # theta order [coefficient, input unit]; target rows [example, component].
    return torch.einsum("cab,nbk->nack",hamilton_basis(features.dtype,features.device),x).reshape(4*len(x),-1)


@torch.no_grad()
def refit_consumer(model,features,targets,ridge_alpha):
    if features.shape[-2:] != (1,1) or targets.shape[-2:] != (1,1):
        raise ValueError("Only 1x1 synthetic spatial tensors are supported")
    x,y = features[:,:,0,0],targets[:,:,0,0]
    design = quaternion_design(x)
    response = y.reshape(len(y),4,-1).reshape(4*len(y),-1)
    gram = design.T@design
    penalty = ridge_alpha*max(float(gram.diag().mean()),1e-12)
    values = torch.linalg.solve(gram+penalty*torch.eye(len(gram),dtype=gram.dtype,device=gram.device),design.T@response)
    kernel = values.reshape(4,model.consumer.in_channels,model.consumer.out_channels).permute(0,2,1)[:,:,:,None,None]
    model.consumer.kernel.copy_(kernel)
    predictions = model.consumer(features)
    return float((predictions-targets).square().mean())


def parameter_counts(model):
    return {"real_parameters":sum(p.numel() for p in model.parameters()),
            "buffer_elements":sum(b.numel() for b in model.buffers()),
            "producer_quaternion_units":model.producer.out_channels,
            "consumer_quaternion_inputs":model.consumer.in_channels}
