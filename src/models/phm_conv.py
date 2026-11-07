"""PHM convolution and Frobenius Kronecker-SVD initialization.

Known constructions: ZhangEtAl2021, Grassucci2024, VanLoanPitsianis1993.
Block dimension n and Kronecker rank r are distinct; no new-algebra claim.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PHMConv2d(nn.Module):
    def __init__(self, inputs, outputs, kernel_size=3, n=4, rank=None, stride=1, padding=1):
        super().__init__()
        rank = n if rank is None else rank
        if n < 1 or inputs % n or outputs % n or not 1 <= rank <= min(n*n, outputs*inputs*kernel_size**2//n**2):
            raise ValueError("Incompatible PHM dimensions/rank")
        self.in_channels, self.out_channels = inputs, outputs
        self.n, self.rank, self.stride, self.padding = n, rank, stride, padding
        self.basis = nn.Parameter(torch.empty(rank, n, n))
        self.cores = nn.Parameter(torch.empty(rank, outputs//n, inputs//n, kernel_size, kernel_size))
        nn.init.normal_(self.basis, std=1/n**.5)
        nn.init.normal_(self.cores, std=(2/(rank*inputs*kernel_size**2))**.5)

    def effective_kernel(self):
        # Scalar output index(a,u), scalar input index(b,v), shared spatial bases.
        return torch.einsum("rab,ruvxy->aubvxy", self.basis, self.cores).reshape(self.out_channels, self.in_channels, *self.cores.shape[-2:])

    def forward(self, x):
        return F.conv2d(x, self.effective_kernel(), stride=self.stride, padding=self.padding)

    @classmethod
    @torch.no_grad()
    def from_kernel(cls, kernel, n, rank=None, stride=1, padding=1):
        o, i, h, w = kernel.shape
        if h != w or o % n or i % n: raise ValueError("Unsupported kernel partition")
        rank = n if rank is None else rank
        arranged = kernel.double().reshape(n, o//n, n, i//n, h, w).permute(0, 2, 1, 3, 4, 5).reshape(n*n, -1)
        u, s, vh = torch.linalg.svd(arranged, full_matrices=False)
        result = cls(i, o, h, n, rank, stride, padding).to(kernel)
        root = s[:rank].sqrt()
        result.basis.copy_((u[:, :rank]*root).T.reshape(rank, n, n))
        result.cores.copy_((root[:, None]*vh[:rank]).reshape_as(result.cores))
        return result
