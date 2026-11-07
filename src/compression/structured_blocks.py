"""Matched four-component compression for Hamilton and known circulant blocks.

Reuses frozen site moments/ranking and hooks. Both spaces are closed under
matrix products, so structure-restricted reconstruction folds into the next
convolution. Only linear weights are folded; maps never commute through ReLU.
"""

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from src.compression import multilayer as legacy
from src.compression.internal_merging import BASIS as HAMILTON_BASIS
from src.compression.multisource import substitute_multi
from src.compression.quaternion_channels import component_indices
from src.models.circulant_cnn import CIRCULANT_BASIS


def basis_of(model):
    if model.kind=="quaternion": return HAMILTON_BASIS
    if model.kind=="circulant": return CIRCULANT_BASIS
    raise ValueError("Unsupported structured family")


class StructuredMoments(legacy.SiteMoments):
    def __init__(self,gram,rows,basis):
        super().__init__(gram,rows,True);self.basis=np.asarray(basis,dtype=np.float64)

    def fit(self,victim,kept,alpha,mode="structured"):
        if mode=="full_real": return super().fit(victim,kept,alpha,mode)
        if mode!="structured": raise ValueError("Unknown fit")
        kept=list(kept);m=self.moments;basis=self.basis
        sources,cross=m[kept][:,:,kept],m[kept][:,:,victim]
        gram=np.einsum("coi,doj,kilj->kcld",basis,basis,sources).reshape(4*len(kept),-1)
        rhs=np.einsum("coi,kio->kc",basis,cross).reshape(-1)
        solution=self._solve(gram,rhs,alpha)
        sse=float(np.trace(m[victim,:,victim]))-2*solution@rhs+solution@gram@solution
        return np.einsum("kc,coi->koi",solution.reshape(len(kept),4),basis),float(sse)/(4*self.rows)


@torch.no_grad()
def compact(model,site,victims,maps=None):
    width=legacy.units(model,site);kept=sorted(set(range(width))-set(victims))
    if not kept or len(set(victims))!=len(victims) or any(v<0 or v>=width for v in victims): raise ValueError("Bad victims")
    if maps and (sorted(row["victim"] for row in maps)!=sorted(victims) or any(row["kept"]!=kept for row in maps)):
        raise ValueError("Maps must cover exactly the victims and retained sources")
    indices=component_indices(kept,width);result=deepcopy(model)
    conv,bn=legacy.producer(model,site);old=legacy.consumer(model,site)
    if conv.bias is not None or (not isinstance(old,nn.Linear) and old.bias is not None): raise ValueError("Expected bias-free convolutions")
    narrow=type(conv)(conv.in_channels,len(kept),conv.kernel.shape[-1],conv.stride,conv.padding).to(conv.kernel)
    narrow.kernel.copy_(conv.kernel[:,kept])
    narrow_bn=nn.BatchNorm2d(len(indices),eps=bn.eps,momentum=bn.momentum).to(bn.weight)
    for name in ("weight","bias","running_mean","running_var"): getattr(narrow_bn,name).copy_(getattr(bn,name)[indices])
    narrow_bn.num_batches_tracked.copy_(bn.num_batches_tracked)
    block,layer=site;result.blocks[block][3*layer]=narrow;result.blocks[block][3*layer+1]=narrow_bn
    if isinstance(old,nn.Linear):
        new=nn.Linear(len(indices),old.out_features,bias=old.bias is not None).to(old.weight)
        new.weight.copy_(old.weight[:,indices])
        if old.bias is not None: new.bias.copy_(old.bias)
    else:
        new=type(old)(len(kept),old.out_channels,old.kernel.shape[-1],old.stride,old.padding).to(old.kernel)
        new.kernel.copy_(old.kernel[:,:,kept])
    legacy._set_consumer(result,site,new)
    discarded=energy=0.
    for row in maps or []:
        if isinstance(old,nn.Linear):
            outgoing=old.weight[:,component_indices([row["victim"]],width)].double()
            grouped=new.weight.view(len(new.weight),4,len(kept))
            for position,matrix in enumerate(row["matrices"]):
                grouped[:,:,position].add_((outgoing@torch.as_tensor(matrix,device=outgoing.device,dtype=torch.float64)).to(new.weight.dtype))
        else:
            basis=basis_of(model)
            coeff=old.kernel[:,:,row["victim"]].detach().cpu().double().numpy().transpose(1,2,3,0)
            outgoing=np.einsum("...c,coi->...oi",coeff,basis)
            for position,matrix in enumerate(row["matrices"]):
                product=outgoing@np.asarray(matrix)
                addition=np.einsum("...oi,coi->...c",product,basis)/4
                projected=np.einsum("...c,coi->...oi",addition,basis)
                discarded+=float(np.square(product-projected).sum());energy+=float(np.square(product).sum())
                new.kernel[:,:,position].add_(torch.as_tensor(addition.transpose(3,0,1,2),device=new.kernel.device,dtype=new.kernel.dtype))
    lost=discarded/energy if energy>1e-20 else 0.
    if maps and lost>1e-20: raise AssertionError("A structure-restricted map failed exact closure")
    return result.eval(),lost


@torch.no_grad()
def reference(model,site,maps,inputs):
    handle=legacy.output_module(model,site).register_forward_hook(lambda module,args,value:substitute_multi(value,maps,True))
    try: return model(inputs)
    finally: handle.remove()


def energy(model,site):
    old=legacy.consumer(model,site);width=legacy.units(model,site)
    if isinstance(old,nn.Linear): return np.square(old.weight.detach().cpu().double().numpy()).reshape(old.out_features,4,width).sum(axis=(0,1))
    return np.square(old.kernel.detach().cpu().double().numpy()).sum(axis=(0,1,3,4))


def magnitude(model,site,count):
    conv,bn=legacy.producer(model,site)
    raw=conv.kernel.detach().cpu().double().square().sum(dim=(0,2,3,4))
    scale=(bn.weight.detach().cpu().double()/torch.sqrt(bn.running_var.detach().cpu().double()+bn.eps)).reshape(4,conv.out_channels)
    score=(raw*scale.square().sum(0)).sqrt().tolist()
    return sorted(sorted(range(len(score)),key=lambda i:(score[i],i))[:count])
