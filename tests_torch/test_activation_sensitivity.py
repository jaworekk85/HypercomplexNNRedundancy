from copy import deepcopy

import pytest
import torch

from src.cnn_training import first_update_audit
from src.compression.multilayer import SiteMoments,compress_site,consumer_energy,greedy_victims,site_gram,site_maps,sites,substituted_logits
from src.models.activation_cnn import MagnitudeTanh,make_activation_model
from src.models.cifar_cnn_torch import CifarCNN

torch.set_num_threads(2)


def config(activation):
    return {"activation":activation,"negative_slope":.1,"activation_epsilon":1e-6,
            "quaternion_widths":[4,8,8],"real_widths":[8,8,16],"dropout":[.1,.2,.3]}


def test_magnitude_tanh_joint_components_zero_and_gradients():
    layer=MagnitudeTanh()
    value=torch.tensor([1.,2.,-3.,4.],dtype=torch.float64).reshape(1,4,1,1).requires_grad_()
    expected=torch.tanh(value.norm())*value/value.norm()
    torch.testing.assert_close(layer(value),expected,atol=1e-14,rtol=1e-14)
    assert torch.autograd.gradcheck(layer,(value,))
    zero=torch.zeros(1,4,1,1,dtype=torch.float64,requires_grad=True)
    layer(zero).sum().backward()
    torch.testing.assert_close(zero.grad,torch.ones_like(zero),atol=0,rtol=0)
    assert torch.autograd.gradcheck(layer,(zero,))
    # Unit 0 and unit 1 occupy separate component-blocked coordinates.
    blocks=torch.randn(3,4,2,5,5,dtype=torch.float64)
    result=layer(blocks.reshape(3,8,5,5)).reshape_as(blocks)
    radius=blocks.norm(dim=1,keepdim=True)
    torch.testing.assert_close(result,blocks*torch.tanh(radius)/radius)
    assert float(result.norm(dim=1).max())<=1+1e-12


@pytest.mark.parametrize("activation",["split_leaky_relu","magnitude_tanh"])
def test_tracked_updates_and_unchanged_initial_parameters(activation):
    cfg=config(activation)
    torch.manual_seed(21);model=make_activation_model(cfg)
    torch.manual_seed(21);reference=CifarCNN("quaternion",cfg["quaternion_widths"],cfg["real_widths"],cfg["dropout"])
    assert all(torch.equal(v,reference.state_dict()[k]) for k,v in model.state_dict().items())
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in reference.parameters())
    inputs=torch.randn(8,3,32,32)
    audit=first_update_audit(model,inputs,torch.arange(8)%10,torch.optim.AdamW(model.parameters(),lr=.001))
    assert all(audit["checks"].values()) and len(audit["quaternion_layers"])==6


@pytest.mark.parametrize("activation",["split_leaky_relu","magnitude_tanh"])
@pytest.mark.parametrize("fraction",[.25,.5])
def test_all_sites_exactly_fold_and_reload_with_new_activation(activation,fraction):
    cfg=config(activation);torch.manual_seed(22)
    current=make_activation_model(cfg).eval();images=torch.randn(10,3,32,32)
    for site in sites(current):
        width=current.blocks[site[0]][3*site[1]].out_channels
        moments=SiteMoments(*site_gram(current,site,images[:6]),True)
        victims=greedy_victims(moments,round(width*fraction),consumer_energy(current,site),.001,"quaternion_right")
        maps=site_maps(moments,victims,.001,"quaternion_right")
        compact,_,_=compress_site(current,site,victims,maps)
        with torch.no_grad():
            torch.testing.assert_close(compact(images[6:]),substituted_logits(current,site,maps,images[6:]),atol=2e-6,rtol=1e-5)
        current=compact
    narrow=deepcopy(cfg);narrow["quaternion_widths"]=[w-round(w*fraction) for w in cfg["quaternion_widths"]]
    restored=make_activation_model(narrow).eval();restored.load_state_dict(current.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(restored(images),current(images),atol=0,rtol=0)
