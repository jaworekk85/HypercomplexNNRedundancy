from copy import deepcopy

import numpy as np
import pytest
import torch

from scripts.cifar_cnn import make_model
from src.compression import multilayer as old
from src.compression.internal_merging import BASIS
from src.compression.structured_blocks import StructuredMoments,basis_of,compact,energy,magnitude,reference
from src.models.circulant_cnn import CIRCULANT_BASIS,CirculantConv2d,circulant_kernel,make_circulant

torch.set_num_threads(2)


def cfg(): return {"quaternion_widths":[4,4,8],"real_widths":[8,8,16],"dropout":[.1,.2,.3]}


def test_circulant_expansion_closure_and_matched_initial_coefficients():
    value=torch.arange(4,dtype=torch.float64).reshape(4,1,1,1,1)
    expected=torch.tensor([[0,3,2,1],[1,0,3,2],[2,1,0,3],[3,2,1,0]],dtype=torch.float64)
    torch.testing.assert_close(circulant_kernel(value)[:,:,0,0],expected)
    for basis in (BASIS,CIRCULANT_BASIS):
        np.testing.assert_array_equal(np.einsum("coi,doi->cd",basis,basis),4*np.eye(4))
        for left in basis:
            for right in basis:
                product=left@right;coeff=np.einsum("oi,coi->c",product,basis)/4
                np.testing.assert_array_equal(product,np.einsum("c,coi->oi",coeff,basis))
    torch.manual_seed(9);q=make_model(cfg(),"quaternion")
    torch.manual_seed(9);c=make_circulant(cfg())
    assert all(torch.equal(v,c.state_dict()[k]) for k,v in q.state_dict().items())
    assert sum(p.numel() for p in q.parameters())==sum(p.numel() for p in c.parameters())
    torch.manual_seed(9);full=make_circulant({**cfg(),"quaternion_widths":[8,16,32]})
    assert sum(p.numel() for p in full.parameters())==73898


@pytest.mark.parametrize("family",["quaternion","circulant"])
def test_exact_surgery_folding_all_sites_and_legacy_quaternion_equivalence(family):
    torch.manual_seed(11);config=cfg()
    model=make_model(config,"quaternion") if family=="quaternion" else make_circulant(config)
    model.eval();images=torch.randn(10,3,16,16)
    for site in old.sites(model):
        gram,rows=old.site_gram(model,site,images[:6]);moments=StructuredMoments(gram,rows,basis_of(model))
        victims=old.greedy_victims(moments,old.units(model,site)//2,energy(model,site),.001,"structured")
        maps=old.site_maps(moments,victims,.001,"structured")
        result,lost=compact(model,site,victims,maps)
        with torch.no_grad(): torch.testing.assert_close(result(images[6:]),reference(model,site,maps,images[6:]),atol=2e-6,rtol=1e-5)
        assert lost<1e-20
        if family=="quaternion":
            original=old.SiteMoments(gram,rows,True)
            for victim in victims:
                a=moments.fit(victim,maps[0]["kept"],.001)[0]
                b=original.fit(victim,maps[0]["kept"],.001,"quaternion_right")[0]
                np.testing.assert_allclose(a,b,rtol=0,atol=0)
            prior=old.compress_site(model,site,victims,maps)[0]
            assert all(torch.equal(v,prior.state_dict()[k]) for k,v in result.state_dict().items())
            assert magnitude(model,site,len(victims))==old.magnitude_victims(model,site,len(victims))
        model=result
    narrow={**config,"quaternion_widths":[w//2 for w in config["quaternion_widths"]]}
    restored=make_model(narrow,"quaternion") if family=="quaternion" else make_circulant(narrow)
    restored.eval();restored.load_state_dict(model.state_dict())
    with torch.no_grad(): torch.testing.assert_close(restored(images),model(images),rtol=0,atol=0)


def test_circulant_tracked_four_coefficient_gradients_and_actual_updates():
    torch.manual_seed(12);layer=CirculantConv2d(2,3).double()
    inputs=torch.randn(4,8,7,7,dtype=torch.float64)
    before=layer.kernel.detach().clone();optimizer=torch.optim.AdamW(layer.parameters(),lr=.001)
    layer(inputs).square().mean().backward()
    assert layer.kernel.grad is not None and bool(torch.isfinite(layer.kernel.grad).all())
    assert bool((layer.kernel.grad.flatten(1).norm(dim=1)>0).all())
    optimizer.step();assert bool(((layer.kernel-before).flatten(1).norm(dim=1)>0).all())
