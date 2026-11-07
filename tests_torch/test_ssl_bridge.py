from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.cifar_cnn import make_model
from src.models.circulant_cnn import make_circulant
from src.ssl_bridge import SSLAdapter,components_api,partition,sibling_api,train

torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[1]
API,BALANCED=sibling_api(ROOT / "../FewshotLearning")


def fixture(method="soft"):
    config={"seed":0,"architecture":"cnn","embedding_dim":128,"budget_per_class":2,"validation_per_class":1,
        "epochs":2,"batch_size":8,"unlabeled_batch_size":8,"learning_rate":.003,"momentum":.9,"weight_decay":.0005,
        "mean":[.4914,.4822,.4465],"std":[.2023,.1994,.2010],"selection":"last","num_workers":0}
    config=API.select_method(config,"ce_2v_pseudo_soft" if method=="soft" else "ce_2v_u_control")
    images=np.random.default_rng(21).integers(0,256,(100,32,32,3),dtype=np.uint8)
    labels=np.arange(100)%10
    study={"labels_per_class":2,"validation_per_class":1,"calibration_fit_images":10,"calibration_check_images":10}
    ids=partition(labels,study,0,BALANCED)
    model_config={"quaternion_widths":[4,4,8],"real_widths":[8,8,16],"dropout":[.1,.2,.3]}
    torch.manual_seed(23);model=make_model(model_config,"quaternion")
    return config,images,labels,ids,model,model_config


def test_exact_loss_reuse_and_no_hidden_labels_or_calibration_training():
    config,images,labels,ids,model,_=fixture()
    pilot,_,_=components_api(config,images,labels,ids,API)
    assert pilot.unlabeled.labels is None
    assert set(pilot.unlabeled.indices)==set(ids["unlabeled"])
    assert not set(pilot.unlabeled.indices)&set(np.r_[ids["labeled"],ids["validation"],ids["fit"],ids["check"]])
    # Changing every non-L provider label cannot change the training dataset.
    changed=labels.copy();other=np.setdiff1d(np.arange(len(labels)),ids["labeled"]);changed[other]=(changed[other]+3)%10
    other_pilot,_,_=components_api(config,images,changed,ids,API)
    assert other_pilot.train.labels==pilot.train.labels and other_pilot.unlabeled.labels is None
    logits=torch.tensor([[3.,0.],[.1,.2]],requires_grad=True);strong=torch.randn(2,2,requires_grad=True)
    state=API.EpochPseudoState(config);loss,*_=state.loss(logits,strong);loss.backward()
    assert logits.grad is None and strong.grad is not None
    state.finish_epoch();assert state.ready and state.std>=.05


@pytest.mark.parametrize("family",["quaternion","circulant"])
def test_real_joint_batch_updates_adapter_no_added_parameters_and_exact_resume(tmp_path,family):
    config,images,labels,ids,model,model_config=fixture()
    if family=="circulant":
        torch.manual_seed(23);model=make_circulant(model_config)
    adapter=SSLAdapter(model)
    assert sum(p.numel() for p in adapter.parameters())==sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        adapter.eval();probe=torch.randn(2,3,32,32);_,logits=adapter(probe)
        torch.testing.assert_close(logits,model(probe),atol=0,rtol=0)
    def callback(current,step,pilot):
        current.eval()
        return {"weight_sum":float(current.readout.weight.sum())}
    full_model=deepcopy(model);resumed_model=deepcopy(model)
    args=(config,images,labels,ids,API)
    full=train(full_model,*args,tmp_path / "full",{"test":"same"},6,3,[0,2,4,6],callback)
    stopped=train(resumed_model,*args,tmp_path / "resume",{"test":"same"},6,3,[0,2,4,6],callback,max_chunks=1)
    assert stopped["status"]=="stopped_at_epoch_boundary"
    resumed=train(resumed_model,*args,tmp_path / "resume",{"test":"same"},6,3,[0,2,4,6],callback)
    assert all(torch.equal(value,resumed_model.state_dict()[name]) for name,value in full_model.state_dict().items())
    assert full["diagnostics"]==resumed["diagnostics"] and full["soft_state"]==resumed["soft_state"]
    assert all(resumed["first_update"]["checks"].values())


def test_control_has_same_views_and_no_pseudo_loss():
    config,images,labels,ids,_,_=fixture();control=API.select_method({k:v for k,v in config.items() if k not in ("pseudo_weighting","pseudo_target","pseudo_temperature")},"ce_2v_u_control")
    a,la,_=components_api(config,images,labels,ids,API);b,lb,_=components_api(control,images,labels,ids,API)
    a.start_epoch(0);b.start_epoch(0)
    views_a,labels_a=next(iter(la));views_b,labels_b=next(iter(lb))
    assert torch.equal(labels_a,labels_b) and all(torch.equal(x,y) for x,y in zip(views_a,views_b))
    ua,ub=next(a.iterator),next(b.iterator)
    assert all(torch.equal(x,y) for x,y in zip(ua,ub))
    assert control["pseudo_label_weight"]==0 and b.loader is not None and b.soft_state is None
