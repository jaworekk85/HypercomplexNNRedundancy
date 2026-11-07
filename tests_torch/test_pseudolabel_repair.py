from copy import deepcopy

import numpy as np
import pytest
import torch

from src.models.cifar_cnn_torch import CifarCNN
from src.pseudolabel_repair import budget_partition, hard_targets, repair

torch.set_num_threads(2)


def fixture():
    torch.manual_seed(72)
    model=CifarCNN("quaternion",(2,3,4),dropout=(.1,.2,.3))
    images=torch.randint(256,(16,3,16,16),dtype=torch.uint8)
    labels=torch.arange(16)%10
    config={"steps":6,"snapshots":[0,2,6],"checkpoint_every":2,"batch_size":4,
            "learning_rate":.001,"minimum_learning_rate":.00001,"weight_decay":.01,
            "confidence_threshold":0.,"unlabeled_loss_weight":1.}
    return model,images,labels,config


def test_partition_is_nested_class_balanced_and_unlabeled_is_common_disjoint():
    labels=np.repeat(np.arange(10),30)
    selected,pool=budget_partition(np.arange(300),labels,[0,2,5],128,2026)
    assert not len(selected[0]) and set(selected[2])<set(selected[5])
    assert np.bincount(labels[selected[2]]).tolist()==[2]*10
    assert not set(selected[5])&set(pool) and len(set(pool))==128
    logits=torch.tensor([[9.,0.],[.1,.2]],requires_grad=True)
    target,mask=hard_targets(logits,.95)
    assert target.tolist()==[0,1] and mask.tolist()==[True,False]
    assert not target.requires_grad and not mask.requires_grad


def test_no_labels_and_no_accepted_pseudo_labels_has_no_hidden_updates(tmp_path):
    model,images,labels,config=fixture();before=deepcopy(model.state_dict())
    config["confidence_threshold"]=1.
    result=repair(model,images[:0],labels[:0],images,torch.zeros(1,3,1,1),torch.ones(1,3,1,1),
                  config,2026,"self_pseudo",tmp_path,{},lambda m,s:{})
    assert result["optimizer_updates"]==0 and result["pseudo_accepted"]==0
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items())


@pytest.mark.parametrize("method",["supervised","self_pseudo","teacher_pseudo"])
def test_actual_quaternion_updates_frozen_buffers_and_exact_resume(tmp_path,monkeypatch,method):
    original,images,labels,config=fixture();model=deepcopy(original)
    buffers={n:b.clone() for n,b in original.named_buffers()}
    teacher=torch.nn.functional.one_hot(labels,num_classes=10).float()
    args=(images[:4],labels[:4],images[4:],torch.zeros(1,3,1,1),torch.ones(1,3,1,1),config,2026,method)
    callback=lambda m,s:{"diagnostic_only":float(m.readout.weight.sum())}
    full=repair(model,*args,tmp_path / "full",{},callback,teacher[4:])
    interrupted=deepcopy(original);real_save=torch.save

    def fail_after_checkpoint_two(obj,path,*a,**kw):
        if str(path).endswith("last.tmp") and obj["step"]==4:
            raise InterruptedError("simulated interruption")
        return real_save(obj,path,*a,**kw)

    with monkeypatch.context() as context:
        context.setattr(torch,"save",fail_after_checkpoint_two)
        with pytest.raises(InterruptedError):
            repair(interrupted,*args,tmp_path / "resume",{},callback,teacher[4:])
    resumed=repair(interrupted,*args,tmp_path / "resume",{},callback,teacher[4:])
    assert full["history"]==resumed["history"] and resumed["optimizer_updates"]==6
    assert all(torch.equal(value,interrupted.state_dict()[key]) for key,value in model.state_dict().items())
    assert all(torch.equal(value,dict(model.named_buffers())[key]) for key,value in buffers.items())
    kernels={name:r for name,r in resumed["first_update"]["parameters"].items() if name.endswith("kernel")}
    assert len(kernels)==6 and all(r["finite"] and r["maximum_update"]>0 for r in kernels.values())
    # Evaluation truth never enters the optimizer or changes its schedule.
    diagnostic_changed=deepcopy(original)
    repair(diagnostic_changed,*args,tmp_path / "different_diagnostic",{},lambda m,s:{"arbitrary_truth":-99},teacher[4:])
    assert all(torch.equal(v,diagnostic_changed.state_dict()[k]) for k,v in model.state_dict().items())
