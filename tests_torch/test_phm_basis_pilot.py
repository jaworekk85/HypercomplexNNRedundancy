from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from src.compression.phm_activation_fit import normalized, objective
from src.compression.phm_basis_pilot import (compact_block, counts, fit_consumer,
                                          initialize_consumer, kept_groups, set_repair_policy)
from src.compression.structure_catalogue import raw_moments
from src.models.cifar_phm_resnet import CifarPHMResNet20
from src.models.phm_conv import PHMConv2d
from src.trainable_map_repair import repair


@pytest.mark.parametrize("index", [0, 3, 8])
def test_physical_slice_matches_masked_forward_and_zero_removal(index):
    torch.set_num_threads(2)
    torch.manual_seed(31)
    source = CifarPHMResNet20((2,4,8),classes=2).eval().double()
    block = source.blocks[index]
    saved = deepcopy(block.state_dict())
    width = block.conv1.out_channels//4
    kept = list(range(0,width,2))
    narrow = compact_block(block,kept)
    x = torch.randn(4,block.conv1.in_channels,8,8,dtype=torch.float64)
    z = F.relu(block.bn1(block.conv1(x)))
    mask = torch.zeros_like(z)
    scalar = [component*width+k for component in range(4) for k in kept]
    mask[:,scalar] = z[:,scalar]
    expected = F.relu(block.bn2(block.conv2(mask))+block.shortcut(x))
    torch.testing.assert_close(narrow(x),expected,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(narrow.conv1.effective_kernel(),block.conv1.effective_kernel()[scalar],atol=0,rtol=0)
    torch.testing.assert_close(narrow.conv2.effective_kernel(),block.conv2.effective_kernel()[:,scalar],atol=0,rtol=0)
    identity = compact_block(block,list(range(width)))
    torch.testing.assert_close(identity(x),block(x),atol=0,rtol=0)
    assert counts(narrow)["stored_real_parameters"] < counts(block)["stored_real_parameters"]
    for name,value in saved.items():
        torch.testing.assert_close(block.state_dict()[name],value,atol=0,rtol=0)
    with pytest.raises(ValueError):
        compact_block(block,[])


def test_counts_repair_scope_group_ranking_and_serialized_architecture(tmp_path):
    from scripts.phm_basis_compression_pilot import save_model,load_model
    torch.set_num_threads(2)
    torch.manual_seed(11)
    model = CifarPHMResNet20().eval()
    assert counts(model)["stored_real_parameters"] == 272746
    for i in (0,8):
        keep = kept_groups(model.blocks[i],.5)
        assert len(keep) == model.blocks[i].conv1.out_channels//8
        assert keep == kept_groups(model.blocks[i],.5)
        model.blocks[i] = compact_block(model.blocks[i],keep)
    assert counts(model)["stored_real_parameters"] == 233418
    set_repair_policy(model,"preserve",[0,8])
    assert counts(model)["trainable_real_parameters"] == 232266
    set_repair_policy(model,"adapt",[0,8])
    assert counts(model)["trainable_real_parameters"] == 232394
    assert [n for n,p in model.named_parameters() if n.endswith("basis") and p.requires_grad] == ["blocks.0.conv2.basis","blocks.8.conv2.basis"]
    path = tmp_path/"model.pt"
    save_model(model,path)
    baseline = {"quaternion_widths":[8,16,32],"classes":10,"phm_dimensions":[4]*18}
    replica = load_model(path,baseline,"cpu")
    x = torch.randn(2,3,8,8)
    torch.testing.assert_close(model.eval()(x),replica(x),atol=0,rtol=0)


def test_random_span_projection_is_seeded_and_preserve_adapt_initial_states_match():
    torch.manual_seed(18)
    original = PHMConv2d(8,12).double()
    preserve = initialize_consumer(original,"preserve",19)
    adapt = initialize_consumer(original,"adapt",19)
    for name,value in preserve.state_dict().items():
        torch.testing.assert_close(value,adapt.state_dict()[name],atol=0,rtol=0)
    a,b = [initialize_consumer(original,"reinitialize",91) for _ in range(2)]
    torch.testing.assert_close(a.effective_kernel(),b.effective_kernel(),atol=0,rtol=0)
    basis = a.basis.reshape(4,16)
    torch.testing.assert_close(basis@basis.T,torch.eye(4,dtype=torch.float64),atol=1e-12,rtol=1e-12)
    residual = (original.effective_kernel()-a.effective_kernel()).reshape(4,3,4,2,3,3).permute(0,2,1,3,4,5).reshape(16,-1)
    torch.testing.assert_close(basis@residual,torch.zeros(4,residual.shape[1],dtype=torch.float64),atol=1e-12,rtol=1e-12)
    assert not preserve.basis.requires_grad and adapt.basis.requires_grad


@pytest.mark.parametrize("policy", ["preserve","adapt","reinitialize"])
def test_fit_objective_and_interruption_replay(tmp_path,policy):
    torch.set_num_threads(2)
    torch.manual_seed(9)
    layer = initialize_consumer(PHMConv2d(4,8),policy,71)
    x,y = torch.randn(4,4,5,5,dtype=torch.float64),torch.randn(4,8,5,5,dtype=torch.float64)
    moments = raw_moments(x,y)
    pack,scale = normalized(moments)
    torch.testing.assert_close(objective(layer,pack,scale),(layer(x)-y).square().mean()/scale,atol=1e-12,rtol=1e-12)
    config = {"steps":6,"learning_rate":.01,"checkpoint_interval":2}
    stop = tmp_path/"STOP"
    full = deepcopy(layer)
    a = fit_consumer(full,moments,config,tmp_path/"full",stop)
    partial = deepcopy(layer)
    def interrupt(step):
        if step == 2:
            stop.touch()
    assert fit_consumer(partial,moments,config,tmp_path/"resume",stop,after_step=interrupt) is None
    stop.unlink()
    resumed = deepcopy(layer)
    b = fit_consumer(resumed,moments,config,tmp_path/"resume",stop)
    assert a["best_normalized_fit_mse"] == b["best_normalized_fit_mse"]
    assert a["best_step"] == b["best_step"] and b["attempted_steps"] == 6
    for name,value in full.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[name],atol=0,rtol=0)
    if policy == "preserve":
        torch.testing.assert_close(full.basis,layer.basis,atol=0,rtol=0)


def test_frozen_basis_repair_and_completed_epoch_resume(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(8)
    initial = CifarPHMResNet20((1,2,4),classes=2)
    set_repair_policy(initial,"adapt",[0,8])
    x,y = torch.randint(256,(8,3,8,8),dtype=torch.uint8),torch.arange(8)%2
    tensors = {s:x for s in ("train","validation","check_calibration")}
    labels = {s:y for s in tensors}
    mean,std = torch.zeros(1,3,1,1),torch.ones(1,3,1,1)
    config = {"epochs":2,"batch_size":4,"learning_rate":.0001,"minimum_learning_rate":.000001,
              "weight_decay":.0001,"mixed_precision":False,"order_seed_offset":40,"augment_seed_offset":50}
    full,a,c,done = repair(deepcopy(initial),tensors,labels,mean,std,config,9,tmp_path/"full.pt")
    stop = tmp_path/"STOP"
    def interrupt(step):
        if step == 3:
            stop.touch()
    _,_,_,complete = repair(deepcopy(initial),tensors,labels,mean,std,config,9,tmp_path/"resume.pt",stop,after_batch=interrupt)
    assert not complete
    stop.unlink()
    resumed,b,d,done2 = repair(deepcopy(initial),tensors,labels,mean,std,config,9,tmp_path/"resume.pt",stop)
    assert done and done2 and c == d
    for name,value in full.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[name],atol=0,rtol=0)
    for name,p in initial.named_parameters():
        if not p.requires_grad:
            torch.testing.assert_close(full.state_dict()[name],p,atol=0,rtol=0)
    assert a[-1]["validation"] == b[-1]["validation"]


def test_gate_does_not_hide_bad_local_or_generic_training_results():
    from scripts.phm_basis_compression_pilot import decision
    study = {"blocks":[0,8],"policies":["preserve","adapt","reinitialize"],
             "continuation":{"minimum_mean_relative_check_mse_gain":.02,
                 "maximum_single_block_relative_check_mse_increase":.01,"minimum_accuracy_gain":.005}}
    local = [{"block":b,"policy":p,"check_mse":.9 if p == "adapt" else 1.} for b in (0,8) for p in study["policies"]]
    def row(name,ce,accuracy):
        return {"case":name,"after":{"validation":{"cross_entropy":ce,"accuracy":accuracy}}}
    rows = [row("compressed_preserve",.5,.90),row("compressed_adapt",.45,.92),
            row("compressed_reinitialize",.6,.89),row("uncompressed_preserve",.30,.94),row("uncompressed_adapt",.29,.94)]
    assert decision(study,local,rows)["continue_to_replication"]
    rows[-1] = row("uncompressed_adapt",.20,.94)
    assert not decision(study,local,rows)["continue_to_replication"]
    local[-2]["check_mse"] = 1.1
    assert not decision(study,local,rows)["checks"]["local_heldout_signal"]
