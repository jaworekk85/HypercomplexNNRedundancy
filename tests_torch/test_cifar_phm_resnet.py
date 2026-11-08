from copy import deepcopy

import pytest
import torch

from src.cnn_training import first_update_audit
from src.models.cifar_phm_resnet import CifarPHMResNet20
from src.models.phm_conv import PHMConv2d
from src.models.quaternion_conv_torch import QuaternionConv2d
from src.resnet_training import train


def test_qstem_and_all_learned_phm_gradients_input_encoding_and_reload():
    torch.set_num_threads(2)
    torch.manual_seed(15)
    model = CifarPHMResNet20((2,4,8),dimensions=[2,4,8]*6)
    assert sum(isinstance(m,QuaternionConv2d) for m in model.modules())==1
    assert sum(isinstance(m,PHMConv2d) for m in model.modules())==18
    images = torch.randn(6,3,16,16)
    captured = []
    handle = model.stem.register_forward_pre_hook(lambda layer,args: captured.append(args[0].detach().clone()))
    assert model(images).shape==(6,10)
    handle.remove()
    assert torch.count_nonzero(captured[0][:,:1])==0
    torch.testing.assert_close(captured[0][:,1:],images,atol=0,rtol=0)
    audit = first_update_audit(model,images,torch.arange(6),torch.optim.SGD(model.parameters(),lr=.01))
    assert all(audit["checks"].values())
    clone = CifarPHMResNet20((2,4,8),dimensions=[2,4,8]*6).eval()
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(model.eval()(images),clone(images),atol=0,rtol=0)
    assert sum(p.numel() for p in CifarPHMResNet20().parameters())==272746
    with pytest.raises(ValueError):
        CifarPHMResNet20((1,2,4),dimensions=[8]*18)


def test_phm_training_exact_completed_epoch_resume(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(18)
    initial = CifarPHMResNet20((1,2,4),classes=2)
    x,y = torch.randint(256,(8,3,8,8),dtype=torch.uint8),torch.arange(8)%2
    tensors,targets = {"train":x,"validation":x},{"train":y,"validation":y}
    mean,std = torch.zeros(1,3,1,1),torch.ones(1,3,1,1)
    config = {"seed":9,"epochs":2,"batch_size":4,"learning_rate":.01,"warmup_epochs":1,
              "warmup_start_lr":.005,"milestones":[2],"gamma":.1,"momentum":.9,"weight_decay":.0001,
              "mixed_precision":False,"order_seed_offset":40,"augment_seed_offset":50}
    full,a,done = train(deepcopy(initial),tensors,targets,mean,std,config,tmp_path/"full.pt")
    stop = tmp_path/"STOP"
    def interrupt(step):
        if step==3: stop.touch()
    _,_,partial = train(deepcopy(initial),tensors,targets,mean,std,config,tmp_path/"resume.pt",stop,after_batch=interrupt)
    assert not partial
    stop.unlink()
    resumed,b,done2 = train(deepcopy(initial),tensors,targets,mean,std,config,tmp_path/"resume.pt")
    assert done and done2 and a["counts"]==b["counts"]
    assert a["selected_validation"]==b["selected_validation"]
    for name,value in full.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[name],atol=0,rtol=0)
