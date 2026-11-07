from unittest.mock import patch

import pytest

torch=pytest.importorskip("torch")
torch.set_num_threads(2)
from scripts import cifar_cnn as frozen
from scripts.train_fourblock import train
from src.cnn_training import first_update_audit
from src.models.cifar_fourblock import FourBlockCifarCNN
from src.compression.internal_merging import merge_internal,substitute_features
from src.compression.quaternion_merging import right_hamilton
from src.evaluation.block_ablation import predict_from_block


def settings():
    return {"seed":2026,"quaternion_widths":[2,3,4,5],"dropout":[.1,.2,.3,.3],"epochs":3,"batch_size":4,
            "learning_rate":.001,"minimum_learning_rate":.00001,"weight_decay":.0001,"mixed_precision":False}


def test_fourblock_tracked_updates_and_internal_surgery():
    torch.manual_seed(13)
    model=FourBlockCifarCNN((2,3,4,5))
    inputs=torch.randn(8,3,32,32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    assert all(first_update_audit(model,inputs,torch.arange(8)%10,optimizer)["checks"].values())
    model.eval()
    assert model.forward_features(inputs).shape==(8,20,2,2)
    pair={"keeper":0,"victim":2,"maps":{"quaternion_right":{"matrix":right_hamilton([.7,.2,-.1,.1]).tolist()}}}
    compact,kept,_=merge_internal(model,1,[pair],"quaternion_exact")
    features=torch.randn(5,12,8,8)
    expected=predict_from_block(model,1,substitute_features(features,[pair],"quaternion_right"))
    actual=predict_from_block(compact,1,features[:,[component*3+unit for component in range(4) for unit in kept]])
    torch.testing.assert_close(torch.tensor(actual),torch.tensor(expected),atol=2e-6,rtol=1e-5)


@pytest.mark.parametrize("interrupt_epoch",[2,3])
def test_resume_and_completed_checkpoint_finalization(tmp_path,interrupt_epoch):
    config=settings()
    images=torch.randint(256,(12,3,32,32),generator=torch.Generator().manual_seed(19),dtype=torch.uint8)
    tensors={key:images for key in ("train","calibration","validation")}
    labels={key:torch.arange(12)%10 for key in tensors}
    mean,std=torch.full((1,3,1,1),.5),torch.full((1,3,1,1),.2)
    expected=train(config,tensors,labels,mean,std,tmp_path / "reference",{})
    original_save=torch.save
    def interrupt(state,path,*args,**kwargs):
        original_save(state,path,*args,**kwargs)
        if str(path).endswith("last.tmp") and state["epoch"]==interrupt_epoch:
            raise RuntimeError("injected interruption")
    with patch.object(torch,"save",interrupt),pytest.raises(RuntimeError,match="injected"):
        train(config,tensors,labels,mean,std,tmp_path / "resumed",{})
    actual=train(config,tensors,labels,mean,std,tmp_path / "resumed",{})
    a=torch.load(tmp_path / "reference/model.pt",weights_only=True)
    b=torch.load(tmp_path / "resumed/model.pt",weights_only=True)
    assert all(torch.equal(value,b["model"][name]) for name,value in a["model"].items())
    assert expected["selected_validation"]==actual["selected_validation"]
    (tmp_path / "resumed/metrics.json").unlink()
    (tmp_path / "resumed/training.csv").unlink()
    recovered=train(config,tensors,labels,mean,std,tmp_path / "resumed",{})
    assert recovered["selected_validation"]==actual["selected_validation"]
    assert (tmp_path / "resumed/training.csv").exists()
