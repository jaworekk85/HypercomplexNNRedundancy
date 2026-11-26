from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.compression.grouping_statistics_control import (recalibrate_bn,collect_consumer,
    standardize_packs,add_centered_normalization,parameter_hash)
from src.cnn_training import prepare_batch

torch.set_num_threads(2)


class TwoBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,4,3,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4,4,3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(4)

    def forward(self,x):
        return self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))


def fixture():
    torch.manual_seed(71)
    model = TwoBN().eval()
    images = torch.randint(256,(8,3,6,6),dtype=torch.uint8)
    return model,images,torch.zeros(1,3,1,1),torch.ones(1,3,1,1)


def test_sequential_population_calibration_uses_updated_eval_prefix():
    model,images,mean,std = fixture()
    before = deepcopy(model.state_dict())
    result,audit = recalibrate_bn(model,images,mean,std,3)
    x = prepare_batch(images,mean,std)
    with torch.no_grad():
        first = result.conv1(x)
        second = result.conv2(F.relu(result.bn1(first)))
    for layer,values in ((result.bn1,first),(result.bn2,second)):
        torch.testing.assert_close(layer.running_mean,values.mean((0,2,3)),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(layer.running_var,values.var((0,2,3),correction=0),atol=1e-6,rtol=1e-5)
        assert int(layer.num_batches_tracked) == 3
    assert parameter_hash(result) == parameter_hash(model)
    assert audit['trainable_parameters_unchanged']
    assert not result.training and not result.bn1.training
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,before[name],rtol=0,atol=0)


def test_calibration_independent_of_stream_batch_boundaries():
    model,images,mean,std = fixture()
    first,_ = recalibrate_bn(model,images,mean,std,2)
    second,_ = recalibrate_bn(model,images,mean,std,5)
    for name in ('bn1','bn2'):
        a,b = getattr(first,name),getattr(second,name)
        torch.testing.assert_close(a.running_mean,b.running_mean,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(a.running_var,b.running_var,atol=1e-6,rtol=1e-5)


def test_calibrated_state_roundtrip_reproduces_forward():
    model,images,mean,std = fixture()
    result,_ = recalibrate_bn(model,images,mean,std,4)
    replica = deepcopy(model)
    replica.load_state_dict(result.state_dict())
    torch.testing.assert_close(replica(prepare_batch(images,mean,std)),result(prepare_batch(images,mean,std)),rtol=0,atol=0)


def test_collect_consumer_stops_at_layer_and_removes_hook():
    model,images,mean,std = fixture()
    values = collect_consumer(model,'conv2',images,mean,std,3)
    with torch.no_grad():
        expected = model.conv2(F.relu(model.bn1(model.conv1(prepare_batch(images,mean,std)))))
    torch.testing.assert_close(values['targets'],expected)
    assert not model.conv2._forward_hooks


def test_stop_cleans_bn_and_consumer_hooks():
    model,images,mean,std = fixture()
    calls = []
    def stop():
        calls.append(True)
        if len(calls) >= 3:
            raise InterruptedError('technical STOP')
    with pytest.raises(InterruptedError):
        recalibrate_bn(model,images,mean,std,2,stop)
    assert not model.bn1._forward_pre_hooks
    calls.clear()
    with pytest.raises(InterruptedError):
        collect_consumer(model,'conv2',images,mean,std,2,stop)
    assert not model.conv2._forward_hooks


def test_standardization_uses_fit_only_and_handles_constant_channels():
    torch.manual_seed(3)
    fit = torch.randn(6,8,4,4)
    fit[:,0] = 2
    check = torch.randn_like(fit)+20
    packs = {'fit':{'features':fit},'check_calibration':{'features':check}}
    values,stats = standardize_packs(packs)
    assert stats['constant_channels'] == 1
    assert float(values['fit']['features'][:,0].abs().max()) == 0
    torch.testing.assert_close(values['fit']['features'][:,1:].mean((0,2,3)),torch.zeros(7),atol=1e-6,rtol=0)
    torch.testing.assert_close(values['fit']['features'][:,1:].var((0,2,3),correction=0),torch.ones(7),atol=1e-6,rtol=0)
    assert float(values['check_calibration']['features'][:,1:].mean()) > 10
    altered,_ = standardize_packs({'fit':{'features':fit},'check_calibration':{'features':check*100}})
    torch.testing.assert_close(values['fit']['features'],altered['fit']['features'],rtol=0,atol=0)


def test_target_centering_uses_fit_means_without_refitting_weights():
    fit = torch.tensor([1.,3.]).reshape(2,1,1,1)
    check = torch.tensor([10.,12.]).reshape(2,1,1,1)
    rows = [{'fit_mse':.5,'check_mse':2.,'lambda':3.}]
    values,meta = add_centered_normalization(rows,dict(fit=fit,check_calibration=check))
    assert meta['fit_channel_target_mean'] == [2.]
    assert meta['centered_energy']['fit'] == 1
    assert meta['centered_energy']['check_calibration'] == 82
    assert values[0]['check_centered_normalized_mse'] == pytest.approx(2/82)
    assert values[0]['lambda'] == rows[0]['lambda']
    assert 'check_centered_normalized_mse' not in rows[0]
