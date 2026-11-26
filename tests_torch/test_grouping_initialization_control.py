from copy import deepcopy

import pytest
import torch

from src.compression.grouping_initialization_control import (consumer_targets,fit_cell,policy_summary,crossed_summary)
from src.compression.grouping_mechanism import case_groups
from src.models.cifar_resnet import BasicBlock

torch.set_num_threads(2)


def fixture():
    torch.manual_seed(125)
    block = BasicBlock('quaternion',2,8).eval()
    x = torch.randn(8,8,5,5)
    with torch.no_grad():
        h = torch.relu(block.bn1(block.conv1(x)))
        y = block.conv2(h)
    packs = {s:{'features':h[ids],'targets':y[ids]} for s,ids in [('fit',slice(0,4)),('check_calibration',slice(4,8))]}
    assignment = {'kept':[0,1,2,3],'source_width':8,
        'groups':case_groups([0,1,2,3],8,19,0,{'random_repeats':1,'shuffle_seed_offset':210000})}
    assignment['groups'] = {k:v for k,v in assignment['groups'].items() if k == 'original' or '_100_' in k}
    return block,packs,assignment


def run(packs,assignment):
    return fit_cell(packs,{s:p['targets'] for s,p in packs.items()},assignment,(4,8,4,3,3),.001)


def test_target_scaling_leaves_normalized_errors_and_penalties_invariant():
    _,packs,assignment = fixture()
    rows,kernels,_ = run(packs,assignment)
    scaled = {s:{**p,'targets':-7*p['targets']} for s,p in packs.items()}
    changed,values,_ = run(scaled,assignment)
    for row,other,kernel,value in zip(rows,changed,kernels,values):
        assert row['lambda'] == other['lambda']
        assert other['check_normalized_mse'] == pytest.approx(row['check_normalized_mse'],rel=1e-6)
        torch.testing.assert_close(value,-7*kernel,rtol=1e-5,atol=1e-6)


def test_input_scaling_keeps_trace_scaled_ridge_predictions_invariant():
    _,packs,assignment = fixture()
    rows,kernels,_ = run(packs,assignment)
    scaled = {s:{**p,'features':3*p['features']} for s,p in packs.items()}
    changed,values,_ = run(scaled,assignment)
    for row,other,kernel,value in zip(rows,changed,kernels,values):
        assert other['lambda'] == pytest.approx(9*row['lambda'],rel=1e-6)
        assert other['check_normalized_mse'] == pytest.approx(row['check_normalized_mse'],rel=1e-6)
        torch.testing.assert_close(value,kernel/3,rtol=1e-5,atol=1e-6)


def test_cross_targets_use_consumer_without_mutating_weights_or_packs():
    block,packs,_ = fixture()
    before = deepcopy(block.state_dict())
    actual = consumer_targets(block.conv2,packs,'cpu',2)
    for s in packs:
        torch.testing.assert_close(actual[s],packs[s]['targets'])
    for name,value in block.state_dict().items():
        torch.testing.assert_close(value,before[name],rtol=0,atol=0)


def test_held_out_targets_do_not_influence_coefficients():
    _,packs,assignment = fixture()
    rows,kernels,_ = run(packs,assignment)
    altered = deepcopy(packs)
    altered['check_calibration']['targets'] += 5
    changed,values,_ = run(altered,assignment)
    for kernel,value in zip(kernels,values):
        torch.testing.assert_close(kernel,value,rtol=0,atol=0)
    assert rows[0]['check_normalized_mse'] != changed[0]['check_normalized_mse']


def test_architectural_penalty_exists_for_an_untrained_identity_quaternion():
    # H(1)=I. Swapping r/i inputs with fixed output coordinates is not another H(w).
    torch.manual_seed(3)
    h = torch.randn(60,4,1,1)
    packs = {s:{'features':h[ids],'targets':h[ids].clone()} for s,ids in [('fit',slice(0,30)),('check_calibration',slice(30,60))]}
    assignment = {'kept':[0],'source_width':1,'groups':{'original':[(0,1,2,3)],'free_100_0':[(1,0,2,3)]}}
    rows,_,_ = fit_cell(packs,{s:p['targets'] for s,p in packs.items()},assignment,(4,1,1,3,3),1e-9)
    assert rows[0]['check_normalized_mse'] < 1e-15
    assert rows[1]['check_normalized_mse'] > .1


def test_crossed_contrasts_have_expected_sign_and_interaction():
    def rows(delta):
        return [{'case':'original','check_normalized_mse':.1},
                {'case':'role_100_0','check_normalized_mse':.1+delta},
                {'case':'free_100_0','check_normalized_mse':.1+2*delta}]
    result = crossed_summary(dict(initial__initial=rows(.1),initial__trained=rows(.2),
                                  trained__initial=rows(.3),trained__trained=rows(.5)))
    c = result['contrasts']['role']
    assert c['training_increment'] == pytest.approx(.4)
    assert c['feature_increment_at_initial_consumer'] == pytest.approx(.2)
    assert c['consumer_increment_at_initial_features'] == pytest.approx(.1)
    assert c['interaction'] == pytest.approx(.1)


def test_zero_output_energy_is_reported_as_failure():
    _,packs,assignment = fixture()
    zero = {s:torch.zeros_like(p['targets']) for s,p in packs.items()}
    with pytest.raises(ValueError,match='Zero target energy'):
        fit_cell(packs,zero,assignment,(4,8,4,3,3),.001)


def test_stop_interrupts_before_fitting():
    _,packs,assignment = fixture()
    def stop():
        raise InterruptedError('technical STOP')
    with pytest.raises(InterruptedError):
        fit_cell(packs,{s:p['targets'] for s,p in packs.items()},assignment,(4,8,4,3,3),.001,stop=stop)
