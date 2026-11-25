from copy import deepcopy

import pytest
import torch

from src.compression.component_reassembly import packing_indices
from src.compression.grouping_mechanism import (regroup,case_groups,raw,permute_moments,q_moments,
    solve,relative_permutation,dense_permutation_check,physical_fit,fit_producer)
from src.compression.resnet_reassembly import compile_block,original_quartets
from src.models.cifar_resnet import BasicBlock

torch.set_num_threads(2)


def fixture():
    torch.manual_seed(117)
    source = BasicBlock('quaternion',2,8).eval()
    x = torch.randn(6,8,6,6)
    with torch.no_grad():
        h = torch.relu(source.bn1(source.conv1(x)))
        y = source.conv2(h)
    return source,{'inputs':x,'features':h,'targets':y}


@pytest.mark.parametrize('policy',['role','free'])
@pytest.mark.parametrize('fraction',[.5,1.])
def test_fixed_scalar_set_exact_mixing_and_determinism(policy,fraction):
    kept,width = [0,2,4,7],8
    groups = regroup(kept,width,policy,fraction,33)
    assert groups == regroup(kept,width,policy,fraction,33)
    expected = {s for i in kept for s in original_quartets(width)[i]}
    assert {s for q in groups for s in q} == expected
    assert sum(len({s%width for s in q}) > 1 for q in groups) == round(len(kept)*fraction)
    if policy == 'role':
        assert all(s//width == r for q in groups for r,s in enumerate(q))
    else:
        assert any(s//width != r for q in groups for r,s in enumerate(q))


def test_registered_cases_and_private_random_stream():
    cfg = {'shuffle_seed_offset':210000,'random_repeats':10}
    torch.manual_seed(9)
    before = torch.get_rng_state().clone()
    groups = case_groups([0,1,2,3],8,2026,0,cfg)
    assert len(groups) == 23
    assert groups['original'] == original_quartets(8)[:4]
    assert torch.equal(before,torch.get_rng_state())
    assert groups['free_100_0'] != groups['free_100_1']


def test_moment_permutation_equals_recollected_data():
    _,pack = fixture()
    features = pack['features'][:,:16]
    permutation = torch.randperm(16).tolist()
    expected = raw(features[:,permutation],pack['targets'])
    actual = permute_moments(raw(features,pack['targets']),permutation)
    for key in ('gram','cross'):
        torch.testing.assert_close(actual[key],expected[key],rtol=1e-12,atol=1e-12)


def test_whole_unit_reorder_and_dense_inverse_identity():
    source,pack = fixture()
    kept = [0,2,4,7]
    ids = packing_indices([original_quartets(8)[i] for i in kept],8)
    fit = raw(pack['features'][:,ids],pack['targets'])
    shape = (4,8,4,3,3)
    _,original = solve(fit,fit,shape,.1)
    groups = [original_quartets(8)[i] for i in reversed(kept)]
    permuted = permute_moments(fit,relative_permutation(groups,kept,8))
    _,reordered = solve(permuted,permuted,shape,.1)
    assert original['regularized_fit_objective'] == pytest.approx(reordered['regularized_fit_objective'],abs=1e-8)
    mixed = regroup(kept,8,'free',1,33)
    assert dense_permutation_check(source,pack['features'],kept,mixed) < 1e-10


def test_held_out_targets_never_change_ridge_weights():
    _,pack = fixture()
    fit = raw(pack['features'][:,:16],pack['targets'])
    changed_check = raw(pack['features'][:,:16],pack['targets']+5)
    kernel,record = solve(fit,fit,(4,8,4,3,3),.1)
    other,changed = solve(fit,changed_check,(4,8,4,3,3),.1)
    torch.testing.assert_close(kernel,other,rtol=0,atol=0)
    assert changed['check_mse'] != record['check_mse']
    assert record['relative_normal_residual'] < 1e-12


def test_exact_original_producer_and_physical_float64_fit():
    source,pack = fixture()
    groups = original_quartets(8)[:4]
    block,_ = compile_block(source,groups)
    ids = packing_indices(groups,8)
    ideal = raw(pack['features'][:,ids],pack['targets'])
    _,expected = solve(ideal,ideal,tuple(block.conv2.kernel.shape),.1)
    packs = {s:{**pack,'selected_features':pack['features'][:,ids]} for s in ('fit','check_calibration')}
    actual = physical_fit(block,packs,.1,'cpu',2)
    assert actual['check_mse'] == pytest.approx(expected['check_mse'],abs=1e-10)
    assert actual['producer_feature_mse']['fit'] == 0


def test_producer_only_fit_freezes_bn_consumer_and_trains_four_components():
    source,pack = fixture()
    groups = regroup([0,1,2,3],8,'free',1,33)
    block,_ = compile_block(source,groups)
    before = deepcopy(block.state_dict())
    selected = {**pack,'selected_features':pack['features'][:,packing_indices(groups,8)]}
    fitted,audit = fit_producer(block,selected,{'steps':6,'batch_size':6,'learning_rate':.001,'seed_offset':220000},19,'cpu')
    for name,value in fitted.state_dict().items():
        if name != 'conv1.kernel':
            torch.testing.assert_close(value,before[name],rtol=0,atol=0)
    assert all(v > 0 for v in audit['component_update_norms'])
    assert audit['losses'][-1] < audit['losses'][0]
    assert not torch.equal(before['conv1.kernel'],fitted.conv1.kernel)
    # The original input model was a discarded-copy source, never mutated.
    for name,value in block.state_dict().items():
        torch.testing.assert_close(value,before[name],rtol=0,atol=0)


def test_producer_fit_honors_stop_before_update():
    source,pack = fixture()
    groups = regroup([0,1,2,3],8,'free',1,33)
    block,_ = compile_block(source,groups)
    selected = {**pack,'selected_features':pack['features'][:,packing_indices(groups,8)]}
    def stop():
        raise InterruptedError('technical pause')
    with pytest.raises(InterruptedError):
        fit_producer(block,selected,{'steps':6,'batch_size':2,'learning_rate':.001,'seed_offset':220000},19,'cpu',stop)
