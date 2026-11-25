from copy import deepcopy
import pytest
import torch

from src.models.cifar_resnet import BasicBlock, CifarResNet20
from src.compression.resnet_reassembly import (compile_block, original_quartets, sampled_design,
    positions_for, conditional_gains, objective, make_catalogue, select_groups,
    selection_statistics, final_consumer_fit, local_mse)

torch.set_num_threads(2)


def config():
    return dict(role_catalogue_limit=12,full_catalogue_limit=20,neighbour_variants=2,
                exchange_sweeps=1,bank_batch_size=2,ridge_alpha=.001)


def fixture():
    torch.manual_seed(71)
    block = BasicBlock('quaternion',2,4).eval()
    x = torch.randn(4,8,5,5)
    with torch.no_grad():
        h = torch.relu(block.bn1(block.conv1(x)))
        y = block.conv2(h)
    return block,dict(inputs=x,features=h,targets=y)


def test_identity_and_whole_subset_forward():
    block,pack = fixture()
    cloned,_ = compile_block(block,original_quartets(4))
    torch.testing.assert_close(cloned(pack['inputs']),block(pack['inputs']),rtol=0,atol=0)
    kept = [0,2]
    small,_ = compile_block(block,[original_quartets(4)[i] for i in kept])
    mask = torch.zeros_like(pack['features'])
    indices = [r*4+i for r in range(4) for i in kept]
    mask[:,indices] = pack['features'][:,indices]
    expected = torch.relu(block.bn2(block.conv2(mask))+block.shortcut(pack['inputs']))
    torch.testing.assert_close(small(pack['inputs']),expected)


def test_overlapping_bank_matches_individual_and_final_rejects_overlap():
    block,pack = fixture()
    groups = [original_quartets(4)[0],(0,5,10,15)]
    with pytest.raises(ValueError):
        compile_block(block,groups)
    bank,_ = compile_block(block,groups,allow_overlap=True)
    with torch.no_grad():
        all_h = torch.relu(bank.bn1(bank.conv1(pack['inputs'])))
        for index,q in enumerate(groups):
            single,_ = compile_block(block,[q])
            h = torch.relu(single.bn1(single.conv1(pack['inputs'])))
            torch.testing.assert_close(all_h[:,[index+r*2 for r in range(4)]],h)


def test_sampled_design_orientation_with_multiple_outputs_and_padding():
    block,pack = fixture()
    pos = positions_for(4,25,7,17)
    design,target = sampled_design(pack['features'],pack['targets'],pos)
    coefficients = block.conv2.kernel.detach().double().permute(2,0,3,4,1).reshape(4*36,4)
    torch.testing.assert_close(design@coefficients,target,rtol=1e-5,atol=2e-6)


@pytest.mark.parametrize('base',[[],[0],[0,2]])
def test_schur_gains_equal_direct_objective(base):
    torch.manual_seed(3)
    design = torch.randn(60,20,dtype=torch.float64)
    target = torch.randn(60,3,dtype=torch.float64)
    prior = torch.randn(20,3,dtype=torch.float64)
    stats = dict(gram=design.T@design,prior=prior,rhs=design.T@target+.3*prior,
                 penalty=.3,yy=float(target.square().sum()),block_size=4)
    gains = conditional_gains(stats,base)
    for i in set(range(5))-set(base):
        assert gains[i] == pytest.approx(objective(stats,base)-objective(stats,base+[i]),abs=1e-10)


def test_catalogue_search_and_realized_statistics():
    block,pack = fixture()
    bank,meta = make_catalogue(block,pack['features'],2,config())
    assert set(original_quartets(4)).issubset(bank)
    assert all(len(set(q)) == 4 for q in bank)
    assert all(all(s//4 == r for r,s in enumerate(q)) for q in bank[:meta['role_catalogue_count']])
    stats = selection_statistics(block,bank,pack,positions_for(4,25,4,19),config(),'cpu')
    for policy in ('whole_joint','role_fixed','free_roles'):
        groups,trace = select_groups(stats,bank,4,2,policy,config())
        assert len({s for q in groups for s in q}) == 8
        if policy == 'whole_joint':
            assert set(groups).issubset(original_quartets(4))
        if policy == 'role_fixed':
            assert all(s//4 == r for q in groups for r,s in enumerate(q))
        assert all(t['improvement'] > 0 for t in trace['trace'] if t['phase'] == 'exchange')
    # Exact source belongs to the bank and recovers the target with its prior.
    original_ids = [bank.index(q) for q in original_quartets(4)]
    assert objective(stats,original_ids) == pytest.approx(0,abs=1e-8)


def test_full_moment_fit_improves_realized_features():
    block,pack = fixture()
    small,_ = compile_block(block,[(0,5,10,15),(1,6,11,12)])
    before = local_mse(small,pack,'cpu',2)
    fit = final_consumer_fit(small,pack,.01,'cpu',2)
    after = local_mse(small,pack,'cpu',2)
    assert after <= before+1e-8
    assert fit['fit_mse_float64'] == pytest.approx(after,rel=1e-5,abs=1e-7)


def test_actual_compact_counts_and_standard_architecture_reload():
    model = CifarResNet20('quaternion').eval()
    assert sum(p.numel() for p in model.parameters()) == 271594
    for index in (0,8):
        width = model.blocks[index].conv1.out_channels
        model.blocks[index],_ = compile_block(model.blocks[index],original_quartets(width)[:width//2])
    assert sum(p.numel() for p in model.parameters()) == 232266
    clone = CifarResNet20('quaternion',hidden_widths=[b.conv1.out_channels for b in model.blocks]).eval()
    clone.load_state_dict(model.state_dict())
    x = torch.randn(2,3,32,32)
    torch.testing.assert_close(clone(x),model(x),rtol=0,atol=0)
