from copy import deepcopy

import pytest
import torch

from src.compression.grouping_exchange import (canonical, validate_partition, fit_split,
    producer_decoder, projection_cost, proposals, search_partition, fit_inner, prediction_kl)
from src.compression.resnet_reassembly import original_quartets, compile_block
from src.models.cifar_resnet import BasicBlock

torch.set_num_threads(2)


def fixture():
    torch.manual_seed(71)
    source = BasicBlock('quaternion', 2, 4).eval()
    groups = canonical(original_quartets(4)[:2])
    config = {'rounds': 1, 'compatible_proposals': 3, 'random_proposals': 3, 'tolerance': 1e-10}
    return source, groups, config


def test_swaps_preserve_exact_scalar_set_and_roles_are_not_sorted_away():
    source, groups, config = fixture()
    values = proposals(groups, producer_decoder(source), config, 7)
    assert len(values) == len(set(values)) == 6
    assert values == proposals(groups, producer_decoder(source), config, 7)
    for candidate in values:
        validate_partition(candidate, groups, 4)
        assert candidate != groups
    changed_roles = [tuple(reversed(q)) for q in groups]
    assert canonical(changed_roles) != groups
    with pytest.raises(AssertionError, match='Retained scalar set'):
        validate_partition(original_quartets(4)[2:], groups, 4)


def test_projection_cost_matches_actual_compiler():
    source, groups, config = fixture()
    decoder = producer_decoder(source)
    for candidate in [groups]+proposals(groups, decoder, config, 7):
        _, description = compile_block(source, candidate)
        assert projection_cost(candidate, decoder) == pytest.approx(description['producer_projection_sse'], rel=1e-6, abs=1e-8)


def test_search_can_keep_original_without_forcing_mixing():
    source, groups, config = fixture()
    decoder = producer_decoder(source)
    alternate = proposals(groups, decoder, config, 7)[0]
    finalists, cache, traces = search_partition({'original': groups, 'mixed': alternate}, decoder,
        lambda qs: {'score': 0. if qs == groups else 1.}, config, 7)
    assert groups in finalists and cache[groups]['score'] == 0.
    assert not traces[0]['rounds'][0]['accepted']


def test_search_accepts_a_better_nonoriginal_partition():
    source, groups, config = fixture()
    decoder = producer_decoder(source)
    desired = proposals(groups, decoder, config, 7)[0]
    finalists, cache, traces = search_partition({'original': groups}, decoder,
        lambda qs: {'score': 0. if qs == desired else 1.}, config, 7)
    assert desired in finalists and groups in finalists
    assert traces[0]['rounds'][0]['accepted']
    assert len(cache) == 7


def test_inner_and_outer_splits_disjoint_reproducible():
    inner, outer = fit_split(500, 64, 64, 7)
    assert len(set(inner.tolist()) | set(outer.tolist())) == 128
    repeated = fit_split(500, 64, 64, 7)
    assert torch.equal(inner, repeated[0]) and torch.equal(outer, repeated[1])
    with pytest.raises(ValueError):
        fit_split(4, 3, 3, 7)


def test_inner_compact_fit_preserves_teacher_and_solves_system():
    source, groups, _ = fixture()
    before = deepcopy(source.state_dict())
    x = torch.randn(6, 8, 5, 5)
    with torch.no_grad():
        y = source.conv2(torch.relu(source.bn1(source.conv1(x))))
    fitted, record = fit_inner(source, groups, x, y, .01, 'cpu', 2)
    assert record['normal_residual'] < 1e-8
    assert fitted.conv1.out_channels == 2 and fitted.conv2.in_channels == 2
    for name, value in before.items():
        assert torch.equal(value, source.state_dict()[name])


def test_prediction_kl_has_identity_zero_and_temperature_checks():
    x = torch.tensor([[1., 2., -1.], [0., 3., 1.]])
    assert prediction_kl(x, x, 2.) == pytest.approx(0., abs=1e-15)
    assert prediction_kl(x, -x, 2.) > 0
    with pytest.raises(ValueError):
        prediction_kl(x, -x, 0.)
