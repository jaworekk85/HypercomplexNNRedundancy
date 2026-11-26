from copy import deepcopy

import pytest
import torch

from scripts.eurosat_grouping_replication import save_model, load_model
from src.cnn_training import prepare_batch
from src.compression.component_reassembly import packing_indices
from src.compression.eurosat_grouping import (ConvPair, install_pair, compile_pair,
    make_compact, producer_inputs)
from src.compression.grouping_mechanism import dense_permutation_check
from src.compression.resnet_reassembly import original_quartets

torch.set_num_threads(2)


def fixture():
    torch.manual_seed(71)
    config = {'quaternion_widths': [4, 4, 4], 'real_widths': [8, 8, 8], 'dropout': [.1, .2, .3]}
    return config, make_compact(config).eval()


def test_extracted_pair_matches_existing_sequence_prefix():
    _, model = fixture()
    x = torch.randn(3, 4, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(ConvPair(model.blocks[0]).eval()(x), model.blocks[0][:6](x), rtol=0, atol=0)


def test_original_compilation_reproduces_retained_features_without_source_mutation():
    _, model = fixture()
    before = deepcopy(model.state_dict())
    groups = original_quartets(4)[1:3]
    pair, description = compile_pair(model, 0, groups, groups)
    ids = packing_indices(groups, 4)
    x = torch.randn(3, 4, 16, 16)
    with torch.no_grad():
        source_h = torch.relu(model.blocks[0][1](model.blocks[0][0](x)))[:, ids]
        compact_h = torch.relu(pair.bn1(pair.conv1(x)))
        torch.testing.assert_close(source_h, compact_h, rtol=1e-6, atol=1e-6)
    assert description['producer_projection_sse'] == 0
    assert pair.conv1.out_channels == pair.conv2.in_channels == 2
    for name, value in before.items():
        assert torch.equal(value, model.state_dict()[name])


def test_fixed_scalar_set_and_scientific_parameter_count():
    _, model = fixture()
    with pytest.raises(AssertionError, match='Retained scalar set'):
        compile_pair(model, 0, original_quartets(4)[2:], original_quartets(4)[:2])
    config = {'quaternion_widths': [8, 16, 32], 'real_widths': [16, 32, 64], 'dropout': [.1, .2, .3]}
    source, compact = make_compact(config), make_compact(config, [4, 16, 16])
    assert sum(p.numel() for p in source.parameters()) == 73898
    assert sum(p.numel() for p in compact.parameters()) == 44794
    assert compact.blocks[0][3].out_channels == source.blocks[0][3].out_channels
    assert compact.blocks[2][3].out_channels == source.blocks[2][3].out_channels


def test_saved_physically_mixed_model_roundtrip(tmp_path):
    config, source = fixture()
    groups = [list(q) for q in original_quartets(4)[:2]]
    groups[0][0], groups[1][2] = groups[1][2], groups[0][0]
    pair, _ = compile_pair(source, 0, groups)
    install_pair(source, 0, pair)
    path = tmp_path/'model.pt'
    save_model(source, path)
    replica = load_model(path, config, 'cpu')
    x = torch.randn(3, 4, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(source(x), replica(x), rtol=0, atol=0)


def test_producer_input_capture_uses_source_prefix_and_cleans_stop_hook():
    _, model = fixture()
    images = torch.randint(2048, (5, 4, 16, 16), dtype=torch.int16)
    mean, std = torch.zeros(1, 4, 1, 1), torch.ones(1, 4, 1, 1)
    values = producer_inputs(model, 2, images, mean, std, 2)
    with torch.no_grad():
        expected = model.blocks[1](model.blocks[0](prepare_batch(images, mean, std)))
    torch.testing.assert_close(values, expected, rtol=1e-5, atol=1e-6)
    calls = []
    def stop():
        calls.append(True)
        if len(calls) == 2:
            raise InterruptedError('technical STOP')
    with pytest.raises(InterruptedError):
        producer_inputs(model, 2, images, mean, std, 2, stop)
    assert not model.blocks[2][0]._forward_pre_hooks


def test_dense_inverse_permutation_control_with_free_scalar_roles():
    _, model = fixture()
    pair = ConvPair(model.blocks[0]).eval()
    groups = [list(q) for q in original_quartets(4)[:2]]
    groups[0][0], groups[1][2] = groups[1][2], groups[0][0]
    with torch.no_grad():
        h = torch.relu(pair.bn1(pair.conv1(torch.randn(3, 4, 8, 8))))
    assert dense_permutation_check(pair, h, [0, 1], groups) < 1e-8
