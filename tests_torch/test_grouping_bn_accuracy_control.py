from copy import deepcopy

import pytest
import torch
from torch import nn

from scripts.grouping_bn_accuracy_control import (
    reconstruct, assert_bn_only, grouping_contrasts, save_block, write,
)
from scripts.resnet_component_reassembly_pilot import make_model
from src.compression.resnet_reassembly import compile_block, original_quartets


def test_reconstruction_uses_saved_physical_blocks_and_preserves_source(tmp_path):
    torch.manual_seed(13)
    source = make_model({'real_widths': [4, 4, 4], 'quaternion_widths': [2, 2, 2], 'classes': 2}).eval()
    state = deepcopy(source.state_dict())
    expected = deepcopy(source)
    for index in (0, 8):
        groups = [original_quartets(2)[1]]
        block, _ = compile_block(source.blocks[index], groups)
        with torch.no_grad():
            block.conv2.kernel.add_(.02)
        expected.blocks[index] = block
        folder = tmp_path/f'block_{index}'/'case'
        digest = save_block(block, groups, folder/'block.pt')
        write(folder/'result.json', {'checkpoint': digest})
    rebuilt = reconstruct(source, tmp_path, 'case', [0, 8], 'cpu')
    for name, value in expected.state_dict().items():
        assert torch.equal(value, rebuilt.state_dict()[name])
    for name, value in state.items():
        assert torch.equal(value, source.state_dict()[name])
    with (tmp_path/'block_0'/'case'/'block.pt').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='Artifact changed'):
        reconstruct(source, tmp_path, 'case', [0, 8], 'cpu')


def test_bn_only_audit_rejects_weight_and_non_bn_buffer_changes():
    source = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4)).eval()
    source.register_buffer('unrelated', torch.zeros(1))
    changed = deepcopy(source)
    changed[1].running_mean.add_(1)
    assert assert_bn_only(source, changed)['changed_bn_buffers'] == ['1.running_mean']
    changed.unrelated.add_(1)
    with pytest.raises(AssertionError, match='Non-BN buffer'):
        assert_bn_only(source, changed)
    changed = deepcopy(source)
    with torch.no_grad():
        changed[1].weight.add_(1)
    with pytest.raises(AssertionError, match='trainable'):
        assert_bn_only(source, changed)


def test_primary_contrast_uses_matched_original_gain_and_keeps_negative_results():
    def row(case, scores):
        return {'seed': 19, 'case': case,
                **{stage: {s: {'accuracy': score, 'cross_entropy': 1.}
                            for s in ('validation', 'check_calibration')}
                   for stage, score in zip(('native', 'bn_only', 'historical_repair'), scores)}}
    rows = [row('parent', [.95, .96, .96]), row('original', [.8, .9, .92]),
            row('role_100_0', [.7, .85, .9]), row('free_100_0', [.7, .75, .89])]
    contrasts = {r['case']: r['validation'] for r in grouping_contrasts(rows)}
    assert len(contrasts) == 2
    assert contrasts['role_100_0']['bn_gap_reduction_pp'] == pytest.approx(5.)
    assert contrasts['free_100_0']['bn_gap_reduction_pp'] == pytest.approx(-5.)
    assert contrasts['free_100_0']['gap_pp']['historical_repair'] == pytest.approx(3.)
