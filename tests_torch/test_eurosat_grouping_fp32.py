from copy import deepcopy

import torch

from scripts.resnet_component_reassembly_pilot import write
from src.compression import eurosat_grouping_fp32 as fp32
from src.compression.eurosat_grouping import make_compact
from src.compression.resnet_reassembly import original_quartets


def test_fp32_enforcement_disables_tf32_and_enables_determinism():
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    fp32.enforce_fp32(19)
    assert not torch.backends.cudnn.allow_tf32
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.benchmark
    assert torch.are_deterministic_algorithms_enabled()


def test_fresh_reference_generation_preserves_masks_source_and_completed_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(fp32, 'ROOT', tmp_path)
    fp32.enforce_fp32(19)
    model = make_compact({'quaternion_widths': [4, 4, 4], 'real_widths': [8, 8, 8], 'dropout': [.1, .2, .3]}).eval()
    original = original_quartets(4)[:2]
    role = [list(q) for q in original]
    role[0][1], role[1][1] = role[1][1], role[0][1]
    free = [list(q) for q in original]
    free[0][0], free[1][1] = free[1][1], free[0][0]
    cases = ['original', 'role_100_0', 'free_100_0']
    assignment = {'kept': [0, 1], 'groups': dict(zip(cases, [original, role, free])), 'source_width': 4}
    write(tmp_path/'old/seed_19/site_0/assignments.json', assignment)
    config = {'blocks': [0], 'cases': cases, 'local_study': 'fresh', 'frozen_assignments': 'old',
              'batch_size': 2, 'ridge_alpha': .001}
    tensors = {s: torch.randint(256, (4, 4, 16, 16), dtype=torch.int16) for s in ('fit', 'check_calibration')}
    mean, std = torch.zeros(1, 4, 1, 1), torch.ones(1, 4, 1, 1)
    before = deepcopy(model.state_dict())
    args = (config, model, 19, tensors, mean, std, lambda: None, lambda **kw: None)
    first, second = fp32.fresh_references(*args), fp32.fresh_references(*args)
    assert first == second and len(first) == 1
    assert first[0]['new_exact_feature_fits'] == 3
    assert first[0]['tf32_disabled'] and first[0]['frozen_masks_preserved']
    assert [r['case'] for r in first[0]['rows']] == cases
    for key, value in before.items():
        assert torch.equal(value, model.state_dict()[key])
