from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from src.compression.component_selection import (complete_moves, validate, shortlist,
    projection_distances, starts, search)
from src.compression.resnet_reassembly import original_quartets, compile_block
from src.models.cifar_resnet import BasicBlock


def test_all_neighbors_are_complete_injective_and_include_scalar_set_changes():
    groups = original_quartets(4)[:2]
    moves = complete_moves(groups, 4, 'component')
    selected = {s for q in groups for s in q}
    assert all(moves[key] for key in ('unit', 'replace', 'swap'))
    assert len(moves['replace']) == 8*8
    for category, candidates in moves.items():
        for q in candidates:
            validate(q, 4, 2, 'component')
            assert len({s for group in q for s in group}) == 8
            if category == 'replace':
                assert len(selected-{s for group in q for s in group}) == 1
    assert set(complete_moves(groups, 4, 'unit')['unit']) <= set(moves['unit'])
    assert any(any(s//4 != r for q in candidate for r, s in enumerate(q)) for candidate in moves['swap'])


def test_whole_unit_neighbors_are_exhaustive_and_component_policy_keeps_them_legal():
    groups = original_quartets(4)[:2]
    moves = complete_moves(groups, 4, 'unit')
    assert len(moves['unit']) == 2*2
    assert not moves['replace'] and not moves['swap']
    for candidate in moves['unit']:
        validate(candidate, 4, 2, 'unit')
    with pytest.raises(ValueError, match='non-intact'):
        validate(complete_moves(groups, 4, 'component')['replace'][0], 4, 2, 'unit')


def test_projection_shortlist_does_not_reserve_all_slots_for_intact_groups():
    block = BasicBlock('quaternion', 2, 4).eval()
    groups = original_quartets(4)[:2]
    candidates, counts = shortlist(groups, 4, 'component', projection_distances(block), 12, 19)
    pools = complete_moves(groups, 4, 'component')
    assert len(candidates) <= 12
    assert all(counts[k] for k in pools)
    assert all(set(pools[k]).intersection(candidates) for k in pools)
    assert shortlist(groups, 4, 'component', projection_distances(block), 12, 19)[0] == candidates


def test_search_can_select_components_from_outside_the_original_retained_set():
    block = BasicBlock('quaternion', 2, 4).eval()
    baseline = tuple(original_quartets(4)[:2])
    target = complete_moves(baseline, 4, 'component')['replace'][0]
    def objective(groups):
        return {'score': 0. if groups == target else 1.}
    best, cache, traces = search([baseline, target], 4, 2, 'component', projection_distances(block),
                                  objective, {'rounds': 1, 'per_round': 12, 'tolerance': 1e-12}, 19)
    assert best == target
    assert objective(best)['score'] < objective(baseline)['score']


def test_energy_and_saliency_starts_use_complete_fixed_size_assignments():
    block = BasicBlock('quaternion', 2, 4).eval()
    features = torch.rand(8, 16, 4, 4)
    values = starts(block, features, 2, 19)
    assert len(values['unit']) == len(values['component']) == 3
    for policy in ('unit', 'component'):
        for groups in values[policy]:
            validate(groups, 4, 2, policy)


def full_smoke(config, output=None):
    from scripts.component_selection_study import (ROOT, execute_seed, source_pair,
        pack_paths, source_model, make_model, make_compact, install)
    from scripts.resnet_component_reassembly_pilot import write, collect_source
    from src.compression.grouping_statistics_control import collect_consumer
    from src.compression.eurosat_grouping import producer_inputs
    from src.trainable_map_repair import atomic_save
    from src.data.cifar_cached import file_sha256
    torch.set_num_threads(2)
    output = output or ROOT/'.cache'/('component_selection_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    tiny = deepcopy(config)
    tiny.update(output=str(output), seeds=[19], device='cpu', batch_size=2,
                expected_parameters={}, predecessors={}, baselines={})
    tiny['search'].update(rounds=1, per_round=3, inner_images=4, outer_images=4)
    tiny['repair'].update(epochs=2, batch_size=4, mixed_precision=False)
    tiny['snapshots'] = [0, 1, 2]
    base = {'real_widths': [4, 4, 4], 'quaternion_widths': [2, 2, 2], 'classes': 2, 'dropout': [0., 0., 0.]}
    completed = []
    for dataset in tiny['datasets']:
        baseline_path = output/(dataset+'_baseline.json')
        write(baseline_path, base)
        tiny['baselines'][dataset] = str(baseline_path)
        tiny['predecessors'][dataset] = str(output/'references'/dataset)
        torch.manual_seed(19)
        source = (make_model(base) if dataset == 'cifar10' else make_compact(base)).eval()
        channels = 3 if dataset == 'cifar10' else 4
        tensors = {s: torch.randint(256, (n, channels, 16, 16), dtype=torch.uint8)
                   for s, n in [('train', 8), ('validation', 4), ('fit', 8), ('check_calibration', 4)]}
        labels = {s: torch.arange(len(x)) % 2 for s, x in tensors.items()}
        mean, std = torch.zeros(1, channels, 1, 1), torch.ones(1, channels, 1, 1)
        compact = deepcopy(source)
        for site, index in enumerate(tiny['blocks'][dataset]):
            for split, files in pack_paths(tiny, dataset, 19, site, index).items():
                if dataset == 'cifar10':
                    pack = collect_source(source, index, tensors[split], mean, std, 2, lambda: None)
                else:
                    pack = collect_consumer(source, f'blocks.{index}.3', tensors[split], mean, std, 2)
                    inputs = producer_inputs(source, index, tensors[split], mean, std, 2)
                    path = Path(files['inputs'])
                    atomic_save(inputs, path)
                    write(path.with_suffix('.json'), {'sha256': file_sha256(path)})
                path = Path(files['native'])
                atomic_save(pack, path)
                write(path.with_suffix('.json'), {'sha256': file_sha256(path)})
            block = source_pair(source, dataset, index)
            pair, _ = compile_block(block, original_quartets(2)[:1])
            install(compact, dataset, index, pair)
        tiny['expected_parameters'][dataset] = sum(p.numel() for p in compact.parameters())
        execute_seed(tiny, dataset, source, 19, tensors, labels, mean, std, output, lambda: None, lambda **kw: None)
        from scripts.resnet_component_reassembly_pilot import read
        before = [read(output/dataset/'seed_19/models'/case/'result.json') for case in tiny['cases']]
        execute_seed(tiny, dataset, source, 19, tensors, labels, mean, std, output, lambda: None, lambda **kw: None)
        after = [read(output/dataset/'seed_19/models'/case/'result.json') for case in tiny['cases']]
        assert before == after
        completed += before
    assert len(completed) == 6
    return {'status': 'passed', 'models': 6, 'datasets': tiny['datasets'], 'output': str(output),
            'complete_model_resume_exact': True}


def test_full_synthetic_selection_compilation_repair_and_resume(tmp_path):
    import json
    config = json.loads((Path(__file__).resolve().parents[1]/'configs/component_selection_study.json').read_text())
    assert full_smoke(config, tmp_path)['models'] == 6
