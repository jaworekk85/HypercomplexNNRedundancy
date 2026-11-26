"""Fresh deterministic FP32 references after detecting archived TF32 features."""
from pathlib import Path

import torch

from scripts.resnet_component_reassembly_pilot import ROOT, read, write, checked_pack
from src.cnn_training import seed_everything
from src.compression.component_reassembly import packing_indices
from src.compression.grouping_initialization_control import consumer_targets
from src.compression.grouping_mechanism import raw, q_moments, permute_moments, relative_permutation, solve
from src.compression.grouping_statistics_control import collect_consumer
from src.trainable_map_repair import atomic_save
from src.data.cifar_cached import file_sha256


def enforce_fp32(seed):
    seed_everything(seed)
    if (torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.benchmark or not torch.are_deterministic_algorithms_enabled()):
        raise AssertionError('FP32 deterministic flags not enforced')


def fresh_references(config, source, seed, tensors, mean, std, stop, state):
    generated = []
    for site, index in enumerate(config['blocks']):
        stop()
        if torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32:
            raise AssertionError('Fresh references require TF32 disabled')
        folder = ROOT/config['local_study']/f'seed_{seed}'/f'site_{site}'
        previous = ROOT/config['frozen_assignments']/f'seed_{seed}'/f'site_{site}'
        original_assignment = read(previous/'assignments.json')
        assignment = {**original_assignment, 'groups': {case: original_assignment['groups'][case] for case in config['cases']}}
        if (folder/'assignments.json').exists() and read(folder/'assignments.json') != assignment:
            raise ValueError('FP32 fixed assignments changed')
        write(folder/'assignments.json', assignment)
        packs = {}
        for split in ('fit', 'check_calibration'):
            state(stage='fresh_fp32_native_features', seed=seed, block=index, split=split)
            packs[split] = checked_pack(folder/'native_trained'/(split+'.pt'), folder/'native_trained'/(split+'.json'),
                lambda split=split: collect_consumer(source, f'blocks.{index}.3', tensors[split], mean, std,
                                                     config['batch_size'], stop))
            probe = collect_consumer(source, f'blocks.{index}.3', tensors[split][:config['batch_size']],
                                     mean, std, config['batch_size'], stop)
            for key in ('features', 'targets'):
                torch.testing.assert_close(probe[key], packs[split][key][:len(probe[key])], rtol=0, atol=0)
        path = folder/'native__trained'/'result.json'
        if path.exists():
            result = read(path)
            if file_sha256(ROOT/result['coefficient_bank']) != result['coefficient_sha256']:
                raise ValueError('Fresh ideal coefficients changed')
            generated.append(result)
            continue
        width = source.blocks[index][3].in_channels
        retained = assignment['groups']['original']
        ids = packing_indices(retained, width)
        state(stage='fresh_fp32_exact_moments', seed=seed, block=index)
        moments = {s: raw(p['features'][:, ids], p['targets']) for s, p in packs.items()}
        gram, _ = q_moments(moments['fit'], source.blocks[index][3].out_channels)
        penalty = config['ridge_alpha']*float(gram.diag().mean())
        shape = (4, source.blocks[index][3].out_channels, len(retained), 3, 3)
        rows, kernels = [], []
        for case in config['cases']:
            stop()
            permutation = relative_permutation(assignment['groups'][case], assignment['kept'], width)
            kernel, row = solve(permute_moments(moments['fit'], permutation),
                                permute_moments(moments['check_calibration'], permutation), shape, penalty)
            rows.append({'case': case, **row})
            kernels.append(kernel)
        bank = path.parent/'coefficients.pt'
        atomic_save({'kernels': kernels, 'cases': config['cases']}, bank)
        restored = torch.load(bank, map_location='cpu', weights_only=True)['kernels']
        for actual, expected in zip(restored, kernels):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        result = {'dataset': 'eurosat', 'seed': seed, 'site': site, 'cell': 'native__trained_fp32',
                  'rows': rows, 'coefficient_bank': bank.relative_to(ROOT).as_posix(),
                  'coefficient_sha256': file_sha256(bank), 'new_exact_feature_fits': len(rows),
                  'tf32_disabled': True, 'frozen_masks_preserved': True}
        write(path, result)
        generated.append(result)
        print(f'FP32 REFERENCES seed{seed} block{index}: {len(rows)} exact-feature fits', flush=True)
    return generated
