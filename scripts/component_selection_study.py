"""Prepare/freeze, then compare complete unit versus component selection."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
from itertools import product
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from scripts.resnet_component_reassembly_pilot import (read, write, now, make_model, collect_source,
    checked_pack, assert_replay, Paused)
from scripts.grouping_mechanism_study import parent_folder, save_block, restore_block
from scripts.grouping_exchange_pilot import logits_on
from scripts.grouping_repair_curve import data
from scripts.run_activation_queue import lock
from src.cnn_training import seed_everything
from src.compression.component_reassembly import packing_indices
from src.compression.component_selection import (starts, projection_distances, search, validate)
from src.compression.eurosat_grouping import (ConvPair, install_pair, make_compact, producer_inputs)
from src.compression.grouping_exchange import canonical, fit_split, fit_inner, block_outputs, prediction_kl
from src.compression.grouping_mechanism import raw, q_moments, physical_fit
from src.compression.grouping_statistics_control import collect_consumer
from src.compression.resnet_reassembly import compile_block
from src.data.cifar_cached import file_sha256
from src.grouping_repair_curve import trajectory, discarded_update_audit, measured
from src.trainable_map_repair import atomic_save

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/component_selection_study.json'
NEW = [CONFIG, 'scripts/component_selection_study.py', 'src/compression/component_selection.py',
       'tests_torch/test_component_selection.py']


def source_model(config, dataset, seed, device):
    baseline = read(ROOT/config['baselines'][dataset])
    source = (make_model(baseline) if dataset == 'cifar10' else make_compact(baseline)).to(device).eval()
    folder = parent_folder(seed) if dataset == 'cifar10' else ROOT/f'results/eurosat_cnn/seed_{seed}/quaternion'
    source.load_state_dict(torch.load(folder/'model.pt', map_location=device, weights_only=True)['model'])
    return source


def source_pair(source, dataset, index):
    return source.blocks[index] if dataset == 'cifar10' else ConvPair(source.blocks[index]).eval()


def install(model, dataset, index, pair):
    if dataset == 'cifar10':
        model.blocks[index] = pair
    else:
        install_pair(model, index, pair)


def pack_paths(config, dataset, seed, site, index):
    previous = config['predecessors'][dataset]
    folder = f'{previous}/seed_{seed}/block_{index}'
    if dataset == 'cifar10':
        return {split: {'native': f'{folder}/{split}.pt'} for split in ('fit', 'check_calibration')}
    native = f'{previous}/native_fp32/seed_{seed}/site_{site}/native_trained'
    return {split: {'native': f'{native}/{split}.pt', 'inputs': f'{folder}/{split}_inputs.pt'}
            for split in ('fit', 'check_calibration')}


def load_packs(config, source, dataset, seed, site, index, tensors, mean, std, stop):
    packs = {}
    paths = pack_paths(config, dataset, seed, site, index)
    for split, files in paths.items():
        stop()
        def missing():
            raise FileNotFoundError('Missing registered native activation/input pack')
        native = ROOT/files['native']
        pack = checked_pack(native, native.with_suffix('.json'), missing)
        if dataset == 'eurosat':
            path = ROOT/files['inputs']
            pack = {**pack, 'inputs': checked_pack(path, path.with_suffix('.json'), missing)}
            probe = collect_consumer(source, f'blocks.{index}.3', tensors[split][:config['batch_size']],
                                     mean, std, config['batch_size'], stop)
            inputs = producer_inputs(source, index, tensors[split][:config['batch_size']],
                                     mean, std, config['batch_size'], stop)
            torch.testing.assert_close(inputs, pack['inputs'][:len(inputs)], rtol=0, atol=0)
        else:
            probe = collect_source(source, index, tensors[split][:config['batch_size']],
                                   mean, std, config['batch_size'], stop)
            torch.testing.assert_close(probe['inputs'], pack['inputs'][:len(probe['inputs'])], rtol=0, atol=0)
        for key in ('features', 'targets'):
            torch.testing.assert_close(probe[key], pack[key][:len(probe[key])], rtol=0, atol=0)
        packs[split] = pack
    return packs


def site_search(config, block, pack, inner, outer, folder, seed, stop, progress):
    if (folder/'search.json').exists():
        result = read(folder/'search.json')
        for rows in result['finalists'].values():
            for row in rows:
                if file_sha256(folder/row['file']) != row['sha256']:
                    raise ValueError('Saved finalist changed')
        return result
    width, count = block.conv1.out_channels, block.conv1.out_channels//2
    warm = starts(block, pack['features'][inner], count, seed)
    baseline = warm['baseline']
    ids = packing_indices(baseline, width)
    gram, _ = q_moments(raw(pack['features'][inner][:, ids], pack['targets'][inner]), block.conv2.out_channels)
    penalty = config['ridge_alpha']*float(gram.diag().mean())
    if penalty <= 0:
        raise ValueError('Nonpositive common ridge scale')
    target = block_outputs(block, pack['inputs'][outer], config['device'], config['batch_size'], stop)
    distances = projection_distances(block)
    evaluated = {}
    def fit(groups):
        return fit_inner(block, groups, pack['inputs'][inner], pack['targets'][inner], penalty,
                         config['device'], config['batch_size'], stop)
    def score(groups):
        key = canonical(groups)
        if key not in evaluated:
            candidate, record = fit(key)
            predicted = block_outputs(candidate, pack['inputs'][outer], config['device'], config['batch_size'], stop)
            evaluated[key] = {'score': float((predicted.double()-target.double()).square().mean()), 'fit': record}
            del candidate, predicted
        return evaluated[key]
    score(baseline)
    winners, traces = {'unit_energy': baseline}, {}
    for policy, case in [('unit', 'unit_search'), ('component', 'component_search')]:
        best, cache, trace = search(warm[policy], width, count, policy, distances, score,
                                     config['search'], seed, stop, progress)
        winners[case] = best
        traces[case] = {'traces': trace, 'policy_unique_assignments': len(cache)}
    finalists = {}
    for case in config['cases']:
        groups_list = [baseline]
        if winners[case] != baseline:
            groups_list.append(winners[case])
        finalists[case] = []
        for number, groups in enumerate(groups_list):
            candidate, record = fit(groups)
            name = f'{case}_{number}.pt'
            digest = save_block(candidate, groups, folder/name)['sha256']
            finalists[case].append({'groups': groups, 'score': score(groups)['score'],
                                     'file': name, 'sha256': digest, 'inner_fit': record})
            del candidate
    result = {'baseline': baseline, 'inner_ids': inner.tolist(), 'outer_ids': outer.tolist(),
              'lambda_inner': penalty, 'finalists': finalists, 'searches': traces,
              'evaluations': [{'groups': key, **value} for key, value in evaluated.items()],
              'unique_candidate_fits': len(evaluated), 'finalist_refits': sum(map(len, finalists.values())),
              'whole_unit_completion_reserve': False, 'held_out_check_or_validation_used': False}
    write(folder/'search.json', result)
    return read(folder/'search.json')


def network_selection(config, dataset, source, searches, outer_images, mean, std, folder, stop):
    path = folder/'network_selection.json'
    if path.exists():
        return read(path)
    teacher = logits_on(source, outer_images, mean, std, config['batch_size'], stop)
    selection = {}
    for case in config['cases']:
        rows = []
        for combination in product(*(range(len(s['finalists'][case])) for s in searches)):
            stop()
            model = deepcopy(source)
            for index, result, choice in zip(config['blocks'][dataset], searches, combination):
                record = result['finalists'][case][choice]
                block = source_pair(source, dataset, index)
                pair = restore_block(block, folder/f'block_{index}'/record['file'],
                                     {'sha256': record['sha256']}, config['device'])
                install(model, dataset, index, pair)
            logits = logits_on(model, outer_images, mean, std, config['batch_size'], stop)
            rows.append({'combination': list(combination),
                         'teacher_kl': prediction_kl(teacher, logits, config['temperature'])})
            del model, logits
        selection[case] = {'winner': min(rows, key=lambda r: (r['teacher_kl'], r['combination'])), 'candidates': rows}
    write(path, selection)
    return selection


def state_hash(model):
    from hashlib import sha256
    digest = sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def execute_seed(config, dataset, source, seed, tensors, labels, mean, std, output, stop, state):
    folder = output/dataset/f'seed_{seed}'
    inner, outer = fit_split(len(tensors['fit']), config['search']['inner_images'],
                              config['search']['outer_images'], seed+config['search']['seed_offset'])
    searches = []
    for site, index in enumerate(config['blocks'][dataset]):
        stop()
        block = source_pair(source, dataset, index)
        state(stage='site_search', dataset=dataset, seed=seed, block=index)
        packs = load_packs(config, source, dataset, seed, site, index, tensors, mean, std, stop)
        def progress(**fields):
            state(stage='site_search', dataset=dataset, seed=seed, block=index, **fields)
            if fields['unique_fits'] % 10 == 0:
                print(f"SEARCH {dataset} seed{seed} block{index} {fields['policy']}: {fields['unique_fits']} assignments", flush=True)
        result = site_search(config, block, packs['fit'], inner, outer, folder/f'block_{index}',
                             seed+config['search']['seed_offset']+index*10000, stop, progress)
        searches.append(result)
        del packs, block
        gc.collect()
    selection = network_selection(config, dataset, source, searches, tensors['fit'][outer], mean, std, folder, stop)
    initial_hashes = {}
    for case in config['cases']:
        stop()
        case_folder = folder/'models'/case
        native = case_folder/'native.pt'
        metadata = case_folder/'native.json'
        if native.exists():
            info = read(metadata)
            if file_sha256(native) != info['sha256']:
                raise ValueError('Compact initialization changed')
        else:
            model, local = deepcopy(source), []
            for site, (index, search_result, choice) in enumerate(zip(config['blocks'][dataset], searches,
                                                                    selection[case]['winner']['combination'])):
                state(stage='final_full_fit', dataset=dataset, seed=seed, case=case, block=index)
                groups = search_result['finalists'][case][choice]['groups']
                block = source_pair(source, dataset, index)
                validate(groups, block.conv1.out_channels, block.conv1.out_channels//2,
                         'unit' if case.startswith('unit') else 'component')
                pair, description = compile_block(block, groups)
                pair = pair.to(config['device']).eval()
                packs = load_packs(config, source, dataset, seed, site, index, tensors, mean, std, stop)
                penalty = search_result['lambda_inner']*len(tensors['fit'])/len(inner)
                selected = {s: {**p, 'selected_features': p['features'][:, packing_indices(groups, block.conv1.out_channels)]}
                            for s, p in packs.items()}
                record = physical_fit(pair, selected, penalty, config['device'], config['batch_size'], stop)
                install(model, dataset, index, pair)
                baseline_set = {x for q in search_result['baseline'] for x in q}
                kept_set = {x for q in groups for x in q}
                local.append({'block': index, 'groups': groups, 'description': description, 'physical_fit': record,
                              'retained_scalar_changes_from_energy': len(kept_set-baseline_set)})
                del packs, selected, block
                gc.collect()
            hidden = ([b.conv1.out_channels for b in model.blocks] if dataset == 'cifar10'
                      else [b[0].out_channels for b in model.blocks])
            count = sum(p.numel() for p in model.parameters())
            if count != config['expected_parameters'][dataset]:
                raise AssertionError('Actual compact parameter budget changed')
            atomic_save({'model': {n: v.detach().cpu() for n, v in model.state_dict().items()}, 'hidden_widths': hidden}, native)
            info = {'sha256': file_sha256(native), 'hidden_widths': hidden, 'local': local,
                    'real_trainable_parameters': count, 'initial_state_hash': state_hash(model)}
            write(metadata, info)
            del model
        baseline = read(ROOT/config['baselines'][dataset])
        def reload_model(path):
            model = (make_model(baseline, info['hidden_widths']) if dataset == 'cifar10'
                     else make_compact(baseline, info['hidden_widths'])).to(config['device'])
            model.load_state_dict(torch.load(path, map_location=config['device'], weights_only=True)['model'])
            return model.eval()
        seed_everything(seed)
        model = reload_model(native)
        initial_hashes[case] = info['initial_state_hash']
        def progress_epoch(row, counts):
            state(stage='repair', dataset=dataset, seed=seed, case=case, epoch=row['epoch'],
                  validation=row['validation'], epoch_seconds=row['epoch_seconds'], optimizer_counts=dict(counts))
            print(f"REPAIR {dataset} seed{seed} {case}: {row['epoch']}/{config['repair']['epochs']} val={100*row['validation']['accuracy']:.2f}% ({row['epoch_seconds']:.1f}s)", flush=True)
        result = trajectory(model, reload_model, tensors, labels, mean, std, config['repair'], seed,
                            case_folder, config['snapshots'], output/'STOP', progress_epoch)
        if result is None:
            raise Paused()
        result.update(dataset=dataset, case=case, local=info['local'], initial_state_hash=info['initial_state_hash'],
                      identical_initialization_to=[c for c, h in initial_hashes.items() if c != case and h == info['initial_state_hash']])
        write(case_folder/'result.json', result)
        report(config, output, 'running')
        del model, result
        gc.collect()
        if mean.device.type == 'cuda':
            torch.cuda.empty_cache()


def report(config, output, status):
    rows = [read(p) for p in sorted(output.glob('*/seed_*/models/*/result.json'))]
    lookup = {(r['dataset'], r['seed'], r['case']): r for r in rows}
    comparisons = []
    for (dataset, seed, case), component in lookup.items():
        if case != 'component_search' or (dataset, seed, 'unit_search') not in lookup:
            continue
        units = lookup[(dataset, seed, 'unit_search')]
        for a, b in zip(component['snapshots'], units['snapshots']):
            comparisons.append({'dataset': dataset, 'seed': seed, 'epoch': a['epoch'],
                'component_minus_unit_accuracy_pp': 100*(a['metrics']['validation']['accuracy']-b['metrics']['validation']['accuracy']),
                'component_minus_unit_ce': a['metrics']['validation']['cross_entropy']-b['metrics']['validation']['cross_entropy']})
    result = {'status': status, 'completed_models': len(rows), 'results': rows, 'paired_comparisons': comparisons,
              'official_test_evaluated': False, 'updated_utc': now()}
    write(output/'summary.json', result)
    lines = ['# Component selection versus whole-unit pruning', '', 'Status: '+status, '',
             '## Measured results', '', '| Setting | Seed | Method | Epoch30 accuracy (%) | CE |', '|---|---:|---|---:|---:|']
    for r in rows:
        m = r['snapshots'][-1]['metrics']['validation']
        lines.append(f"| {r['dataset']} | {r['seed']} | {r['case']} | {100*m['accuracy']:.2f} | {m['cross_entropy']:.6f} |")
    lines += ['', '## Interpretation limits', '', config['evaluation'], '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def register(config, freeze=True):
    previous = read(ROOT/config['parent_protocol'])
    sources, inputs = dict(previous['source_hashes']), dict(previous['input_hashes'])
    inputs[config['parent_protocol']] = file_sha256(ROOT/config['parent_protocol'])
    for dataset in config['datasets']:
        for seed in config['seeds']:
            for site, index in enumerate(config['blocks'][dataset]):
                for files in pack_paths(config, dataset, seed, site, index).values():
                    for name in files.values():
                        path = ROOT/name
                        meta = path.with_suffix('.json')
                        if file_sha256(path) != read(meta)['sha256']:
                            raise ValueError('Native pack changed: '+name)
                        inputs[name], inputs[meta.relative_to(ROOT).as_posix()] = file_sha256(path), file_sha256(meta)
    for name, digest in {**sources, **inputs}.items():
        if file_sha256(ROOT/name) != digest:
            raise ValueError('Frozen predecessor changed: '+name)
    sources.update({name: file_sha256(ROOT/name) for name in NEW})
    # Explicitly include imported wrappers not necessarily in the earlier chain.
    for name in ('scripts/grouping_repair_curve.py', 'src/grouping_repair_curve.py', 'scripts/grouping_exchange_pilot.py'):
        sources[name] = file_sha256(ROOT/name)
    definition = {'config': config, 'source_hashes': sources, 'input_hashes': inputs,
                  'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                                  'cuda': torch.version.cuda}, 'official_test_evaluated': False,
                  'validation_previously_explored': True}
    output = ROOT/config['output']
    if (output/'protocol.json').exists() and read(output/'protocol.json') != definition:
        raise ValueError('Registered component-selection protocol changed')
    if freeze:
        write(output/'protocol.json', definition)
        lines = ['# Component-level selection versus whole-unit pruning', '', '## Hypothesis', '', config['hypothesis']]
        for key in ('scope', 'selection', 'scoring', 'evaluation', 'data', 'checks', 'resume'):
            lines += ['', '## '+key.capitalize(), '', config[key]]
        lines += ['', 'Prior art: '+', '.join(config['prior_art'])+'; canonical references.bib.', '']
        (output/'PROTOCOL.md').write_text('\n'.join(lines), encoding='utf-8')
    return definition


def smoke(config):
    from tests_torch.test_component_selection import full_smoke
    return full_smoke(config)


def prepare(config):
    output = ROOT/config['output']
    record = {'started_utc': now(), 'scientific_training': False}
    try:
        checked = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', NEW[-1],
                                  'tests_torch/test_grouping_repair_curve.py'], cwd=ROOT, capture_output=True, text=True)
        record['tests'] = {'returncode': checked.returncode, 'stdout': checked.stdout, 'stderr': checked.stderr}
        print(checked.stdout, flush=True)
        if checked.returncode:
            raise RuntimeError('Component-selection tests failed')
        with lock('compression_curve_runner.lock'):
            record['smoke'] = smoke(config)
            torch.set_num_threads(config['cpu_threads'])
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable')
            definition = register(config, False)
            record['cuda_audits'], record['parent_replays'], record['native_batches'] = {}, [], []
            for dataset in config['datasets']:
                tensors, labels, mean, std = data(config, dataset, torch.device(config['device']))
                for seed in config['seeds']:
                    seed_everything(seed)
                    source = source_model(config, dataset, seed, mean.device)
                    parent = (parent_folder(seed) if dataset == 'cifar10' else ROOT/f'results/eurosat_cnn/seed_{seed}/quaternion')
                    scores, _ = measured(source, tensors, labels, mean, std, config['repair']['batch_size'])
                    expected = read(parent/'metrics.json')['selected_validation']
                    assert_replay({'validation': expected}, scores)
                    record['parent_replays'].append({'dataset': dataset, 'seed': seed, 'validation': expected})
                    for site, index in enumerate(config['blocks'][dataset]):
                        packs = load_packs(config, source, dataset, seed, site, index, tensors, mean, std, lambda: None)
                        record['native_batches'].append({'dataset': dataset, 'seed': seed, 'site': index, 'replay_exact': True})
                        del packs
                    if seed == config['seeds'][0]:
                        model = deepcopy(source)
                        for index in config['blocks'][dataset]:
                            block = source_pair(source, dataset, index)
                            warm = starts(block, torch.randn(8, 4*block.conv1.out_channels, 4, 4),
                                          block.conv1.out_channels//2, seed)
                            pair, _ = compile_block(block, warm['component'][-1])
                            install(model, dataset, index, pair.to(mean.device))
                        if sum(p.numel() for p in model.parameters()) != config['expected_parameters'][dataset]:
                            raise AssertionError('Compact architecture count mismatch')
                        record['cuda_audits'][dataset] = discarded_update_audit(model, tensors, labels, mean, std)
                        del model
                    print(f'PREFLIGHT {dataset} seed{seed}: parent and native references replayed', flush=True)
                    del source
                del tensors, labels, mean, std
                gc.collect()
                torch.cuda.empty_cache()
            if register(config) != definition:
                raise ValueError('Sources changed during preparation')
            record.update(status='passed', source_hashes=definition['source_hashes'], updated_utc=now())
            write(output/'readiness.json', record)
            write(output/'state.json', {'status': 'ready_not_started', 'updated_utc': now()})
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', updated_utc=now())
        write(output/'state.json', {'status': 'prepare_failed', 'error': record['error'], 'updated_utc': now()})
        raise
    finally:
        path = output/'development_checks.json'
        history = read(path) if path.exists() else []
        history.append(record)
        write(path, history)
    print('READY: component-selection comparison frozen before scientific selection/repair.', flush=True)


def run(config):
    output = ROOT/config['output']
    with lock('compression_curve_runner.lock'):
        definition = register(config)
        ready = read(output/'readiness.json')
        if ready['status'] != 'passed' or ready['source_hashes'] != definition['source_hashes']:
            raise ValueError('Checks do not cover the registered sources')
        def stop():
            if (output/'STOP').exists():
                raise Paused()
        def state(**fields):
            write(output/'state.json', {'status': 'running', 'pid': os.getpid(), **fields, 'updated_utc': now()})
        try:
            for dataset in config['datasets']:
                seed_everything(config['seeds'][0])
                tensors, labels, mean, std = data(config, dataset, torch.device(config['device']))
                for seed in config['seeds']:
                    stop()
                    seed_everything(seed)
                    source = source_model(config, dataset, seed, mean.device)
                    execute_seed(config, dataset, source, seed, tensors, labels, mean, std, output, stop, state)
                    del source
                    gc.collect()
                    torch.cuda.empty_cache()
                del tensors, labels, mean, std
                gc.collect()
                torch.cuda.empty_cache()
            result = report(config, output, 'completed')
            if result['completed_models'] != 18:
                raise AssertionError('Incomplete component-selection model count')
            register(config)
            count = 0
            for path in output.glob('*/seed_*/models/*/epoch_*/result.json'):
                for name, digest in read(path)['artifacts'].items():
                    if file_sha256(path.parent/name) != digest:
                        raise ValueError('Completed snapshot artifact changed')
                    count += 1
            if count != 180:
                raise AssertionError('Missing snapshot artifacts')
            write(output/'completion_audit.json', {'status': 'passed', 'completed_models': 18,
                'artifact_hashes_checked': count, 'frozen_source_input_hashes_checked': len({**definition['source_hashes'], **definition['input_hashes']}),
                'updated_utc': now()})
            write(output/'state.json', {'status': 'completed', 'completed_models': 18, 'updated_utc': now()})
        except Paused:
            report(config, output, 'paused')
            write(output/'state.json', {'status': 'paused', 'updated_utc': now()})
        except BaseException as error:
            report(config, output, 'failed')
            write(output/'state.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}', 'updated_utc': now()})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    config = read(ROOT/CONFIG)
    torch.set_num_threads(config['cpu_threads'])
    run(config) if args.run else prepare(config)
