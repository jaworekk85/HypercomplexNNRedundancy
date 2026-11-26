"""Prepare then --run the nine-model EuroSAT physical grouping replication."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import platform
from statistics import mean as average
import subprocess
import sys
import time

import numpy as np
import torch

from scripts.run_activation_queue import lock
from scripts.resnet_component_reassembly_pilot import (ROOT, read, write, now, checked_pack,
    metrics, assert_replay, Paused)
from scripts.grouping_mechanism_study import save_block, restore_block, verified
from src.cnn_training import evaluate, prepare_batch, seed_everything, first_update_audit
from src.compression.component_reassembly import packing_indices
from src.compression.eurosat_grouping import (ConvPair, install_pair, compile_pair, make_compact,
    producer_inputs, load_classification_data)
from src.compression.grouping_mechanism import physical_fit, dense_permutation_check
from src.compression.grouping_statistics_control import collect_consumer
from src.compression.resnet_reassembly import local_mse, original_quartets
from src.data.cifar_cached import file_sha256
from src.trainable_map_repair import atomic_save, repair

CONFIG = 'configs/eurosat_grouping_replication.json'
NEW = [CONFIG, 'scripts/eurosat_grouping_replication.py', 'src/compression/eurosat_grouping.py',
       'tests_torch/test_eurosat_grouping.py']


def save_model(model, path):
    atomic_save({'model': {n: v.detach().cpu() for n, v in model.state_dict().items()},
                 'hidden_widths': [b[0].out_channels for b in model.blocks]}, path)
    return file_sha256(path)


def load_model(path, baseline, device):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    result = make_compact(baseline, payload['hidden_widths']).to(device)
    result.load_state_dict(payload['model'])
    return result.eval()


def register(config):
    previous = read(ROOT/config['parent_protocol'])
    sources, inputs = dict(previous['source_hashes']), dict(previous['input_hashes'])
    paths = [config['parent_protocol'], config['eurosat_config']]
    for seed in config['seeds']:
        base = f'results/eurosat_cnn/seed_{seed}/quaternion'
        if file_sha256(ROOT/base/'model.pt') != previous['input_hashes'][base+'/model.pt']:
            raise ValueError('EuroSAT parent checkpoint changed')
        paths += [base+'/model.pt', base+'/metrics.json']
        for site in range(len(config['blocks'])):
            folder = ROOT/config['local_study']/f'seed_{seed}'/f'site_{site}'
            cell = read(folder/'native__trained'/'result.json')
            if file_sha256(ROOT/cell['coefficient_bank']) != cell['coefficient_sha256']:
                raise ValueError('Frozen native coefficients changed')
            paths += [cell['coefficient_bank']]
            paths += [(folder/name).relative_to(ROOT).as_posix() for name in
                      ('assignments.json', 'native__trained/result.json', 'native_trained/fit.pt',
                       'native_trained/fit.json', 'native_trained/check_calibration.pt',
                       'native_trained/check_calibration.json')]
    inputs.update({p: file_sha256(ROOT/p) for p in paths})
    for path, digest in {**sources, **inputs}.items():
        if file_sha256(ROOT/path) != digest:
            raise ValueError('Frozen dependency changed: '+path)
    sources.update({p: file_sha256(ROOT/p) for p in NEW})
    definition = {'config': config, 'source_hashes': sources, 'input_hashes': inputs,
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'numpy': np.__version__, 'cuda': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None},
        'official_test_evaluated': False, 'validation_previously_explored': True}
    output = ROOT/config['output']
    if (output/'protocol.json').exists() and read(output/'protocol.json') != definition:
        raise ValueError('EuroSAT grouping protocol changed')
    if not (output/'protocol.json').exists():
        write(output/'protocol.json', definition)
    lines = ['# EuroSAT physical grouping replication', '', '## Hypothesis', '', config['hypothesis']]
    for key in ('scope', 'data', 'realization', 'full_models', 'primary', 'checks', 'interpretation'):
        lines += ['', '## '+key.capitalize(), '', config[key]]
    lines += ['', 'Prior art: '+', '.join(config['prior_art'])+'; canonical references.bib.', '']
    (output/'PROTOCOL.md').write_text('\n'.join(lines), encoding='utf-8')
    return definition


@torch.no_grad()
def logits_on(model, images, mean, std, batch, stop):
    values = []
    model.eval()
    for x in images.split(batch):
        stop()
        values.append(model(prepare_batch(x, mean, std)).cpu())
    return torch.cat(values)


def report(config, output, status, scientific=True):
    local = [read(p) for p in sorted(output.glob('seed_*/block_*/*/result.json'))]
    rows = [read(p) for p in sorted(output.glob('seed_*/models/*/result.json'))]
    comparisons = []
    for seed in config['seeds']:
        selected = {r['case']: r for r in rows if r['seed'] == seed}
        if len(selected) != len(config['cases']):
            continue
        original = selected['original']
        for case in config['cases']:
            if case == 'original':
                continue
            mixed = selected[case]
            contrasts = {}
            for split in ('validation', 'check_calibration'):
                contrasts[split] = {
                    'before_gap_pp': 100*(original['before'][split]['accuracy']-mixed['before'][split]['accuracy']),
                    'after_gap_pp': 100*(original['after'][split]['accuracy']-mixed['after'][split]['accuracy']),
                    'after_mixed_minus_original_ce': mixed['after'][split]['cross_entropy']-original['after'][split]['cross_entropy']}
            comparisons.append({'seed': seed, 'case': case, **contrasts})
    aggregates = []
    for case in config['cases']:
        selected = [r for r in comparisons if r['case'] == case]
        if selected:
            aggregates.append({'case': case, 'source_seeds': len(selected),
                **{split: {key: average(r[split][key] for r in selected) for key in selected[0][split]}
                   for split in ('validation', 'check_calibration')}})
    result = {'status': status, 'scientific_results': scientific, 'local_results': local,
              'results': rows, 'comparisons': comparisons, 'aggregates': aggregates,
              'completed_local_pairs': len(local), 'completed_models': len(rows),
              'official_test_evaluated': False, 'updated_utc': now()}
    write(output/'summary.json', result)
    lines = ['# EuroSAT physical grouping replication', '', 'Status: '+status, '',
             '## Hypothesis', '', config['hypothesis'], '', '## Measured full-network results', '',
             '| Seed | Case | Actual trainables | Native val | Final epoch3 val | Final val CE |',
             '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['seed']} | {r['case']} | {r['real_trainable_parameters']} | {100*r['before']['validation']['accuracy']:.2f}% | {100*r['after']['validation']['accuracy']:.2f}% | {r['after']['validation']['cross_entropy']:.6f} |")
    lines += ['', '## Primary paired gaps', '',
              'Positive accuracy gap favors original grouping; CE delta is mixed minus original.', '',
              '| Seed | Case | Native val gap pp | Final val gap pp | Final CE delta |',
              '|---|---|---:|---:|---:|']
    for r in comparisons:
        v = r['validation']
        lines.append(f"| {r['seed']} | {r['case']} | {v['before_gap_pp']:.2f} | {v['after_gap_pp']:.2f} | {v['after_mixed_minus_original_ce']:.6f} |")
    lines += ['', '## Measured local results', '',
              '| Seed | Block | Case | Exact-feature check MSE (reused) | Physical check MSE | Producer projection SSE |',
              '|---|---|---|---:|---:|---:|']
    for r in local:
        lines.append(f"| {r['seed']} | {r['block']} | {r['case']} | {r['ideal']['check_mse']:.6g} | {r['physical']['check_mse']:.6g} | {r['description']['producer_projection_sse']:.6g} |")
    lines += ['', '## Interpretation and limits', '', config['scope'], '', config['data'], '',
              config['full_models'], '', config['interpretation'], '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def local_pairs(config, source, seed, tensors, mean, std, output, stop, state):
    for site, index in enumerate(config['blocks']):
        folder = output/f'seed_{seed}'/f'block_{index}'
        previous = ROOT/config['local_study']/f'seed_{seed}'/f'site_{site}'
        assignment = read(previous/'assignments.json')
        native = read(previous/'native__trained'/'result.json')
        ideal = {r['case']: r for r in native['rows']}
        source_pair = ConvPair(source.blocks[index]).eval()
        packs = {}
        for split in ('fit', 'check_calibration'):
            stop()
            state(stage='source_inputs', seed=seed, block=index, split=split)
            def missing():
                raise FileNotFoundError('Missing frozen native EuroSAT activation bank')
            pack = checked_pack(previous/'native_trained'/(split+'.pt'), previous/'native_trained'/(split+'.json'), missing)
            inputs = checked_pack(folder/(split+'_inputs.pt'), folder/(split+'_inputs.json'),
                lambda: producer_inputs(source, index, tensors[split], mean, std, config['batch_size'], stop))
            probe = collect_consumer(source, f'blocks.{index}.3', tensors[split][:config['batch_size']],
                                     mean, std, config['batch_size'], stop)
            for key in ('features', 'targets'):
                torch.testing.assert_close(probe[key], pack[key][:len(probe[key])], rtol=0, atol=0)
            packs[split] = {'inputs': inputs, 'features': pack['features'], 'targets': pack['targets']}
            del pack, probe
        for case in config['cases']:
            stop()
            target = folder/case
            if (target/'result.json').exists():
                row = read(target/'result.json')
                verified(target/'block.pt', row['checkpoint'])
                continue
            started = time.perf_counter()
            groups = assignment['groups'][case]
            retained = assignment['groups']['original']
            pair, description = compile_pair(source, index, groups, retained)
            pair = pair.to(config['device']).eval()
            ids = packing_indices(groups, source_pair.conv1.out_channels)
            selected = {split: {**pack, 'selected_features': pack['features'][:, ids]}
                        for split, pack in packs.items()}
            state(stage='physical_consumer_fit', seed=seed, block=index, case=case)
            physical = physical_fit(pair, selected, ideal['original']['lambda'], config['device'], config['batch_size'], stop)
            inverse_error = dense_permutation_check(source_pair, packs['fit']['features'], assignment['kept'], groups)
            replay = local_mse(pair, packs['check_calibration'], config['device'], config['batch_size'])
            if abs(replay-physical['check_mse']) > 1e-5*max(1., abs(replay)):
                raise AssertionError('Physical fit did not replay in float32')
            digest = save_block(pair, groups, target/'block.pt')
            replica = restore_block(source_pair, target/'block.pt', digest, config['device'])
            probe_x = packs['check_calibration']['inputs'][:config['batch_size']].to(mean.device)
            with torch.no_grad():
                torch.testing.assert_close(pair(probe_x), replica(probe_x), rtol=0, atol=0)
            row = {'seed': seed, 'block': index, 'site': site, 'case': case, 'kept_units': assignment['kept'],
                   'description': description, 'ideal': ideal[case], 'ideal_reused': True, 'physical': physical,
                   'checkpoint': digest, 'native_source_batch_bitwise_replay': True,
                   'physical_float32_check_mse': replay, 'saved_pair_forward_replay': True,
                   'dense_inverse_permutation_max_abs': inverse_error, 'seconds': time.perf_counter()-started}
            write(target/'result.json', row)
            print(f"LOCAL seed{seed} block{index} {case}: check MSE={physical['check_mse']:.5f} ({row['seconds']:.1f}s)", flush=True)
            report(config, output, 'running', not config.get('synthetic', False))
            del pair, replica, selected
            gc.collect()
        del packs, source_pair
        gc.collect()
        if mean.device.type == 'cuda':
            torch.cuda.empty_cache()


def whole_models(config, baseline, source, seed, tensors, targets, mean, std, output, stop, state):
    for case in config['cases']:
        stop()
        folder = output/f'seed_{seed}'/'models'/case
        if (folder/'result.json').exists():
            row = read(folder/'result.json')
            for name, digest in row['artifacts'].items():
                if file_sha256(folder/name) != digest:
                    raise ValueError('Completed EuroSAT model artifact changed')
            continue
        started = time.perf_counter()
        model = deepcopy(source)
        for index in config['blocks']:
            local = output/f'seed_{seed}'/f'block_{index}'/case
            row = read(local/'result.json')
            pair = restore_block(ConvPair(source.blocks[index]), local/'block.pt', row['checkpoint'], config['device'])
            install_pair(model, index, pair)
        count = sum(p.numel() for p in model.parameters())
        if count != config['expected_compact_parameters']:
            raise AssertionError('Wrong actual EuroSAT compact size')
        state(stage='before_repair', seed=seed, case=case)
        before = metrics(model, tensors, targets, mean, std, config['repair']['batch_size'])
        native_logits = {s: logits_on(model, tensors[s], mean, std, config['repair']['batch_size'], stop) for s in before}
        native_sha = save_model(model, folder/'native.pt')
        probe = deepcopy(model)
        audit = first_update_audit(probe, prepare_batch(tensors['train'][:8], mean, std), targets['train'][:8],
                                    torch.optim.SGD(probe.parameters(), lr=.001))
        write(folder/'discarded_update_audit.json', audit)
        del probe
        def on_epoch(row, counters):
            state(stage='repair', seed=seed, case=case, epoch=row['epoch'], validation=row['validation'], counts=counters)
            print(f"REPAIR seed{seed} {case} {row['epoch']}/{config['repair']['epochs']}: val={100*row['validation']['accuracy']:.2f}% ({row['epoch_seconds']:.1f}s)", flush=True)
        model, history, counters, done = repair(model, tensors, targets, mean, std, config['repair'], seed,
                                                folder/'last.pt', output/'STOP', on_epoch)
        if not done:
            raise Paused()
        after = metrics(model, tensors, targets, mean, std, config['repair']['batch_size'])
        after_logits = {s: logits_on(model, tensors[s], mean, std, config['repair']['batch_size'], stop) for s in after}
        digest = save_model(model, folder/'model.pt')
        replica = load_model(folder/'model.pt', baseline, config['device'])
        assert_replay(after, metrics(replica, tensors, targets, mean, std, config['repair']['batch_size']))
        for split in after:
            torch.testing.assert_close(after_logits[split], logits_on(replica, tensors[split], mean, std,
                                       config['repair']['batch_size'], stop), rtol=0, atol=0)
        native_replica = load_model(folder/'native.pt', baseline, config['device'])
        assert_replay(before, metrics(native_replica, tensors, targets, mean, std, config['repair']['batch_size']))
        for split in before:
            torch.testing.assert_close(native_logits[split], logits_on(native_replica, tensors[split], mean, std,
                                       config['repair']['batch_size'], stop), rtol=0, atol=0)
        atomic_save({'before': native_logits, 'after': after_logits,
                     'labels': {s: targets[s].cpu() for s in after}}, folder/'logits.pt')
        artifacts = {n: file_sha256(folder/n) for n in ('native.pt', 'model.pt', 'logits.pt', 'discarded_update_audit.json')}
        assert artifacts['native.pt'] == native_sha and artifacts['model.pt'] == digest
        row = {'seed': seed, 'case': case, 'before': before, 'after': after, 'history': history,
               'optimizer_counts': counters, 'real_trainable_parameters': count,
               'checkpoint_bytes': (folder/'model.pt').stat().st_size, 'artifacts': artifacts,
               'native_and_final_model_logit_replay_passed': True, 'discarded_update_audit_passed': True,
               'test_evaluated': False, 'seconds': time.perf_counter()-started}
        write(folder/'result.json', row)
        report(config, output, 'running', not config.get('synthetic', False))
        print(f"DONE seed{seed} {case}: {100*before['validation']['accuracy']:.2f}% -> {100*after['validation']['accuracy']:.2f}%", flush=True)
        del model, replica, native_replica, native_logits, after_logits
        gc.collect()


def execute(config, baseline, source_loader, tensors, targets, mean, std, output):
    output.mkdir(parents=True, exist_ok=True)
    def stop():
        if (output/'STOP').exists():
            raise Paused()
    def state(**fields):
        write(output/'state.json', {'status': 'running', 'pid': os.getpid(), **fields, 'updated_utc': now()})
    try:
        for seed in config['seeds']:
            source = source_loader(seed)
            if sum(p.numel() for p in source.parameters()) != config['expected_parent_parameters']:
                raise AssertionError('Wrong parent parameter count')
            local_pairs(config, source, seed, tensors, mean, std, output, stop, state)
            whole_models(config, baseline, source, seed, tensors, targets, mean, std, output, stop, state)
            del source
            gc.collect()
            if mean.device.type == 'cuda':
                torch.cuda.empty_cache()
        result = report(config, output, 'completed', not config.get('synthetic', False))
        if result['completed_models'] != len(config['seeds'])*len(config['cases']):
            raise AssertionError('Incomplete EuroSAT model count')
        write(output/'state.json', {'status': 'completed', 'completed_models': result['completed_models'], 'updated_utc': now()})
        return result
    except Paused:
        report(config, output, 'paused', not config.get('synthetic', False))
        write(output/'state.json', {'status': 'paused', 'updated_utc': now()})
    except BaseException as error:
        report(config, output, 'failed', not config.get('synthetic', False))
        write(output/'state.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}', 'updated_utc': now()})
        raise


def smoke(config):
    output = ROOT/'.cache'/('eurosat_grouping_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    previous = output/'predecessor'
    tiny = deepcopy(config)
    tiny.update(seeds=[19], local_study=previous.relative_to(ROOT).as_posix(), device='cpu', batch_size=2, synthetic=True)
    tiny['repair'].update(epochs=1, batch_size=4, mixed_precision=False)
    baseline = read(ROOT/config['eurosat_config'])
    baseline.update(quaternion_widths=[4, 4, 4], dropout=[.1, .2, .3])
    seed_everything(19)
    source = make_compact(baseline).eval()
    tensors = {s: torch.randint(2048, (n, 4, 16, 16), dtype=torch.int16)
               for s, n in [('train', 8), ('fit', 4), ('validation', 4), ('check_calibration', 4)]}
    targets = {s: torch.arange(len(x)) % 10 for s, x in tensors.items()}
    mean, std = torch.zeros(1, 4, 1, 1), torch.ones(1, 4, 1, 1)
    compact = deepcopy(source)
    for site, index in enumerate(tiny['blocks']):
        folder = previous/'seed_19'/f'site_{site}'
        groups = original_quartets(4)[:2]
        role = [list(q) for q in groups]
        role[0][1], role[1][1] = role[1][1], role[0][1]
        free = [list(q) for q in groups]
        free[0][0], free[1][1] = free[1][1], free[0][0]
        write(folder/'assignments.json', {'kept': [0, 1], 'groups': dict(zip(tiny['cases'], [groups, role, free]))})
        for split in ('fit', 'check_calibration'):
            checked_pack(folder/'native_trained'/(split+'.pt'), folder/'native_trained'/(split+'.json'),
                         lambda split=split: collect_consumer(source, f'blocks.{index}.3', tensors[split], mean, std, 2))
        write(folder/'native__trained'/'result.json', {'rows': [{'case': case, 'lambda': .01, 'check_mse': 0.} for case in tiny['cases']]})
        pair, _ = compile_pair(source, index, groups)
        install_pair(compact, index, pair)
    tiny['expected_parent_parameters'] = sum(p.numel() for p in source.parameters())
    tiny['expected_compact_parameters'] = sum(p.numel() for p in compact.parameters())
    args = (tiny, baseline, lambda seed: deepcopy(source), tensors, targets, mean, std, output)
    first = execute(*args)
    second = execute(*args)
    if first['results'] != second['results'] or first['local_results'] != second['local_results']:
        raise AssertionError('Completed resume changed EuroSAT results')
    return {'status': 'passed', 'output': output.relative_to(ROOT).as_posix(), 'local_pairs': 6,
            'models': 3, 'repair_epochs': 3, 'completed_resume_passed': True}


def prepare(config):
    output = ROOT/config['output']
    record = {'started_utc': now(), 'scientific_results': False}
    try:
        tests = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', 'tests_torch/test_eurosat_grouping.py'],
                               cwd=ROOT, capture_output=True, text=True)
        record['tests'] = {'returncode': tests.returncode, 'stdout': tests.stdout, 'stderr': tests.stderr}
        print(tests.stdout, flush=True)
        if tests.returncode:
            raise RuntimeError('EuroSAT grouping tests failed')
        with lock('compression_curve_runner.lock'):
            record['smoke'] = smoke(config)
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable')
            seed_everything(17019)
            baseline = read(ROOT/config['eurosat_config'])
            hidden = list(baseline['quaternion_widths'])
            for index in config['blocks']:
                hidden[index] //= 2
            model = make_compact(baseline, hidden).cuda()
            if sum(p.numel() for p in model.parameters()) != config['expected_compact_parameters']:
                raise AssertionError('Expected scientific compact count wrong')
            record['discarded_gpu_update_audit'] = first_update_audit(model, torch.randn(8, 4, 64, 64, device='cuda'),
                torch.arange(8, device='cuda') % 10, torch.optim.SGD(model.parameters(), lr=.001))
            del model
            torch.cuda.empty_cache()
            definition = register(config)
            record.update(status='passed', source_hashes=definition['source_hashes'], updated_utc=now())
            write(output/'readiness.json', record)
            write(output/'state.json', {'status': 'ready_not_started', 'updated_utc': now()})
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        history = read(output/'development_checks.json') if (output/'development_checks.json').exists() else []
        history.append(record)
        write(output/'development_checks.json', history)
    print('READY: EuroSAT grouping replication frozen before scientific fitting', flush=True)


def run(config):
    output = ROOT/config['output']
    with lock('compression_curve_runner.lock'):
        definition = register(config)
        ready = read(output/'readiness.json')
        if ready['status'] != 'passed' or ready['source_hashes'] != definition['source_hashes']:
            raise ValueError('Readiness does not cover current EuroSAT sources')
        baseline = read(ROOT/config['eurosat_config'])
        device = torch.device(config['device'])
        seed_everything(config['seeds'][0])
        tensors, targets, mean, std = load_classification_data(ROOT/baseline['data'], read(ROOT/baseline['data']/'data.json'), device)
        parent_scores = {}
        def sources(seed):
            seed_everything(seed)
            folder = ROOT/f'results/eurosat_cnn/seed_{seed}/quaternion'
            model = make_compact(baseline).to(device).eval()
            model.load_state_dict(torch.load(folder/'model.pt', map_location=device, weights_only=True)['model'])
            scores = metrics(model, tensors, targets, mean, std, config['repair']['batch_size'])
            assert_replay({'validation': read(folder/'metrics.json')['selected_validation']}, {'validation': scores['validation']})
            parent_scores[str(seed)] = scores
            write(output/'parent_replays.json', parent_scores)
            return model
        result = execute(config, baseline, sources, tensors, targets, mean, std, output)
        if result is not None:
            register(config)
            count = 0
            for path in output.glob('seed_*/block_*/*/result.json'):
                verified(path.parent/'block.pt', read(path)['checkpoint'])
                count += 1
            for path in output.glob('seed_*/block_*/*_inputs.json'):
                if file_sha256(path.with_suffix('.pt')) != read(path)['sha256']:
                    raise ValueError('Producer input artifact changed')
                count += 1
            for path in output.glob('seed_*/models/*/result.json'):
                for name, digest in read(path)['artifacts'].items():
                    if file_sha256(path.parent/name) != digest:
                        raise ValueError('Model artifact changed')
                    count += 1
            write(output/'completion_audit.json', {'status': 'passed', 'completed_models': result['completed_models'],
                'completed_local_pairs': result['completed_local_pairs'], 'artifact_hashes_checked': count,
                'frozen_source_input_hashes_checked': len({**definition['source_hashes'], **definition['input_hashes']}),
                'updated_utc': now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    config = read(ROOT/CONFIG)
    torch.set_num_threads(config['cpu_threads'])
    run(config) if args.run else prepare(config)
