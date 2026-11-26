"""Prepare, freeze, then run matched 30-epoch compact grouping trajectories."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

from scripts.run_activation_queue import lock
from scripts.resnet_component_reassembly_pilot import (read, write, now, make_model,
    load_data, Paused)
from scripts.grouping_mechanism_study import parent_folder
from scripts.grouping_bn_accuracy_control import reconstruct
from scripts.eurosat_grouping_replication import load_model as load_eurosat
from src.compression.eurosat_grouping import load_classification_data
from src.cnn_training import seed_everything
from src.data.cifar_cached import file_sha256
from src.grouping_repair_curve import (preserved_rng, measured, trajectory, save_snapshot,
                                       discarded_update_audit, paired_gaps)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/grouping_repair_curve.json'
NEW = [CONFIG, 'scripts/grouping_repair_curve.py', 'src/grouping_repair_curve.py',
       'tests_torch/test_grouping_repair_curve.py']


def initial_model(config, dataset, seed, case, device):
    baseline = read(ROOT/config['baselines'][dataset])
    previous = ROOT/config['predecessors'][dataset]/f'seed_{seed}'
    if dataset == 'eurosat':
        return load_eurosat(previous/'models'/case/'native.pt', baseline, device)
    source = make_model(baseline).to(device).eval()
    source.load_state_dict(torch.load(parent_folder(seed)/'model.pt', map_location=device, weights_only=True)['model'])
    return reconstruct(source, previous, case, [0, 8], device)


def historical(config, dataset, seed, case):
    folder = ROOT/config['predecessors'][dataset]/f'seed_{seed}'/'models'/case
    row = read(folder/'result.json')
    values = torch.load(folder/'logits.pt', map_location='cpu', weights_only=True)
    return row['before'], values['before']


def data(config, dataset, device):
    baseline = read(ROOT/config['baselines'][dataset])
    if dataset == 'cifar10':
        return load_data(baseline, device)
    folder = ROOT/baseline['data']
    return load_classification_data(folder, read(folder/'data.json'), device)


def register(config, freeze=True):
    sources, inputs = {}, {}
    for dataset in config['datasets']:
        predecessor = config['predecessors'][dataset]
        path = predecessor+'/protocol.json'
        previous = read(ROOT/path)
        for bank, inherited in ((sources, previous['source_hashes']), (inputs, previous['input_hashes'])):
            for name, digest in inherited.items():
                if name in bank and bank[name] != digest:
                    raise ValueError('Conflicting predecessor hashes: '+name)
                bank[name] = digest
        inputs[path] = file_sha256(ROOT/path)
        for seed in config['seeds']:
            for case in config['cases']:
                folder = f'{predecessor}/seed_{seed}/models/{case}'
                row = read(ROOT/folder/'result.json')
                expected = row['artifacts']['logits.pt'] if dataset == 'eurosat' else row['logits']['sha256']
                if file_sha256(ROOT/folder/'logits.pt') != expected:
                    raise ValueError('Historical logits changed: '+folder)
                paths = [folder+'/result.json', folder+'/logits.pt']
                if dataset == 'eurosat':
                    if file_sha256(ROOT/folder/'native.pt') != row['artifacts']['native.pt']:
                        raise ValueError('Native EuroSAT model changed')
                    paths.append(folder+'/native.pt')
                else:
                    for block in (0, 8):
                        local = f'{predecessor}/seed_{seed}/block_{block}/{case}'
                        record = read(ROOT/local/'result.json')
                        if file_sha256(ROOT/local/'block.pt') != record['checkpoint']['sha256']:
                            raise ValueError('Native CIFAR block changed')
                        paths += [local+'/result.json', local+'/block.pt']
                inputs.update({p: file_sha256(ROOT/p) for p in paths})
    for path, digest in {**sources, **inputs}.items():
        if file_sha256(ROOT/path) != digest:
            raise ValueError('Frozen dependency changed: '+path)
    # The CIFAR reconstruction helper is an imported predecessor addition.
    sources['scripts/grouping_bn_accuracy_control.py'] = file_sha256(ROOT/'scripts/grouping_bn_accuracy_control.py')
    sources.update({p: file_sha256(ROOT/p) for p in NEW})
    definition = {'config': config, 'source_hashes': sources, 'input_hashes': inputs,
                  'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                                  'numpy': np.__version__, 'cuda': torch.version.cuda},
                  'official_test_evaluated': False, 'validation_previously_explored': True}
    output = ROOT/config['output']
    path = output/'protocol.json'
    if path.exists() and read(path) != definition:
        raise ValueError('Repair-curve protocol changed')
    if freeze and not path.exists():
        write(path, definition)
    lines = ['# Matched grouping repair trajectories', '', '## Hypothesis', '', config['hypothesis']]
    for key in ('scope', 'schedule', 'primary', 'data', 'checks', 'resume'):
        lines += ['', '## '+key.capitalize(), '', config[key]]
    lines += ['', 'Prior art: '+', '.join(config['prior_art'])+'; canonical references.bib.', '']
    if freeze:
        (output/'PROTOCOL.md').write_text('\n'.join(lines), encoding='utf-8')
    return definition


def report(config, output, status):
    rows = [read(p) for p in sorted(output.glob('*/seed_*/models/*/result.json'))]
    gaps = paired_gaps(rows)
    averages = []
    for dataset in config['datasets']:
        for epoch in config['snapshots']:
            for case in config['cases'][1:]:
                subset = [r for r in gaps if r['dataset'] == dataset and r['epoch'] == epoch and r['case'] == case]
                if subset:
                    averages.append({'dataset': dataset, 'epoch': epoch, 'case': case, 'source_seeds': len(subset),
                        'validation_gap_pp': sum(r['validation']['original_minus_mixed_accuracy_pp'] for r in subset)/len(subset),
                        'validation_ce_delta': sum(r['validation']['mixed_minus_original_ce'] for r in subset)/len(subset)})
    value = {'status': status, 'completed_models': len(rows), 'expected_models': 18,
             'results': rows, 'paired_gaps': gaps, 'descriptive_means': averages,
             'official_test_evaluated': False, 'updated_utc': now()}
    write(output/'summary.json', value)
    lines = ['# Matched grouping repair trajectories', '', 'Status: '+status, '',
             '## Measured validation results', '',
             '| Dataset | Seed | Assignment | Epoch | Accuracy (%) | CE | Successful / attempted |',
             '|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        for s in r['snapshots']:
            m, counts = s['metrics']['validation'], s['optimizer_counts']
            lines.append(f"| {r['dataset']} | {r['seed']} | {r['case']} | {s['epoch']} | {100*m['accuracy']:.2f} | {m['cross_entropy']:.6f} | {counts['successful']}/{counts['attempted']} |")
    lines += ['', '## Interpretation boundaries', '', config['primary'], '', config['schedule'], '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return value


def smoke(config):
    from tests_torch.test_grouping_repair_curve import tiny_fixture
    model, reload_model, tensors, labels, mean, std, repair_config = tiny_fixture()
    output = ROOT/'.cache'/('repair_curve_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    row = trajectory(model, reload_model, tensors, labels, mean, std, repair_config, 19, output, [0, 1, 2])
    again = trajectory(reload_model(output/'epoch_000'/'model.pt'), reload_model, tensors, labels,
                       mean, std, repair_config, 19, output, [0, 1, 2])
    if row != again:
        raise AssertionError('Completed synthetic resume changed records')
    return {'status': 'passed', 'folder': output.relative_to(ROOT).as_posix(), 'epochs': 2}


def prepare(config):
    output = ROOT/config['output']
    record = {'started_utc': now(), 'scientific_training': False}
    try:
        tests = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', NEW[-1]],
                               cwd=ROOT, capture_output=True, text=True)
        record['tests'] = {'returncode': tests.returncode, 'stdout': tests.stdout, 'stderr': tests.stderr}
        print(tests.stdout, flush=True)
        if tests.returncode:
            raise RuntimeError('Repair-curve tests failed')
        with lock('compression_curve_runner.lock'):
            record['smoke'] = smoke(config)
            torch.set_num_threads(config['cpu_threads'])
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable')
            # Validate predecessors now; register new sources after read-only replays
            # and discarded-copy update checks, before any scientific repair SGD.
            definition = register(config, freeze=False)
            record['initial_replays'], record['cuda_audits'] = [], {}
            for dataset in config['datasets']:
                seed_everything(config['seeds'][0])
                tensors, labels, mean, std = data(config, dataset, torch.device(config['device']))
                for seed in config['seeds']:
                    for case in config['cases']:
                        seed_everything(seed)
                        model = initial_model(config, dataset, seed, case, mean.device)
                        if sum(p.numel() for p in model.parameters()) != config['expected_parameters'][dataset]:
                            raise AssertionError('Compact parameter count changed')
                        def reload_model(path):
                            replica = initial_model(config, dataset, seed, case, mean.device)
                            replica.load_state_dict(torch.load(path, map_location=mean.device, weights_only=True)['model'])
                            return replica.eval()
                        scores, logits = historical(config, dataset, seed, case)
                        folder = output/dataset/f'seed_{seed}'/'models'/case/'epoch_000'
                        save_snapshot(model, reload_model, tensors, labels, mean, std, config['repair']['batch_size'],
                                      folder, 0, {'attempted': 0, 'successful': 0}, scores, logits)
                        record['initial_replays'].append({'dataset': dataset, 'seed': seed, 'case': case,
                                                          'predecessor_logits_exact': True})
                        if seed == config['seeds'][0] and case == 'free_100_0':
                            record['cuda_audits'][dataset] = discarded_update_audit(model, tensors, labels, mean, std)
                        print(f'INITIAL REPLAY {dataset} seed{seed} {case}: passed', flush=True)
                        del model, logits
                del tensors, labels, mean, std
                gc.collect()
                torch.cuda.empty_cache()
            if len(record['initial_replays']) != 18:
                raise AssertionError('Missing initial replays')
            if register(config) != definition:
                raise ValueError('Sources changed during preparation')
            record.update(status='passed', source_hashes=definition['source_hashes'], updated_utc=now())
            write(output/'readiness.json', record)
            write(output/'state.json', {'status': 'ready_not_started', 'initial_models_replayed': 18, 'updated_utc': now()})
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', updated_utc=now())
        write(output/'state.json', {'status': 'prepare_failed', 'error': record['error'], 'updated_utc': now()})
        raise
    finally:
        path = output/'development_checks.json'
        history = read(path) if path.exists() else []
        history.append(record)
        write(path, history)
    print('READY: 18 initial models replayed; no new scientific SGD yet.', flush=True)


def run(config):
    output = ROOT/config['output']
    with lock('compression_curve_runner.lock'):
        definition = register(config)
        ready = read(output/'readiness.json')
        if ready['status'] != 'passed' or ready['source_hashes'] != definition['source_hashes']:
            raise ValueError('Readiness does not cover frozen sources')
        def stop():
            if (output/'STOP').exists():
                raise Paused()
        def state(**fields):
            write(output/'state.json', {'status': 'running', 'pid': os.getpid(), **fields, 'updated_utc': now()})
        try:
            for dataset in config['datasets']:
                stop()
                seed_everything(config['seeds'][0])
                state(dataset=dataset, stage='loading_data')
                tensors, labels, mean, std = data(config, dataset, torch.device(config['device']))
                for seed in config['seeds']:
                    for case in config['cases']:
                        stop()
                        seed_everything(seed)
                        model = initial_model(config, dataset, seed, case, mean.device)
                        def reload_model(path):
                            replica = initial_model(config, dataset, seed, case, mean.device)
                            replica.load_state_dict(torch.load(path, map_location=mean.device, weights_only=True)['model'])
                            return replica.eval()
                        def progress(row, counts):
                            state(dataset=dataset, seed=seed, case=case, stage='repair', epoch=row['epoch'],
                                  validation=row['validation'], optimizer_counts=dict(counts), epoch_seconds=row['epoch_seconds'])
                            print(f"REPAIR {dataset} seed{seed} {case}: {row['epoch']}/30 val={100*row['validation']['accuracy']:.2f}% ({row['epoch_seconds']:.1f}s)", flush=True)
                        scores, logits = historical(config, dataset, seed, case)
                        state(dataset=dataset, seed=seed, case=case, stage='initial_replay')
                        folder = output/dataset/f'seed_{seed}'/'models'/case
                        row = trajectory(model, reload_model, tensors, labels, mean, std, config['repair'], seed,
                                         folder, config['snapshots'], output/'STOP', progress, scores, logits)
                        if row is None:
                            raise Paused()
                        if row['real_trainable_parameters'] != config['expected_parameters'][dataset]:
                            raise AssertionError('Parameter count changed during repair')
                        write(folder/'result.json', {**row, 'dataset': dataset, 'case': case})
                        report(config, output, 'running')
                        del model, logits, row
                        gc.collect()
                        torch.cuda.empty_cache()
                del tensors, labels, mean, std
                gc.collect()
                torch.cuda.empty_cache()
            summary = report(config, output, 'completed')
            if summary['completed_models'] != 18:
                raise AssertionError('Incomplete repair trajectories')
            register(config)
            checked = 0
            for path in output.glob('*/seed_*/models/*/epoch_*/result.json'):
                record = read(path)
                for name, digest in record['artifacts'].items():
                    if file_sha256(path.parent/name) != digest:
                        raise ValueError('Final snapshot changed')
                    checked += 1
            write(output/'completion_audit.json', {'status': 'passed', 'completed_models': 18,
                'snapshots': 90, 'artifact_hashes_checked': checked,
                'frozen_source_input_hashes_checked': len({**definition['source_hashes'], **definition['input_hashes']}),
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
