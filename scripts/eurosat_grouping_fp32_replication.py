"""Corrected version: fresh FP32 references, unchanged frozen nine-model design."""
import argparse
from copy import deepcopy
import gc
import os
import platform
import subprocess
import sys

import numpy as np
import torch

from scripts import eurosat_grouping_replication as base
from scripts.run_activation_queue import lock
from scripts.resnet_component_reassembly_pilot import ROOT, read, write, now, metrics, assert_replay, Paused
from src.cnn_training import first_update_audit
from src.compression.eurosat_grouping import make_compact, load_classification_data
from src.compression.eurosat_grouping_fp32 import enforce_fp32, fresh_references
from src.data.cifar_cached import file_sha256

CONFIG = 'configs/eurosat_grouping_fp32_replication.json'
NEW = [CONFIG, 'scripts/eurosat_grouping_fp32_replication.py', 'src/compression/eurosat_grouping_fp32.py',
       'tests_torch/test_eurosat_grouping_fp32.py']


def effective_config():
    new = read(ROOT/CONFIG)
    config = {**read(ROOT/new['base_config']), **new}
    config['scope'] = ('Fresh deterministic FP32 references and18new exact-feature fits, followed by the original18physical pair fits/nine-model design. '
                       + config['scope'])
    return config


def register(config):
    previous = read(ROOT/config['parent_protocol'])
    sources, inputs = dict(previous['source_hashes']), dict(previous['input_hashes'])
    for path, digest in {**sources, **inputs}.items():
        if file_sha256(ROOT/path) != digest:
            raise ValueError('Frozen dependency changed: '+path)
    inputs.update({p: file_sha256(ROOT/p) for p in
                   [config['parent_protocol'], 'results/eurosat_grouping_replication/precision_diagnostic.json']})
    sources.update({p: file_sha256(ROOT/p) for p in NEW})
    definition = {'config': config, 'source_hashes': sources, 'input_hashes': inputs,
                  'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                                  'numpy': np.__version__, 'cuda': torch.version.cuda},
                  'official_test_evaluated': False, 'validation_previously_explored': True}
    output = ROOT/config['output']
    if (output/'protocol.json').exists() and read(output/'protocol.json') != definition:
        raise ValueError('Corrected EuroSAT protocol changed')
    if not (output/'protocol.json').exists():
        write(output/'protocol.json', definition)
    lines = ['# EuroSAT grouping replication with fresh FP32 references', '', '## Hypothesis', '', config['hypothesis']]
    for key in ('correction', 'precision', 'scope', 'data', 'realization', 'full_models', 'primary', 'checks', 'additional_checks', 'interpretation'):
        lines += ['', '## '+key.capitalize(), '', config[key]]
    lines += ['', 'Prior art: '+', '.join(config['prior_art'])+'; canonical references.bib.', '']
    (output/'PROTOCOL.md').write_text('\n'.join(lines), encoding='utf-8')
    return definition


def finish_report(config, output, status):
    result = base.report(config, output, status)
    references = [read(p) for p in sorted((ROOT/config['local_study']).glob('seed_*/site_*/native__trained/result.json'))]
    result.update(new_exact_feature_fits=sum(r['new_exact_feature_fits'] for r in references),
                  reference_precision='deterministic FP32, TF32 disabled',
                  historical_tf32_errors_reused=False)
    write(output/'summary.json', result)
    path = output/'REPORT.md'
    text = path.read_text(encoding='utf-8').replace('Exact-feature check MSE (reused)', 'Exact-feature check MSE (fresh FP32)')
    text += ('\n## Precision correction\n\n'+config['correction']+'\n\n'
             'Exact-feature rows were newly fitted once in this study and then reused by the physical-fit stage. '
             'The base local-row field ideal_reused describes that within-study reuse, not reuse of historical TF32 errors.\n')
    path.write_text(text, encoding='utf-8')
    return result


def prepare(config):
    output = ROOT/config['output']
    record = {'started_utc': now(), 'scientific_results': False}
    try:
        tests = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q',
                                'tests_torch/test_eurosat_grouping.py', 'tests_torch/test_eurosat_grouping_fp32.py'],
                               cwd=ROOT, capture_output=True, text=True)
        record['tests'] = {'returncode': tests.returncode, 'stdout': tests.stdout, 'stderr': tests.stderr}
        print(tests.stdout, flush=True)
        if tests.returncode:
            raise RuntimeError('Fresh FP32 reference tests failed')
        with lock('compression_curve_runner.lock'):
            record['smoke'] = base.smoke(config)
            enforce_fp32(17019)
            baseline = read(ROOT/config['eurosat_config'])
            hidden = list(baseline['quaternion_widths'])
            for index in config['blocks']:
                hidden[index] //= 2
            model = make_compact(baseline, hidden).cuda()
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
    print('READY: fresh-FP32 EuroSAT replication registered', flush=True)


def run(config):
    output = ROOT/config['output']
    with lock('compression_curve_runner.lock'):
        definition = register(config)
        if read(output/'readiness.json')['source_hashes'] != definition['source_hashes']:
            raise ValueError('FP32 readiness mismatch')
        baseline = read(ROOT/config['eurosat_config'])
        device = torch.device(config['device'])
        enforce_fp32(config['seeds'][0])
        tensors, targets, mean, std = load_classification_data(ROOT/baseline['data'], read(ROOT/baseline['data']/'data.json'), device)
        def stop():
            if (output/'STOP').exists():
                raise Paused()
        def state(**fields):
            write(output/'state.json', {'status': 'running', 'pid': os.getpid(), **fields, 'updated_utc': now()})
        try:
            parent_scores = {}
            for seed in config['seeds']:
                stop()
                enforce_fp32(seed)
                folder = ROOT/f'results/eurosat_cnn/seed_{seed}/quaternion'
                source = make_compact(baseline).to(device).eval()
                source.load_state_dict(torch.load(folder/'model.pt', map_location=device, weights_only=True)['model'])
                scores = metrics(source, tensors, targets, mean, std, config['repair']['batch_size'])
                assert_replay({'validation': read(folder/'metrics.json')['selected_validation']}, {'validation': scores['validation']})
                parent_scores[str(seed)] = scores
                write(output/'parent_replays.json', parent_scores)
                fresh_references(config, source, seed, tensors, mean, std, stop, state)
                base.local_pairs(config, source, seed, tensors, mean, std, output, stop, state)
                base.whole_models(config, baseline, source, seed, tensors, targets, mean, std, output, stop, state)
                finish_report(config, output, 'running')
                del source
                gc.collect()
                torch.cuda.empty_cache()
            result = finish_report(config, output, 'completed')
            if result['completed_models'] != 9 or result['completed_local_pairs'] != 18 or result['new_exact_feature_fits'] != 18:
                raise AssertionError('Incomplete FP32 replication')
            register(config)
            # Audit all numerical payloads with recorded checksums, including fresh references.
            checked = {}
            for path in output.rglob('*.json'):
                if path.name == 'launch.json':
                    continue
                row = read(path)
                if isinstance(row, dict) and 'sha256' in row and path.with_suffix('.pt').exists():
                    checked[path.with_suffix('.pt')] = row['sha256']
                if isinstance(row, dict) and 'coefficient_bank' in row:
                    checked[ROOT/row['coefficient_bank']] = row['coefficient_sha256']
                if isinstance(row, dict) and 'artifacts' in row:
                    checked.update({path.parent/n: d for n, d in row['artifacts'].items()})
                if isinstance(row, dict) and isinstance(row.get('checkpoint'), dict):
                    checked[path.parent/'block.pt'] = row['checkpoint']['sha256']
            for path, digest in checked.items():
                if file_sha256(path) != digest:
                    raise ValueError('Final FP32 artifact changed: '+str(path))
            write(output/'completion_audit.json', {'status': 'passed', 'completed_models': 9,
                'new_exact_feature_fits': 18, 'physical_pairs': 18, 'artifact_hashes_checked': len(checked),
                'frozen_source_input_hashes_checked': len({**definition['source_hashes'], **definition['input_hashes']}),
                'updated_utc': now()})
            write(output/'state.json', {'status': 'completed', 'completed_models': 9, 'updated_utc': now()})
        except Paused:
            finish_report(config, output, 'paused')
            write(output/'state.json', {'status': 'paused', 'updated_utc': now()})
        except BaseException as error:
            finish_report(config, output, 'failed')
            write(output/'state.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}', 'updated_utc': now()})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    config = effective_config()
    torch.set_num_threads(config['cpu_threads'])
    run(config) if args.run else prepare(config)
