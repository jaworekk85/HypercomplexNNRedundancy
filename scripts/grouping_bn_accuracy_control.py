"""BN-only intervention on saved compact models; prepare before --run."""
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
from torch import nn

from scripts.run_activation_queue import lock
from scripts.resnet_component_reassembly_pilot import (
    ROOT, read, write, now, make_model, save_model, load_model, load_data,
    assert_replay, Paused,
)
from scripts.grouping_mechanism_study import parent_folder, restore_block, save_block
from src.cnn_training import evaluate, first_update_audit, prepare_batch, seed_everything
from src.compression.grouping_statistics_control import recalibrate_bn, parameter_hash
from src.compression.resnet_reassembly import compile_block, original_quartets
from src.data.cifar_cached import file_sha256
from src.trainable_map_repair import atomic_save

CONFIG = 'configs/grouping_bn_accuracy_control.json'
NEW = [CONFIG, 'scripts/grouping_bn_accuracy_control.py',
       'tests_torch/test_grouping_bn_accuracy_control.py']


def reconstruct(source, folder, case, blocks, device):
    result = deepcopy(source).eval()
    for index in blocks:
        local = folder/f'block_{index}'/case
        row = read(local/'result.json')
        result.blocks[index] = restore_block(source.blocks[index], local/'block.pt',
                                             row['checkpoint'], device)
    return result


def assert_bn_only(before, after):
    if parameter_hash(before) != parameter_hash(after):
        raise AssertionError('BN-only intervention changed trainable parameters')
    allowed = {name+'.'+field for name, layer in before.named_modules()
               if isinstance(layer, nn.BatchNorm2d)
               for field in ('running_mean', 'running_var', 'num_batches_tracked')}
    previous, current = dict(before.named_buffers()), dict(after.named_buffers())
    if previous.keys() != current.keys():
        raise AssertionError('Buffer structure changed')
    changed = []
    for name, value in previous.items():
        if not torch.equal(value, current[name]):
            if name not in allowed:
                raise AssertionError(f'Non-BN buffer changed: {name}')
            changed.append(name)
    return {'trainable_parameters_unchanged': True, 'non_bn_buffers_unchanged': True,
            'changed_bn_buffers': changed, 'parameter_sha256': parameter_hash(before)}


def grouping_contrasts(rows):
    contrasts = []
    for seed in sorted({r['seed'] for r in rows}):
        available = {r['case']: r for r in rows if r['seed'] == seed}
        if 'original' not in available:
            continue
        original = available['original']
        for case, mixed in available.items():
            if case in ('original', 'parent'):
                continue
            splits = {}
            for split in ('validation', 'check_calibration'):
                gaps = {stage: 100*(original[stage][split]['accuracy'] - mixed[stage][split]['accuracy'])
                        for stage in ('native', 'bn_only', 'historical_repair')}
                splits[split] = {'gap_pp': gaps,
                                'bn_gap_reduction_pp': gaps['native'] - gaps['bn_only']}
            contrasts.append({'seed': seed, 'case': case, **splits})
    return contrasts


def register(config):
    previous = read(ROOT/config['parent_protocol'])
    statistics = read(ROOT/config['statistics_protocol'])
    sources = {**previous['source_hashes'], **statistics['source_hashes']}
    inputs = dict(previous['input_hashes'])
    paths = [config['parent_protocol'], config['statistics_protocol']]
    baseline = read(ROOT/config['baseline_config'])
    paths += [baseline['split_source'], baseline['data_statistics']]
    for seed in config['seeds']:
        for index in config['blocks']:
            for case in config['cases']:
                local = f"{config['grouping_study']}/seed_{seed}/block_{index}/{case}"
                row = read(ROOT/local/'result.json')
                if file_sha256(ROOT/local/'block.pt') != row['checkpoint']['sha256']:
                    raise ValueError(f'Physical block changed: {local}')
                paths += [local+'/result.json', local+'/block.pt']
        for case in config['cases']:
            folder = f"{config['grouping_study']}/seed_{seed}/models/{case}"
            row = read(ROOT/folder/'result.json')
            for name, key in (('model.pt', 'checkpoint'), ('logits.pt', 'logits')):
                if file_sha256(ROOT/folder/name) != row[key]['sha256']:
                    raise ValueError(f'Historical artifact changed: {folder}/{name}')
                paths.append(folder+'/'+name)
            paths.append(folder+'/result.json')
    inputs.update({p: file_sha256(ROOT/p) for p in paths})
    for path, digest in {**sources, **inputs}.items():
        if file_sha256(ROOT/path) != digest:
            raise ValueError(f'Frozen dependency changed: {path}')
    sources.update({p: file_sha256(ROOT/p) for p in NEW})
    definition = {'config': config, 'source_hashes': sources, 'input_hashes': inputs,
                  'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                                  'numpy': np.__version__, 'cuda': torch.version.cuda},
                  'official_test_evaluated': False, 'validation_previously_explored': True}
    output = ROOT/config['output']
    protocol = output/'protocol.json'
    if protocol.exists() and read(protocol) != definition:
        raise ValueError('BN accuracy-control protocol changed')
    if not protocol.exists():
        write(protocol, definition)
    lines = ['# BN-only accuracy control', '', '## Hypothesis', '', config['hypothesis']]
    for key in ('scope', 'data', 'intervention', 'primary', 'secondary', 'checks', 'resume'):
        lines += ['', '## '+key.capitalize(), '', config[key]]
    lines += ['', 'Prior art: '+', '.join(config['prior_art'])+'; canonical references.bib.', '']
    (output/'PROTOCOL.md').write_text('\n'.join(lines), encoding='utf-8')
    return definition


def evaluate_splits(model, tensors, targets, mean, std, batch, stop):
    scores, logits = {}, {}
    for split in ('validation', 'check_calibration'):
        stop()
        scores[split], values = evaluate(model, tensors[split], targets[split], mean, std, batch)
        logits[split] = torch.from_numpy(values)
    return scores, logits


def model_case(config, baseline, model, historical, historical_logits, tensors,
               targets, mean, std, bn_images, folder, seed, case, stop, progress):
    started = time.perf_counter()
    stop()
    native, native_logits = evaluate_splits(model, tensors, targets, mean, std,
                                             config['evaluation_batch_size'], stop)
    assert_replay(historical['before'], native)
    if historical_logits is not None:
        for split in native_logits:
            torch.testing.assert_close(native_logits[split], historical_logits['before'][split], rtol=0, atol=0)
            if not torch.equal(targets[split].cpu(), historical_logits['labels'][split]):
                raise AssertionError('Historical label order changed')
    native_sha = save_model(model, folder/'native.pt')
    calibrated, bn_audit = recalibrate_bn(model, bn_images, mean, std, config['bn_batch_size'], stop,
                                          lambda **fields: progress(stage='bn_recalibration', **fields))
    immutability = assert_bn_only(model, calibrated)
    bn_scores, bn_logits = evaluate_splits(calibrated, tensors, targets, mean, std,
                                          config['evaluation_batch_size'], stop)
    bn_sha = save_model(calibrated, folder/'bn.pt')
    replica = load_model(folder/'bn.pt', baseline, mean.device)
    replay_scores, replay_logits = evaluate_splits(replica, tensors, targets, mean, std,
                                                   config['evaluation_batch_size'], stop)
    assert_replay(bn_scores, replay_scores)
    for split in bn_logits:
        torch.testing.assert_close(bn_logits[split], replay_logits[split], rtol=0, atol=0)
    logits = {'native': native_logits, 'bn_only': bn_logits,
              'labels': {s: targets[s].cpu() for s in native_logits}}
    atomic_save(logits, folder/'logits.pt')
    write(folder/'bn_audit.json', {'bn': bn_audit, 'immutability': immutability})
    artifacts = {name: file_sha256(folder/name) for name in ('native.pt', 'bn.pt', 'logits.pt', 'bn_audit.json')}
    assert artifacts['native.pt'] == native_sha and artifacts['bn.pt'] == bn_sha
    row = {'seed': seed, 'case': case, 'native': native, 'bn_only': bn_scores,
           'historical_repair': historical.get('after'),
           'real_trainable_parameters': sum(p.numel() for p in model.parameters()),
           'trainable_parameters_unchanged': True, 'non_bn_buffers_unchanged': True,
           'native_predecessor_metrics_replayed': True,
           'native_predecessor_logits_replayed': historical_logits is not None,
           'calibrated_state_logit_replay_passed': True, 'artifacts': artifacts,
           'elapsed_seconds': time.perf_counter()-started, 'official_test_evaluated': False}
    write(folder/'result.json', row)
    return row


def verified_completed(folder):
    row = read(folder/'result.json')
    for name, digest in row['artifacts'].items():
        if file_sha256(folder/name) != digest:
            raise ValueError(f'Completed artifact changed: {folder/name}')
    return row


def report(output, status, scientific=True):
    rows = [read(path) for path in sorted(output.glob('seed_*/*/result.json'))]
    contrasts = grouping_contrasts(rows)
    aggregates = []
    for case in sorted({r['case'] for r in contrasts}):
        selected = [r for r in contrasts if r['case'] == case]
        aggregates.append({'case': case, 'source_seed_count': len(selected),
                           **{split: {'mean_'+stage+'_gap_pp': average(r[split]['gap_pp'][stage] for r in selected)
                                      for stage in ('native', 'bn_only', 'historical_repair')}
                              for split in ('validation', 'check_calibration')},
                           'mean_validation_bn_gap_reduction_pp': average(r['validation']['bn_gap_reduction_pp'] for r in selected)})
    summary = {'status': status, 'scientific_results': scientific, 'results': rows,
               'comparisons': contrasts, 'aggregates': aggregates,
               'completed_models': len(rows), 'official_test_evaluated': False, 'updated_utc': now()}
    write(output/'summary.json', summary)
    lines = ['# BN-only full-network accuracy control', '', 'Status: '+status, '',
             '## Hypothesis', '', 'BN-only recalibration may reduce the accuracy cost of mixing groups.', '',
             '## Measured results', '', 'All accuracies below are percentages. Repair is the historical final epoch3.', '',
             '| Seed | Model | Native val | BN-only val | Repair val | BN-only val CE |',
             '|---|---|---:|---:|---:|---:|']
    for r in rows:
        repair = r['historical_repair']
        repair_text = '-' if repair is None else f"{100*repair['validation']['accuracy']:.2f}"
        lines.append(f"| {r['seed']} | {r['case']} | {100*r['native']['validation']['accuracy']:.2f} | "
                     f"{100*r['bn_only']['validation']['accuracy']:.2f} | {repair_text} | {r['bn_only']['validation']['cross_entropy']:.6f} |")
    lines += ['', '## Paired primary contrasts', '',
              'Gap = original accuracy minus mixed accuracy, in percentage points. Positive reduction means BN closes the gap.', '',
              '| Seed | Mixed case | Native gap | BN-only gap | BN gap reduction | Historical repair gap |',
              '|---|---|---:|---:|---:|---:|']
    for r in contrasts:
        v = r['validation']
        lines.append(f"| {r['seed']} | {r['case']} | {v['gap_pp']['native']:.2f} | {v['gap_pp']['bn_only']:.2f} | "
                     f"{v['bn_gap_reduction_pp']:.2f} | {v['gap_pp']['historical_repair']:.2f} |")
    lines += ['', '## Interpretation limits', '',
              'Exploratory paired intervention on previously inspected validation/CHECK splits. Three model seeds; '
              'no significance, equivalence, gain or novelty claim. BN-only changes running statistics, never weights/affine. '
              'Historical repair also changes BN and weights and did not start from this intervention. '
              'No official test, new supervised training or further automatic study.', '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def execute(config, baseline, source_loader, tensors, targets, mean, std, bn_ids, output, scientific=True):
    output.mkdir(parents=True, exist_ok=True)
    bn_images = tensors['train'][bn_ids.to(tensors['train'].device)]
    def stop():
        if (output/'STOP').exists():
            raise Paused()
    try:
        for seed in config['seeds']:
            source, parent_before = source_loader(seed)
            for case in ['parent']+config['cases']:
                stop()
                folder = output/f'seed_{seed}'/case
                if (folder/'result.json').exists():
                    verified_completed(folder)
                    continue
                if case == 'parent':
                    model, historical, logits = source, {'before': parent_before}, None
                else:
                    previous = ROOT/config['grouping_study']/f'seed_{seed}'
                    model = reconstruct(source, previous, case, config['blocks'], mean.device)
                    historical = read(previous/'models'/case/'result.json')
                    logits = torch.load(previous/'models'/case/'logits.pt', map_location='cpu', weights_only=True)
                expected = config.get('expected_parent_parameters', 271594) if case == 'parent' else config.get('expected_compact_parameters', 232266)
                if sum(p.numel() for p in model.parameters()) != expected:
                    raise AssertionError('Unexpected physical parameter count')
                def progress(**fields):
                    write(output/'state.json', {'status': 'running', 'pid': os.getpid(), 'seed': seed,
                                                 'case': case, **fields, 'updated_utc': now()})
                progress(stage='native_replay')
                row = model_case(config, baseline, model, historical, logits, tensors, targets,
                                 mean, std, bn_images, folder, seed, case, stop, progress)
                print(f"DONE seed{seed} {case}: native={100*row['native']['validation']['accuracy']:.2f}% "
                      f"BN={100*row['bn_only']['validation']['accuracy']:.2f}% ({row['elapsed_seconds']:.1f}s)", flush=True)
                report(output, 'running', scientific)
                del model, historical, logits
                gc.collect()
            del source
        result = report(output, 'completed', scientific)
        if result['completed_models'] != len(config['seeds'])*(1+len(config['cases'])):
            raise AssertionError('Incomplete model count')
        artifacts_checked = 0
        for path in output.glob('seed_*/*/result.json'):
            artifacts_checked += len(verified_completed(path.parent)['artifacts'])
        write(output/'completion_audit.json', {'status': 'passed', 'artifact_hashes_checked': artifacts_checked,
                                               'completed_models': result['completed_models'], 'updated_utc': now()})
        write(output/'state.json', {'status': 'completed', 'completed_models': result['completed_models'], 'updated_utc': now()})
        return result
    except Paused:
        report(output, 'paused', scientific)
        write(output/'state.json', {'status': 'paused', 'updated_utc': now()})
    except BaseException as error:
        report(output, 'failed', scientific)
        write(output/'state.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}', 'updated_utc': now()})
        raise


def smoke(config):
    output = ROOT/'.cache'/('bn_accuracy_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    previous = output/'predecessor'
    tiny = deepcopy(config)
    tiny.update(seeds=[19], cases=['original'], grouping_study=previous.relative_to(ROOT).as_posix(),
                bn_batch_size=3, evaluation_batch_size=2)
    baseline = read(ROOT/config['baseline_config'])
    baseline.update(quaternion_widths=[2, 2, 2], classes=2)
    seed_everything(19)
    source = make_model(baseline).eval()
    compact = deepcopy(source)
    for index in tiny['blocks']:
        groups = [original_quartets(2)[0]]
        compact.blocks[index], _ = compile_block(source.blocks[index], groups)
        folder = previous/'seed_19'/f'block_{index}'/'original'
        digest = save_block(compact.blocks[index], groups, folder/'block.pt')
        write(folder/'result.json', {'checkpoint': digest})
    tiny['expected_parent_parameters'] = sum(p.numel() for p in source.parameters())
    tiny['expected_compact_parameters'] = sum(p.numel() for p in compact.parameters())
    tensors = {s: torch.randint(256, (n, 3, 8, 8), dtype=torch.uint8)
               for s, n in [('train', 8), ('validation', 4), ('check_calibration', 4)]}
    targets = {s: torch.arange(len(x)) % 2 for s, x in tensors.items()}
    mean, std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
    parent_scores, _ = evaluate_splits(source, tensors, targets, mean, std, 2, lambda: None)
    scores, logits = evaluate_splits(compact, tensors, targets, mean, std, 2, lambda: None)
    folder = previous/'seed_19'/'models'/'original'
    write(folder/'result.json', {'before': scores, 'after': scores})
    atomic_save({'before': logits, 'labels': {s: targets[s] for s in logits}}, folder/'logits.pt')
    args = (tiny, baseline, lambda seed: (deepcopy(source), parent_scores), tensors, targets,
            mean, std, torch.arange(6), output)
    first = execute(*args, scientific=False)
    second = execute(*args, scientific=False)
    if first['results'] != second['results']:
        raise AssertionError('Completed resume changed results')
    return {'status': 'passed', 'output': output.relative_to(ROOT).as_posix(), 'models': 2,
            'completed_resume_passed': True}


def prepare(config):
    output = ROOT/config['output']
    record = {'started_utc': now(), 'scientific_results': False}
    try:
        tests = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q',
                                'tests_torch/test_grouping_statistics_control.py',
                                'tests_torch/test_grouping_bn_accuracy_control.py'],
                               cwd=ROOT, capture_output=True, text=True)
        record['tests'] = {'returncode': tests.returncode, 'stdout': tests.stdout, 'stderr': tests.stderr}
        print(tests.stdout, flush=True)
        if tests.returncode:
            raise RuntimeError('BN accuracy-control checks failed')
        with lock('compression_curve_runner.lock'):
            record['smoke'] = smoke(config)
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable')
            seed_everything(17019)
            baseline = read(ROOT/config['baseline_config'])
            widths = [w for w in baseline['quaternion_widths'] for _ in range(3)]
            for index in config['blocks']:
                widths[index] //= 2
            model = make_model(baseline, widths).cuda()
            record['discarded_gpu_update_audit'] = first_update_audit(
                model, torch.randn(8, 3, 32, 32, device='cuda'),
                torch.arange(8, device='cuda') % baseline['classes'],
                torch.optim.SGD(model.parameters(), lr=.001))
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
    print('READY: BN-only full-model accuracy control registered', flush=True)


def run(config):
    output = ROOT/config['output']
    with lock('compression_curve_runner.lock'):
        definition = register(config)
        ready = read(output/'readiness.json')
        if ready['status'] != 'passed' or ready['source_hashes'] != definition['source_hashes']:
            raise ValueError('Readiness does not cover current sources')
        baseline = read(ROOT/config['baseline_config'])
        seed_everything(config['bn_subset_seed'])
        device = torch.device(config['device'])
        tensors, targets, mean, std = load_data(baseline, device)
        bn_ids = torch.randperm(len(tensors['train']), generator=torch.Generator().manual_seed(config['bn_subset_seed']))[:config['bn_train_images']]
        with np.load(ROOT/baseline['split_source']) as split:
            absolute = split['train'][bn_ids.numpy()]
            assert not set(absolute.tolist()).intersection(split['validation'].tolist()+split['calibration'].tolist())
        subset = {'train_positions': bn_ids.tolist(), 'absolute_indices': absolute.tolist(),
                  'seed': config['bn_subset_seed'], 'train_only': True, 'labels_used': False}
        if (output/'bn_subset.json').exists() and read(output/'bn_subset.json') != subset:
            raise ValueError('Calibration subset changed')
        write(output/'bn_subset.json', subset)
        def sources(seed):
            seed_everything(seed)
            model = make_model(baseline).to(device).eval()
            folder = parent_folder(seed)
            model.load_state_dict(torch.load(folder/'model.pt', map_location=device, weights_only=True)['model'])
            return model, {'validation': read(folder/'metrics.json')['selected_validation']}
        result = execute(config, baseline, sources, tensors, targets, mean, std, bn_ids, output)
        if result is not None:
            register(config)
            audit = read(output/'completion_audit.json')
            audit.update(frozen_source_input_hashes_checked=len({**definition['source_hashes'], **definition['input_hashes']}),
                         bn_subset_sha256=file_sha256(output/'bn_subset.json'), all_trainable_parameters_unchanged=True)
            write(output/'completion_audit.json', audit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    config = read(ROOT/CONFIG)
    torch.set_num_threads(config['cpu_threads'])
    run(config) if args.run else prepare(config)
