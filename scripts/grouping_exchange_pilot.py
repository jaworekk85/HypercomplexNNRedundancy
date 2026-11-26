"""Prepare and run bounded fixed-set Hamilton grouping search."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
from itertools import product
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
from scripts.resnet_component_reassembly_pilot import (
    ROOT, read, write, now, make_model, save_model, load_model, load_data,
    collect_source, checked_pack, metrics, assert_replay, Paused,
)
from scripts.grouping_mechanism_study import parent_folder, save_block, restore_block, verified
from src.cnn_training import evaluate, prepare_batch, seed_everything, first_update_audit
from src.compression.grouping_exchange import (
    canonical, fit_split, producer_decoder, fit_inner, block_outputs,
    search_partition, prediction_kl, same_state, validate_partition,
)
from src.compression.grouping_mechanism import physical_fit
from src.compression.resnet_reassembly import compile_block, original_quartets
from src.data.cifar_cached import file_sha256
from src.trainable_map_repair import atomic_save, repair

CONFIG = 'configs/grouping_exchange_pilot.json'
NEW = [CONFIG, 'scripts/grouping_exchange_pilot.py', 'src/compression/grouping_exchange.py',
       'tests_torch/test_grouping_exchange.py']


def register(config):
    previous = read(ROOT/config['parent_protocol'])
    sources, inputs = dict(previous['source_hashes']), dict(previous['input_hashes'])
    paths = [config['parent_protocol']]
    for seed in config['seeds']:
        for index in config['blocks']:
            folder = ROOT/config['grouping_study']/f'seed_{seed}'/f'block_{index}'
            paths += [(folder/name).relative_to(ROOT).as_posix() for name in
                      ('assignments.json', 'fit.pt', 'fit.json', 'check_calibration.pt', 'check_calibration.json')]
            for case in ('original',):
                row = read(folder/case/'result.json')
                verified(folder/case/'block.pt', row['checkpoint'])
                paths += [(folder/case/name).relative_to(ROOT).as_posix() for name in ('result.json', 'block.pt')]
        for case in ('original', 'role_50_0', 'free_50_0', 'role_100_0', 'free_100_0'):
            folder = ROOT/config['grouping_study']/f'seed_{seed}'/'models'/case
            row = read(folder/'result.json')
            verified(folder/'model.pt', row['checkpoint'])
            verified(folder/'logits.pt', row['logits'])
            paths += [(folder/name).relative_to(ROOT).as_posix() for name in ('result.json', 'model.pt', 'logits.pt')]
    inputs.update({p: file_sha256(ROOT/p) for p in paths})
    for path, digest in {**sources, **inputs}.items():
        if file_sha256(ROOT/path) != digest:
            raise ValueError('Frozen dependency changed: '+path)
    sources.update({p: file_sha256(ROOT/p) for p in NEW})
    definition = {'config': config, 'source_hashes': sources, 'input_hashes': inputs,
                  'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                                  'numpy': np.__version__, 'cuda': torch.version.cuda},
                  'test_evaluated': False, 'validation_previously_explored': True}
    output = ROOT/config['output']
    path = output/'protocol.json'
    if path.exists() and read(path) != definition:
        raise ValueError('Registered exchange pilot changed')
    if not path.exists():
        write(path, definition)
    lines = ['# Fixed-set grouping exchange pilot', '', '## Hypothesis', '', config['hypothesis']]
    for key in ('scope', 'selection', 'search', 'global_selection', 'controls', 'decision', 'checks'):
        lines += ['', '## '+key.capitalize(), '', config[key]]
    lines += ['', 'Prior art: '+', '.join(config['prior_art'])+'; canonical references.bib.', '']
    (output/'PROTOCOL.md').write_text('\n'.join(lines), encoding='utf-8')
    return definition


@torch.no_grad()
def logits_on(model, images, mean, std, batch, stop):
    model.eval()
    result = []
    for x in images.split(batch):
        stop()
        result.append(model(prepare_batch(x, mean, std)).cpu())
    return torch.cat(result)


def load_verified_pack(folder, split):
    return checked_pack(folder/(split+'.pt'), folder/(split+'.json'),
                        lambda: (_ for _ in ()).throw(FileNotFoundError('Missing frozen pack')))


def local_search(config, source, seed, index, inner_ids, selection_ids, output, stop, state):
    folder = output/f'seed_{seed}'/f'block_{index}'
    previous = ROOT/config['grouping_study']/f'seed_{seed}'/f'block_{index}'
    source_block = source.blocks[index]
    if (folder/'search.json').exists():
        result = read(folder/'search.json')
        for row in result['finalists']:
            verified(folder/row['checkpoint_file'], row['checkpoint'])
        return result
    started = time.perf_counter()
    pack = load_verified_pack(previous, 'fit')
    assignment = read(previous/'assignments.json')['groups']
    original_groups = canonical(assignment['original'])
    starts = {'original': original_groups, 'free_50_0': canonical(assignment['free_50_0'])}
    penalty = read(previous/'original'/'result.json')['physical']['lambda']*len(inner_ids)/len(pack['inputs'])
    selection_inputs = pack['inputs'][selection_ids]
    target_outputs = block_outputs(source_block, selection_inputs, config['device'], config['batch_size'], stop)
    def evaluate_assignment(groups):
        block, fit = fit_inner(source_block, groups, pack['inputs'][inner_ids], pack['targets'][inner_ids],
                               penalty, config['device'], config['batch_size'], stop)
        predicted = block_outputs(block, selection_inputs, config['device'], config['batch_size'], stop)
        score = float((predicted.double()-target_outputs.double()).square().mean())
        # Keep small state dictionaries; source activation banks are not duplicated.
        return {'score': score, 'fit': fit,
                'model': {n: v.detach().cpu().clone() for n, v in block.state_dict().items()}}
    finalists, cache, traces = search_partition(
        starts, producer_decoder(source_block), evaluate_assignment, config,
        seed+config['search_seed_offset']+index*10000, stop,
        lambda **fields: state(stage='group_search', seed=seed, block=index, **fields))
    rows = []
    for candidate_index, groups in enumerate(finalists):
        block, _ = compile_block(source_block, groups)
        block.load_state_dict(cache[groups]['model'])
        name = f'finalist_{candidate_index}.pt'
        digest = save_block(block, groups, folder/name)
        rows.append({'groups': groups, 'outer_block_mse': cache[groups]['score'], 'inner_fit': cache[groups]['fit'],
                     'checkpoint_file': name, 'checkpoint': digest, 'is_original': groups == original_groups})
    evaluations = [{'groups': groups, 'outer_block_mse': value['score'], 'inner_fit': value['fit']}
                   for groups, value in cache.items()]
    result = {'seed': seed, 'block': index, 'finalists': rows, 'traces': traces, 'evaluations': evaluations,
              'unique_fitted_assignments': len(cache), 'inner_fit_positions': inner_ids.tolist(),
              'outer_selection_positions': selection_ids.tolist(), 'seconds': time.perf_counter()-started,
              'check_or_validation_used': False}
    write(folder/'search.json', result)
    print(f"SEARCH seed{seed} block{index}: {len(cache)} fits, {len(rows)} finalists ({result['seconds']:.1f}s)", flush=True)
    return read(folder/'search.json')


def global_select(config, source, seed, searches, images, mean, std, output, stop, state):
    folder = output/f'seed_{seed}'
    path = folder/'selection.json'
    if path.exists():
        return read(path)
    teacher = logits_on(source, images, mean, std, config['batch_size'], stop)
    rows = []
    for combination in product(*(range(len(s['finalists'])) for s in searches)):
        state(stage='whole_network_selection', seed=seed, combination=list(combination))
        model = deepcopy(source)
        for index, local, candidate in zip(config['blocks'], searches, combination):
            row = local['finalists'][candidate]
            model.blocks[index] = restore_block(source.blocks[index], folder/f'block_{index}'/row['checkpoint_file'],
                                                 row['checkpoint'], config['device'])
        value = prediction_kl(teacher, logits_on(model, images, mean, std, config['batch_size'], stop), config['temperature'])
        rows.append({'combination': list(combination), 'teacher_kl': value,
                     'changed_sites': sum(not local['finalists'][c]['is_original'] for local, c in zip(searches, combination))})
        del model
    winner = min(rows, key=lambda r: (r['teacher_kl'], r['changed_sites'], r['combination']))
    result = {'seed': seed, 'candidates': rows, 'selected': winner, 'fit_only': True,
              'temperature': config['temperature'], 'selection_before_all_fit_refit': True}
    write(path, result)
    return result


def final_model(config, baseline, source, seed, searches, selection, tensors, targets, mean, std,
                output, stop, state):
    folder = output/f'seed_{seed}'
    path = folder/'result.json'
    if path.exists():
        row = read(path)
        for name, digest in row['artifacts'].items():
            if file_sha256(folder/name) != digest:
                raise ValueError('Completed guided artifact changed')
        return row
    started = time.perf_counter()
    original_model, guided = deepcopy(source), deepcopy(source)
    refits = []
    for index, local, choice in zip(config['blocks'], searches, selection['selected']['combination']):
        stop()
        state(stage='final_all_fit_refit', seed=seed, block=index)
        previous = ROOT/config['grouping_study']/f'seed_{seed}'/f'block_{index}'
        packs = {split: load_verified_pack(previous, split) for split in ('fit', 'check_calibration')}
        original_info = read(previous/'original'/'result.json')
        original_groups = original_info['description']['quartets']
        penalty = original_info['physical']['lambda']
        original, _ = compile_block(source.blocks[index], original_groups)
        original = original.to(config['device']).eval()
        physical_fit(original, packs, penalty, config['device'], config['batch_size'], stop)
        historical = restore_block(source.blocks[index], previous/'original'/'block.pt', original_info['checkpoint'], config['device'])
        if not same_state(original, historical):
            raise AssertionError('Original final refit did not reproduce frozen state exactly')
        original_model.blocks[index] = original
        groups = local['finalists'][choice]['groups']
        validate_partition(groups, original_groups, source.blocks[index].conv1.out_channels)
        block, description = compile_block(source.blocks[index], groups)
        block = block.to(config['device']).eval()
        fitted = physical_fit(block, packs, penalty, config['device'], config['batch_size'], stop)
        guided.blocks[index] = block
        digest = save_block(block, groups, folder/f'block_{index}'/'selected.pt')
        refits.append({'block': index, 'groups': groups, 'description': description, 'physical_fit': fitted,
                       'checkpoint': digest, 'original_refit_bitwise_replay': True})
    previous = ROOT/config['grouping_study']/f'seed_{seed}'/'models'/'original'
    control = read(previous/'result.json')
    old_logits = torch.load(previous/'logits.pt', map_location='cpu', weights_only=True)
    native_control = metrics(original_model, tensors, targets, mean, std, config['repair']['batch_size'])
    assert_replay(control['before'], native_control)
    for split in native_control:
        values = logits_on(original_model, tensors[split], mean, std, config['repair']['batch_size'], stop)
        torch.testing.assert_close(values, old_logits['before'][split], rtol=0, atol=0)
    if sum(p.numel() for p in guided.parameters()) != config.get('expected_parameters', 232266):
        raise AssertionError('Wrong guided physical parameter count')
    before = metrics(guided, tensors, targets, mean, std, config['repair']['batch_size'])
    native_identical = same_state(guided, original_model)
    native_sha = save_model(guided, folder/'native.pt')
    if native_identical and not config.get('force_smoke_repair', False):
        guided = load_model(previous/'model.pt', baseline, config['device'])
        after = metrics(guided, tensors, targets, mean, std, config['repair']['batch_size'])
        assert_replay(control['after'], after)
        history, counts, reused = control['history'], control['optimizer_counts'], True
    else:
        probe = deepcopy(guided)
        gradient_audit = first_update_audit(probe, prepare_batch(tensors['train'][:8], mean, std), targets['train'][:8],
                                            torch.optim.SGD(probe.parameters(), lr=.001))
        write(folder/'discarded_update_audit.json', gradient_audit)
        del probe
        def on_epoch(row, counts):
            state(stage='repair', seed=seed, epoch=row['epoch'], validation=row['validation'], counts=counts)
            print(f"REPAIR seed{seed} {row['epoch']}/{config['repair']['epochs']}: val={100*row['validation']['accuracy']:.2f}%", flush=True)
        guided, history, counts, done = repair(guided, tensors, targets, mean, std, config['repair'], seed,
                                               folder/'last.pt', output/'STOP', on_epoch)
        if not done:
            raise Paused()
        after = metrics(guided, tensors, targets, mean, std, config['repair']['batch_size'])
        reused = False
    logits = {'after': {s: logits_on(guided, tensors[s], mean, std, config['repair']['batch_size'], stop) for s in after},
              'labels': {s: targets[s].cpu() for s in after}}
    if reused:
        for split in after:
            torch.testing.assert_close(logits['after'][split], old_logits['after'][split], rtol=0, atol=0)
    digest = save_model(guided, folder/'model.pt')
    replica = load_model(folder/'model.pt', baseline, config['device'])
    assert_replay(after, metrics(replica, tensors, targets, mean, std, config['repair']['batch_size']))
    for split in after:
        torch.testing.assert_close(logits['after'][split], logits_on(replica, tensors[split], mean, std,
                                   config['repair']['batch_size'], stop), rtol=0, atol=0)
    atomic_save(logits, folder/'logits.pt')
    row = {'seed': seed, 'before': before, 'after': after, 'original_before': control['before'],
           'original_after': control['after'], 'history': history, 'optimizer_counts': counts,
           'reused_original_repair': reused, 'native_bitwise_identical_to_original': native_identical,
           'new_successful_updates': 0 if reused else counts['successful'],
           'real_trainable_parameters': sum(p.numel() for p in guided.parameters()), 'refits': refits,
           'validation_gain_pp': 100*(after['validation']['accuracy']-control['after']['validation']['accuracy']),
           'validation_ce_delta': after['validation']['cross_entropy']-control['after']['validation']['cross_entropy'],
           'artifacts': {n: file_sha256(folder/n) for n in ('native.pt', 'model.pt', 'logits.pt')},
           'saved_model_logit_replay_passed': True, 'original_native_replay_passed': True,
           'seconds': time.perf_counter()-started, 'test_evaluated': False}
    assert row['artifacts']['native.pt'] == native_sha and row['artifacts']['model.pt'] == digest
    write(path, row)
    print(f"RESULT seed{seed}: guided={100*after['validation']['accuracy']:.2f}% original={100*control['after']['validation']['accuracy']:.2f}%, reused={reused}", flush=True)
    return row


def report(config, output, status, scientific=True):
    rows = [read(p) for p in sorted(output.glob('seed_*/result.json'))]
    searches = [read(p) for p in sorted(output.glob('seed_*/block_*/search.json'))]
    decision = {'status': 'pending'}
    if len(rows) == len(config['seeds']):
        gain = average(r['validation_gain_pp'] for r in rows)
        ce = average(r['validation_ce_delta'] for r in rows)
        decision = {'status': 'evaluated', 'mean_accuracy_gain_pp': gain, 'mean_ce_delta': ce,
                    'practical_screen_passed': gain >= .5 and ce <= 0, 'significance_claimed': False,
                    'automatic_followup': False}
    result = {'status': status, 'scientific_results': scientific, 'results': rows,
              'completed_local_searches': len(searches),
              'unique_local_fits': sum(s['unique_fitted_assignments'] for s in searches),
              'new_repaired_models': sum(not r['reused_original_repair'] for r in rows),
              'decision': decision, 'updated_utc': now(), 'test_evaluated': False}
    write(output/'summary.json', result)
    lines = ['# Fixed-set grouping exchange pilot', '', 'Status: '+status, '', '## Hypothesis', '', config['hypothesis'],
             '', '## Measured final epoch results', '', '| Seed | Original val | Guided val | Gain pp | CE delta | Repair reused |',
             '|---|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['seed']} | {100*r['original_after']['validation']['accuracy']:.2f} | {100*r['after']['validation']['accuracy']:.2f} | {r['validation_gain_pp']:.2f} | {r['validation_ce_delta']:.6f} | {r['reused_original_repair']} |")
    lines += ['', '## Decision', '', str(decision), '', '## Interpretation limits', '', config['scope'], '',
              config['selection'], '', config['controls'], '',
              'Bounded heuristic, no optimality, significance, equivalence, learned-semantics or novelty claim. '
              'Original groups are allowed; unchanged results must be reported, not forced into mixed groups. '
              'Previous pilot gate stays failed; no automatic further run.', '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def execute(config, baseline, source_loader, tensors, targets, mean, std, output, scientific=True):
    output.mkdir(parents=True, exist_ok=True)
    def stop():
        if (output/'STOP').exists():
            raise Paused()
    def state(**fields):
        write(output/'state.json', {'status': 'running', 'pid': os.getpid(), **fields, 'updated_utc': now()})
    try:
        for seed in config['seeds']:
            stop()
            source = source_loader(seed)
            inner, selection_ids = fit_split(len(tensors['fit']), config['inner_images'], config['selection_images'],
                                             seed+config['split_seed_offset'])
            searches = [local_search(config, source, seed, index, inner, selection_ids, output, stop, state)
                        for index in config['blocks']]
            choice = global_select(config, source, seed, searches, tensors['fit'][selection_ids.to(mean.device)],
                                   mean, std, output, stop, state)
            final_model(config, baseline, source, seed, searches, choice, tensors, targets, mean, std, output, stop, state)
            report(config, output, 'running', scientific)
            del source, searches
            gc.collect()
            if mean.device.type == 'cuda':
                torch.cuda.empty_cache()
        result = report(config, output, 'completed', scientific)
        write(output/'state.json', {'status': 'completed', 'completed_seeds': len(result['results']), 'updated_utc': now()})
        return result
    except Paused:
        report(config, output, 'paused', scientific)
        write(output/'state.json', {'status': 'paused', 'updated_utc': now()})
    except BaseException as error:
        report(config, output, 'failed', scientific)
        write(output/'state.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}', 'updated_utc': now()})
        raise


def smoke(config):
    output = ROOT/'.cache'/('grouping_exchange_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    previous = output/'predecessor'
    tiny = deepcopy(config)
    tiny.update(seeds=[19], grouping_study=previous.relative_to(ROOT).as_posix(), device='cpu',
                inner_images=2, selection_images=2, rounds=1, compatible_proposals=1,
                random_proposals=1, batch_size=2, force_smoke_repair=True)
    tiny['repair'].update(epochs=1, batch_size=4, mixed_precision=False)
    baseline = read(ROOT/config['baseline_config'])
    baseline.update(quaternion_widths=[4, 4, 4], classes=2)
    seed_everything(19)
    source = make_model(baseline).eval()
    tensors = {s: torch.randint(256, (n, 3, 8, 8), dtype=torch.uint8)
               for s, n in [('train', 8), ('fit', 8), ('validation', 4), ('check_calibration', 4)]}
    targets = {s: torch.arange(len(x)) % 2 for s, x in tensors.items()}
    mean, std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
    original = deepcopy(source)
    for index in tiny['blocks']:
        folder = previous/'seed_19'/f'block_{index}'
        packs = {s: checked_pack(folder/(s+'.pt'), folder/(s+'.json'),
                                 lambda s=s: collect_source(source, index, tensors[s], mean, std, 2, lambda: None))
                 for s in ('fit', 'check_calibration')}
        groups = original_quartets(4)[:2]
        alternate = [list(q) for q in groups]
        alternate[0][0], alternate[1][0] = alternate[1][0], alternate[0][0]
        write(folder/'assignments.json', {'groups': {'original': groups, 'free_50_0': alternate}})
        block, description = compile_block(source.blocks[index], groups)
        fitted = physical_fit(block, packs, .01, 'cpu', 2)
        digest = save_block(block, groups, folder/'original'/'block.pt')
        write(folder/'original'/'result.json', {'physical': fitted, 'description': description, 'checkpoint': digest})
        original.blocks[index] = block
    tiny['expected_parameters'] = sum(p.numel() for p in original.parameters())
    scores = metrics(original, tensors, targets, mean, std, 4)
    folder = previous/'seed_19'/'models'/'original'
    save_model(original, folder/'model.pt')
    write(folder/'result.json', {'before': scores, 'after': scores, 'history': [], 'optimizer_counts': {'successful': 0, 'attempted': 0}})
    logits = {s: logits_on(original, tensors[s], mean, std, 4, lambda: None) for s in scores}
    atomic_save({'before': logits, 'after': logits, 'labels': {s: targets[s] for s in scores}}, folder/'logits.pt')
    args = (tiny, baseline, lambda seed: deepcopy(source), tensors, targets, mean, std, output)
    first = execute(*args, scientific=False)
    second = execute(*args, scientific=False)
    if first['results'] != second['results']:
        raise AssertionError('Completed resume changed results')
    return {'status': 'passed', 'output': output.relative_to(ROOT).as_posix(),
            'local_fits': first['unique_local_fits'], 'new_repaired_models': first['new_repaired_models'], 'completed_resume_passed': True}


def prepare(config):
    output = ROOT/config['output']
    record = {'started_utc': now(), 'scientific_results': False}
    try:
        tests = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', 'tests_torch/test_grouping_exchange.py'],
                               cwd=ROOT, capture_output=True, text=True)
        record['tests'] = {'returncode': tests.returncode, 'stdout': tests.stdout, 'stderr': tests.stderr}
        print(tests.stdout, flush=True)
        if tests.returncode:
            raise RuntimeError('Exchange tests failed')
        with lock('compression_curve_runner.lock'):
            record['smoke'] = smoke(config)
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable')
            seed_everything(17019)
            baseline = read(ROOT/config['baseline_config'])
            hidden = [w for w in baseline['quaternion_widths'] for _ in range(3)]
            for index in config['blocks']:
                hidden[index] //= 2
            model = make_model(baseline, hidden).cuda()
            record['discarded_gpu_update_audit'] = first_update_audit(model, torch.randn(8, 3, 32, 32, device='cuda'),
                torch.arange(8, device='cuda') % baseline['classes'], torch.optim.SGD(model.parameters(), lr=.001))
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
    print('READY: grouping exchange pilot frozen before scientific search', flush=True)


def run(config):
    output = ROOT/config['output']
    with lock('compression_curve_runner.lock'):
        definition = register(config)
        ready = read(output/'readiness.json')
        if ready['status'] != 'passed' or ready['source_hashes'] != definition['source_hashes']:
            raise ValueError('Readiness does not cover current sources')
        baseline = read(ROOT/config['baseline_config'])
        device = torch.device(config['device'])
        seed_everything(config['seeds'][0])
        tensors, targets, mean, std = load_data(baseline, device)
        def sources(seed):
            seed_everything(seed)
            model = make_model(baseline).to(device).eval()
            model.load_state_dict(torch.load(parent_folder(seed)/'model.pt', map_location=device, weights_only=True)['model'])
            return model
        result = execute(config, baseline, sources, tensors, targets, mean, std, output)
        if result is not None:
            register(config)
            count = 0
            for path in output.glob('seed_*/block_*/search.json'):
                for row in read(path)['finalists']:
                    verified(path.parent/row['checkpoint_file'], row['checkpoint'])
                    count += 1
            for path in output.glob('seed_*/result.json'):
                row = read(path)
                for name, digest in row['artifacts'].items():
                    if file_sha256(path.parent/name) != digest:
                        raise ValueError('Final guided artifact changed')
                    count += 1
                for fit in row['refits']:
                    verified(path.parent/f"block_{fit['block']}"/'selected.pt', fit['checkpoint'])
                    count += 1
            write(output/'completion_audit.json', {'status': 'passed', 'artifact_hashes_checked': count,
                'frozen_source_input_hashes_checked': len({**definition['source_hashes'], **definition['input_hashes']}),
                'completed_seeds': len(result['results']), 'updated_utc': now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    config = read(ROOT/CONFIG)
    torch.set_num_threads(config['cpu_threads'])
    run(config) if args.run else prepare(config)
