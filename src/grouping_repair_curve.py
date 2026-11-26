"""Matched repair snapshots around the immutable epoch-resumable trainer."""
from contextlib import contextmanager
from copy import deepcopy
import random

import numpy as np
import torch

from src.cnn_training import evaluate, first_update_audit, prepare_batch
from src.data.cifar_cached import file_sha256
from src.trainable_map_repair import atomic_save, repair
from scripts.resnet_component_reassembly_pilot import read, write, assert_replay


@contextmanager
def preserved_rng():
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
              torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if states[3] is not None:
            torch.cuda.set_rng_state_all(states[3])


def measured(model, tensors, labels, mean, std, batch):
    scores, logits = {}, {}
    for split in ('validation', 'check_calibration'):
        scores[split], values = evaluate(model, tensors[split], labels[split], mean, std, batch)
        logits[split] = torch.from_numpy(values)
    return scores, logits


def checked_snapshot(folder):
    record = read(folder/'result.json')
    for name, expected in record['artifacts'].items():
        if file_sha256(folder/name) != expected:
            raise ValueError('Snapshot artifact changed: '+str(folder/name))
    return record


def save_snapshot(model, reload_model, tensors, labels, mean, std, batch, folder,
                  epoch, counts, expected_scores=None, expected_logits=None):
    mode = model.training
    with preserved_rng():
        scores, logits = measured(model, tensors, labels, mean, std, batch)
        if expected_scores is not None:
            assert_replay(expected_scores, scores)
        if expected_logits is not None:
            for split in scores:
                torch.testing.assert_close(logits[split], expected_logits[split], rtol=0, atol=0)
        if (folder/'result.json').exists():
            record = checked_snapshot(folder)
            assert_replay(record['metrics'], scores)
            if record['epoch'] != epoch or record['optimizer_counts'] != counts:
                raise ValueError('Snapshot epoch/update count changed')
            saved = torch.load(folder/'logits.pt', map_location='cpu', weights_only=True)
            for split in scores:
                torch.testing.assert_close(logits[split], saved['logits'][split], rtol=0, atol=0)
                torch.testing.assert_close(labels[split].cpu(), saved['labels'][split], rtol=0, atol=0)
        else:
            atomic_save({'model': {n: v.detach().cpu() for n, v in model.state_dict().items()}}, folder/'model.pt')
            atomic_save({'logits': logits, 'labels': {s: labels[s].cpu() for s in scores}}, folder/'logits.pt')
            replica = reload_model(folder/'model.pt')
            actual, replay_logits = measured(replica, tensors, labels, mean, std, batch)
            assert_replay(scores, actual)
            for split in scores:
                torch.testing.assert_close(logits[split], replay_logits[split], rtol=0, atol=0)
            del replica
            record = {'epoch': epoch, 'metrics': scores, 'optimizer_counts': dict(counts),
                      'model_and_logits_reload_exact': True,
                      'artifacts': {n: file_sha256(folder/n) for n in ('model.pt', 'logits.pt')}}
            write(folder/'result.json', record)
    model.train(mode)
    return record


def discarded_update_audit(model, tensors, labels, mean, std):
    with preserved_rng():
        probe = deepcopy(model)
        audit = first_update_audit(probe, prepare_batch(tensors['train'][:8], mean, std),
                                   labels['train'][:8], torch.optim.SGD(probe.parameters(), lr=.001))
        del probe
    return audit


def trajectory(model, reload_model, tensors, labels, mean, std, config, seed, folder,
               snapshots, stop_path=None, progress=lambda *args: None,
               expected_scores=None, expected_logits=None, after_batch=None):
    folder.mkdir(parents=True, exist_ok=True)
    if snapshots[0] != 0 or snapshots[-1] != config['epochs'] or sorted(set(snapshots)) != snapshots:
        raise ValueError('Snapshots must be ordered, unique and include initial/final epoch')
    if (folder/'result.json').exists():
        result = read(folder/'result.json')
        if result['config'] != config or result['seed'] != seed or result['snapshot_epochs'] != snapshots:
            raise ValueError('Completed trajectory recipe changed')
        for epoch in snapshots:
            checked_snapshot(folder/f'epoch_{epoch:03d}')
        with preserved_rng():
            final = reload_model(folder/f"epoch_{config['epochs']:03d}"/'model.pt')
            saved = torch.load(folder/f"epoch_{config['epochs']:03d}"/'logits.pt', map_location='cpu', weights_only=True)
            scores, logits = measured(final, tensors, labels, mean, std, config['batch_size'])
            assert_replay(result['snapshots'][-1]['metrics'], scores)
            for split in scores:
                torch.testing.assert_close(logits[split], saved['logits'][split], rtol=0, atol=0)
        return result
    save_snapshot(model, reload_model, tensors, labels, mean, std, config['batch_size'],
                  folder/'epoch_000', 0, {'attempted': 0, 'successful': 0}, expected_scores, expected_logits)
    audit_path = folder/'discarded_update_audit.json'
    if not audit_path.exists():
        write(audit_path, discarded_update_audit(model, tensors, labels, mean, std))
    last = folder/'last.pt'
    if last.exists():
        saved = torch.load(last, map_location=mean.device, weights_only=False)
        if saved['config'] != config or saved['seed'] != seed:
            raise ValueError('Checkpoint recipe changed')
        model.load_state_dict(saved['model'])
        epoch = len(saved['history'])
        if epoch in snapshots:
            save_snapshot(model, reload_model, tensors, labels, mean, std, config['batch_size'],
                          folder/f'epoch_{epoch:03d}', epoch, saved['count'])
        del saved
    def on_epoch(row, counts):
        if row['epoch'] in snapshots:
            save_snapshot(model, reload_model, tensors, labels, mean, std, config['batch_size'],
                          folder/f"epoch_{row['epoch']:03d}", row['epoch'], dict(counts))
        progress(row, counts)
    _, history, counts, done = repair(model, tensors, labels, mean, std, config, seed,
                                      last, stop_path, on_epoch, after_batch)
    if not done:
        return None
    result = {'seed': seed, 'config': config, 'snapshot_epochs': snapshots,
              'snapshots': [checked_snapshot(folder/f'epoch_{e:03d}') for e in snapshots],
              'history': history, 'optimizer_counts': counts,
              'real_trainable_parameters': sum(p.numel() for p in model.parameters()),
              'test_evaluated': False, 'best_epoch_selection': False}
    write(folder/'result.json', result)
    return result


def paired_gaps(rows):
    lookup = {(r['dataset'], r['seed'], r['case']): r for r in rows}
    output = []
    for (dataset, seed, case), mixed in sorted(lookup.items()):
        if case == 'original' or (dataset, seed, 'original') not in lookup:
            continue
        original = {s['epoch']: s for s in lookup[(dataset, seed, 'original')]['snapshots']}
        for snapshot in mixed['snapshots']:
            epoch = snapshot['epoch']
            row = {'dataset': dataset, 'seed': seed, 'case': case, 'epoch': epoch}
            for split in ('validation', 'check_calibration'):
                a, b = original[epoch]['metrics'][split], snapshot['metrics'][split]
                row[split] = {'original_minus_mixed_accuracy_pp': 100*(a['accuracy']-b['accuracy']),
                              'mixed_minus_original_ce': b['cross_entropy']-a['cross_entropy']}
            output.append(row)
    return output
