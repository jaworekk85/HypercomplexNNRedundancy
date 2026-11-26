from copy import deepcopy
import random

import numpy as np
import pytest
import torch
from torch import nn

from src.models.quaternion_conv_torch import QuaternionConv2d
from src.grouping_repair_curve import (preserved_rng, trajectory, save_snapshot,
                                       paired_gaps, checked_snapshot)


def tiny_fixture():
    torch.set_num_threads(2)
    def make():
        return nn.Sequential(QuaternionConv2d(1, 2, 1, padding=0), nn.BatchNorm2d(8),
                             nn.ReLU(), nn.Dropout(.2), nn.AdaptiveAvgPool2d(1),
                             nn.Flatten(), nn.Linear(8, 2)).eval()
    torch.manual_seed(17)
    initial = make()
    def reload_model(path):
        model = make()  # Deliberately consumes RNG, including during snapshot replay.
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model'])
        return model.eval()
    tensors = {s: torch.randint(256, (n, 4, 4, 4), dtype=torch.uint8)
               for s, n in [('train', 8), ('validation', 4), ('check_calibration', 4)]}
    labels = {s: torch.arange(len(x)) % 2 for s, x in tensors.items()}
    config = dict(epochs=2, batch_size=4, learning_rate=.001, minimum_learning_rate=.00001,
                  weight_decay=.0001, mixed_precision=False, order_seed_offset=80, augment_seed_offset=90)
    return initial, reload_model, tensors, labels, torch.zeros(1, 4, 1, 1), torch.ones(1, 4, 1, 1), config


def test_snapshot_preserves_dropout_rng_and_model_mode(tmp_path):
    model, reload_model, tensors, labels, mean, std, config = tiny_fixture()
    model.train()
    torch_state, np_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    saved_state = deepcopy(model.state_dict())
    save_snapshot(model, reload_model, tensors, labels, mean, std, 4, tmp_path, 0,
                  {'attempted': 0, 'successful': 0})
    assert model.training
    assert torch.equal(torch_state, torch.get_rng_state())
    assert np.array_equal(np_state[1], np.random.get_state()[1])
    assert python_state == random.getstate()
    assert all(torch.equal(v, model.state_dict()[n]) for n, v in saved_state.items())


def test_interrupted_epoch_resume_matches_uninterrupted_dropout_training(tmp_path):
    model, reload_model, tensors, labels, mean, std, config = tiny_fixture()
    source = deepcopy(model)
    full = trajectory(model, reload_model, tensors, labels, mean, std, config, 19, tmp_path/'full', [0, 1, 2])
    def interrupt(count):
        if count == 3:  # Partial epoch two; epoch one is saved.
            raise RuntimeError('technical interruption')
    with pytest.raises(RuntimeError, match='technical interruption'):
        trajectory(deepcopy(source), reload_model, tensors, labels, mean, std, config, 19,
                   tmp_path/'resume', [0, 1, 2], after_batch=interrupt)
    resumed = trajectory(deepcopy(source), reload_model, tensors, labels, mean, std, config,
                         19, tmp_path/'resume', [0, 1, 2])
    assert resumed['optimizer_counts'] == full['optimizer_counts'] == {'successful': 4, 'attempted': 4}
    a = torch.load(tmp_path/'full/epoch_002/model.pt', weights_only=True)['model']
    b = torch.load(tmp_path/'resume/epoch_002/model.pt', weights_only=True)['model']
    assert all(torch.equal(v, b[n]) for n, v in a.items())
    assert [r['validation'] for r in full['history']] == [r['validation'] for r in resumed['history']]
    assert trajectory(deepcopy(source), reload_model, tensors, labels, mean, std, config,
                      19, tmp_path/'resume', [0, 1, 2]) == resumed


def test_completed_epoch_snapshot_is_recovered_after_callback_failure(tmp_path):
    model, reload_model, tensors, labels, mean, std, config = tiny_fixture()
    source = deepcopy(model)
    def interrupt_after_epoch(row, counts):
        if row['epoch'] == 1:
            raise RuntimeError('callback interruption')
    with pytest.raises(RuntimeError, match='callback interruption'):
        trajectory(model, reload_model, tensors, labels, mean, std, config, 19,
                   tmp_path, [0, 1, 2], progress=interrupt_after_epoch)
    (tmp_path/'epoch_001/result.json').unlink()
    result = trajectory(source, reload_model, tensors, labels, mean, std, config, 19, tmp_path, [0, 1, 2])
    assert result['snapshots'][1]['epoch'] == 1
    assert checked_snapshot(tmp_path/'epoch_001')['optimizer_counts']['attempted'] == 2


def test_snapshot_hash_and_trajectory_recipe_changes_are_rejected(tmp_path):
    model, reload_model, tensors, labels, mean, std, config = tiny_fixture()
    source = deepcopy(model)
    trajectory(model, reload_model, tensors, labels, mean, std, config, 19, tmp_path, [0, 1, 2])
    with pytest.raises(ValueError, match='recipe changed'):
        trajectory(source, reload_model, tensors, labels, mean, std, {**config, 'learning_rate': .2},
                   19, tmp_path, [0, 1, 2])
    with (tmp_path/'epoch_001/model.pt').open('ab') as handle:
        handle.write(b'corruption')
    with pytest.raises(ValueError, match='artifact changed'):
        trajectory(source, reload_model, tensors, labels, mean, std, config, 19, tmp_path, [0, 1, 2])


def test_paired_gaps_keep_reversals_and_do_not_mix_datasets():
    def row(dataset, case, accuracy):
        return dict(dataset=dataset, seed=19, case=case, snapshots=[dict(epoch=30, metrics={
            s: dict(accuracy=accuracy, cross_entropy=1-accuracy) for s in ('validation', 'check_calibration')})])
    rows = [row('cifar10', 'original', .8), row('cifar10', 'free_100_0', .9),
            row('eurosat', 'original', .95), row('eurosat', 'free_100_0', .85)]
    gaps = paired_gaps(rows)
    assert gaps[0]['validation']['original_minus_mixed_accuracy_pp'] == pytest.approx(-10.)
    assert gaps[1]['validation']['original_minus_mixed_accuracy_pp'] == pytest.approx(10.)
