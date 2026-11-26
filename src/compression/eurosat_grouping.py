"""Physically compact sequential EuroSAT conv pairs using the frozen Q compiler."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.cnn_training import prepare_batch
from src.compression.component_reassembly import packing_indices
from src.compression.resnet_reassembly import compile_block, original_quartets
from src.models.eurosat_cnn import make_fourband


class ConvPair(nn.Module):
    def __init__(self, sequence):
        super().__init__()
        self.conv1, self.bn1 = deepcopy(sequence[0]), deepcopy(sequence[1])
        self.conv2, self.bn2 = deepcopy(sequence[3]), deepcopy(sequence[4])

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))


def install_pair(model, index, pair):
    sequence = model.blocks[index]
    sequence[0], sequence[1], sequence[3], sequence[4] = pair.conv1, pair.bn1, pair.conv2, pair.bn2
    return model


def compile_pair(model, index, groups, retained=None):
    source = ConvPair(model.blocks[index]).eval()
    packing_indices(groups, source.conv1.out_channels)
    if retained is not None and sorted(s for q in groups for s in q) != sorted(s for q in retained for s in q):
        raise AssertionError('Retained scalar set changed')
    return compile_block(source, groups)


def make_compact(config, hidden=None):
    model = make_fourband(config, 'quaternion')
    if hidden is not None:
        if len(hidden) != len(model.blocks):
            raise ValueError('Expected three compact hidden widths')
        for index, width in enumerate(hidden):
            source_width = model.blocks[index][0].out_channels
            if width < 1 or width > source_width:
                raise ValueError('Invalid compact hidden width')
            if width != source_width:
                pair, _ = compile_pair(model, index, original_quartets(source_width)[:width])
                install_pair(model, index, pair)
    return model


class ReachedProducer(Exception):
    pass


@torch.no_grad()
def producer_inputs(model, index, images, mean, std, batch, stop=lambda: None):
    values = []
    def hook(layer, args):
        values.append(args[0].detach().cpu())
        raise ReachedProducer()
    handle = model.blocks[index][0].register_forward_pre_hook(hook)
    try:
        model.eval()
        for x in images.split(batch):
            stop()
            try:
                model(prepare_batch(x, mean, std))
            except ReachedProducer:
                pass
    finally:
        handle.remove()
    if not values:
        raise RuntimeError('Producer not reached')
    return torch.cat(values)


def load_classification_data(folder, metadata, device):
    with np.load(folder/'eurosat_nrgb.npz') as archive:
        images, labels = archive['images'], archive['labels']
    if int(images.max()) > np.iinfo(np.int16).max:
        raise ValueError('EuroSAT int16 conversion would lose values')
    with np.load(folder/'split_indices.npz') as split:
        ids = {'train': split['train'], 'validation': split['validation'],
               'fit': split['calibration'][::2], 'check_calibration': split['calibration'][1::2]}
        all_ids = [*ids.values(), split['test']]
        flat = np.concatenate(all_ids)
        if len(set(flat.tolist())) != len(images) or sorted(flat.tolist()) != list(range(len(images))):
            raise ValueError('Invalid EuroSAT split coverage/disjointness')
    if [len(ids[k]) for k in ids] != [19600, 1000, 500, 500]:
        raise ValueError('Unexpected EuroSAT split sizes')
    tensors = {s: torch.as_tensor(images[v].astype(np.int16), device=device) for s, v in ids.items()}
    targets = {s: torch.as_tensor(labels[v], dtype=torch.long, device=device) for s, v in ids.items()}
    mean, std = [torch.tensor(metadata['statistics'][k], dtype=torch.float32, device=device).reshape(1, 4, 1, 1)
                 for k in ('mean', 'std')]
    return tensors, targets, mean, std
