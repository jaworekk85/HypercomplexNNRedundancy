"""Bounded fixed-set exchanges; standard projection, ridge and teacher matching."""
from copy import deepcopy
from itertools import combinations

import torch
from torch.nn import functional as F

from src.compression.component_reassembly import hamilton_basis, packing_indices
from src.compression.direct_consumer import kernel_of, sse
from src.compression.grouping_mechanism import raw, q_moments
from src.compression.resnet_reassembly import compile_block, producer_features
from src.models.quaternion_conv_torch import hamilton_kernel


def canonical(groups):
    return tuple(sorted(tuple(int(s) for s in q) for q in groups))


def validate_partition(groups, original, width):
    packing_indices(groups, width)
    if sorted(s for q in groups for s in q) != sorted(s for q in original for s in q):
        raise AssertionError('Retained scalar set changed')


def fit_split(count, inner, selection, seed):
    if min(inner, selection) <= 0 or inner+selection > count:
        raise ValueError('Invalid independent FIT subsets')
    ids = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    return ids[:inner], ids[inner:inner+selection]


def producer_decoder(block):
    width = block.conv1.out_channels
    rows = hamilton_kernel(block.conv1.kernel.detach().double().cpu()).reshape(4*width, 4, -1)
    return torch.einsum('cab,sbv->ascv', hamilton_basis(), rows).reshape(4, 4*width, -1)


def projection_cost(groups, decoder):
    values = torch.stack([decoder[r, torch.tensor([q[r] for q in groups])] for r in range(4)], 1)
    return float((values-values.mean(1, keepdim=True)).square().sum())


def proposals(groups, decoder, config, seed):
    groups = canonical(groups)
    candidates = {}
    for a, b in combinations(range(4*len(groups)), 2):
        i, r = divmod(a, 4)
        j, t = divmod(b, 4)
        if i == j:
            continue
        value = [list(q) for q in groups]
        value[i][r], value[j][t] = value[j][t], value[i][r]
        key = canonical(value)
        if key != groups:
            candidates[key] = None
    ranked = sorted(candidates, key=lambda qs: (projection_cost(qs, decoder), qs))
    first = ranked[:config['compatible_proposals']]
    remaining = ranked[len(first):]
    order = torch.randperm(len(remaining), generator=torch.Generator().manual_seed(seed)).tolist()
    return first+[remaining[i] for i in order[:config['random_proposals']]]


@torch.no_grad()
def fit_inner(source, groups, inputs, targets, penalty, device, batch, stop=lambda: None):
    block, description = compile_block(source, groups)
    block = block.to(device).eval()
    features = producer_features(block, inputs, device, batch, stop)
    moments = raw(features, targets)
    gram, rhs = q_moments(moments, block.conv2.out_channels)
    system = gram+penalty*torch.eye(len(gram), dtype=gram.dtype)
    values = torch.linalg.solve(system, rhs)
    residual = float(torch.linalg.vector_norm(system@values-rhs)/max(1., float(torch.linalg.vector_norm(rhs))))
    if not bool(torch.isfinite(values).all()) or residual > 1e-8:
        raise RuntimeError('Inner Hamilton fit failed')
    block.conv2.kernel.copy_(torch.from_numpy(kernel_of(values.numpy(), tuple(block.conv2.kernel.shape))).to(block.conv2.kernel))
    return block, {'lambda': penalty, 'normal_residual': residual,
                   'fit_mse': sse(gram.numpy(), rhs.numpy(), moments['yy'], values.numpy())/moments['count'],
                   **description}


@torch.no_grad()
def block_outputs(block, inputs, device, batch, stop=lambda: None):
    block.eval()
    values = []
    for x in inputs.split(batch):
        stop()
        values.append(block(x.to(device)).cpu())
    return torch.cat(values)


def search_partition(starts, decoder, evaluate, config, seed, stop=lambda: None, progress=lambda **kw: None):
    original = canonical(starts['original'])
    cache, traces, finalists = {}, [], []
    def score(groups):
        key = canonical(groups)
        if key not in cache:
            stop()
            cache[key] = evaluate(key)
            progress(evaluated=len(cache))
        return float(cache[key]['score'])
    score(original)
    finalists.append(original)
    for start_index, (name, groups) in enumerate(starts.items()):
        current = canonical(groups)
        value = score(current)
        trace = {'start': name, 'initial_groups': current, 'initial_score': value, 'rounds': []}
        for round_index in range(config['rounds']):
            candidates = proposals(current, decoder, config, seed+1000*start_index+round_index)
            for candidate in candidates:
                validate_partition(candidate, original, decoder.shape[1]//4)
            winner = min([current]+candidates, key=lambda qs: (score(qs), qs != original, qs))
            best = score(winner)
            accepted = best < value-config['tolerance']*max(1., abs(value))
            trace['rounds'].append({'round': round_index+1, 'proposals': len(candidates),
                                     'before': value, 'best': best, 'accepted': accepted})
            if not accepted:
                break
            current, value = winner, best
        trace.update(final_groups=current, final_score=value)
        traces.append(trace)
        if current not in finalists:
            finalists.append(current)
    return finalists, cache, traces


@torch.no_grad()
def prediction_kl(teacher_logits, student_logits, temperature):
    if temperature <= 0 or teacher_logits.shape != student_logits.shape:
        raise ValueError('Invalid teacher matching shapes/temperature')
    teacher = F.softmax(teacher_logits.double()/temperature, dim=1)
    teacher_log = F.log_softmax(teacher_logits.double()/temperature, dim=1)
    student_log = F.log_softmax(student_logits.double()/temperature, dim=1)
    return float((teacher*(teacher_log-student_log)).sum(1).mean()*temperature**2)


def same_state(first, second):
    a, b = first.state_dict(), second.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
