"""Complete-budget unit/component selection with no quartet-completion reserve.

Projection ranking, discrete exchanges and ridge reconstruction are standard.
This module makes no novelty claim and does not use classification labels.
"""
from itertools import combinations

import torch

from src.compression.component_reassembly import packing_indices
from src.compression.grouping_exchange import canonical, producer_decoder
from src.compression.resnet_reassembly import original_quartets, energy_groups
from src.models.quaternion_conv_torch import hamilton_kernel


def validate(groups, width, count, policy):
    groups = canonical(groups)
    packing_indices(groups, width)
    if len(groups) != count:
        raise ValueError('Incorrect compact quartet count')
    if policy == 'unit' and any(q not in original_quartets(width) for q in groups):
        raise ValueError('Unit-only policy received a non-intact quartet')
    return groups


def starts(block, features, count, seed):
    width = block.conv1.out_channels
    energy = features.double().square().mean((0, 2, 3)).cpu()
    consumer = hamilton_kernel(block.conv2.kernel.detach().double().cpu())
    saliency = energy*consumer.square().sum((0, 2, 3))
    original = original_quartets(width)
    def units(score):
        totals = score.reshape(4, width).sum(0)
        ids = sorted(range(width), key=lambda i: (-float(totals[i]), i))[:count]
        return canonical(original[i] for i in ids)
    energy_units, weighted_units = units(energy), units(saliency)
    generator = torch.Generator().manual_seed(seed)
    random_ids = torch.randperm(width, generator=generator)[:count].tolist()
    return {
        'baseline': energy_units,
        'unit': [energy_units, weighted_units, canonical(original[i] for i in random_ids)],
        'component': [energy_units, weighted_units, canonical(energy_groups(features, count, False))],
    }


def projection_distances(block):
    decoded = producer_decoder(block)
    matrix = decoded.reshape(-1, decoded.shape[-1])
    norms = matrix.square().sum(1)
    return (norms[:, None]+norms[None, :]-2*matrix@matrix.T).clamp_min(0)


def projection_score(groups, distances, width):
    score = 0.
    for q in groups:
        ids = [r*4*width+s for r, s in enumerate(q)]
        score += sum(float(distances[ids[a], ids[b]]) for a, b in combinations(range(4), 2)) / 4
    return score


def complete_moves(groups, width, policy):
    """All legal one-quartet, one-scalar replacement and pair-swap neighbors."""
    groups = canonical(groups)
    count = len(groups)
    validate(groups, width, count, policy)
    selected = {s for q in groups for s in q}
    pool = {'unit': set(), 'replace': set(), 'swap': set()}
    for index, q in enumerate(groups):
        other = selected-set(q)
        for unit in original_quartets(width):
            if not other.intersection(unit):
                proposal = list(groups)
                proposal[index] = unit
                key = canonical(proposal)
                if key != groups:
                    pool['unit'].add(key)
    if policy == 'component':
        absent = sorted(set(range(4*width))-selected)
        for index in range(count):
            for role in range(4):
                for scalar in absent:
                    proposal = [list(q) for q in groups]
                    proposal[index][role] = scalar
                    pool['replace'].add(canonical(proposal))
        for a, b in combinations(range(4*count), 2):
            i, r = divmod(a, 4)
            j, t = divmod(b, 4)
            proposal = [list(q) for q in groups]
            proposal[i][r], proposal[j][t] = proposal[j][t], proposal[i][r]
            key = canonical(proposal)
            if key != groups:
                pool['swap'].add(key)
    elif policy != 'unit':
        raise ValueError('Unknown selection policy')
    return {category: sorted(values) for category, values in pool.items()}


def shortlist(groups, width, policy, distances, per_round, seed):
    pools = complete_moves(groups, width, policy)
    active = [key for key, values in pools.items() if values]
    generator = torch.Generator().manual_seed(seed)
    selected, quartet_costs = [], {}
    def cost(groups):
        for q in groups:
            if q not in quartet_costs:
                quartet_costs[q] = projection_score([q], distances, width)
        return sum(quartet_costs[q] for q in groups)
    # Reserve evaluation slots for actual component replacements and swaps;
    # zero-cost whole-unit moves cannot consume every projection-shortlisted slot.
    for number, category in enumerate(active):
        quota = per_round//len(active) + int(number < per_round % len(active))
        values = pools[category]
        ranked = sorted(values, key=lambda q: (cost(q), q))
        best = ranked[:min(len(ranked), (quota+1)//2)]
        remainder = [q for q in values if q not in best]
        order = torch.randperm(len(remainder), generator=generator).tolist()
        chosen = best+[remainder[i] for i in order[:max(0, quota-len(best))]]
        for q in chosen:
            if q not in selected:
                selected.append(q)
    return selected, {key: len(values) for key, values in pools.items()}


def search(warm_starts, width, count, policy, distances, evaluate, config, seed,
           stop=lambda: None, progress=lambda **fields: None):
    cache, traces = {}, []
    def score(groups):
        key = validate(groups, width, count, policy)
        if key not in cache:
            stop()
            cache[key] = evaluate(key)
            progress(unique_fits=len(cache), policy=policy)
        return float(cache[key]['score'])
    finalists = []
    for start_index, start in enumerate(warm_starts):
        current = validate(start, width, count, policy)
        value = score(current)
        trace = {'start': start_index, 'initial': current, 'initial_score': value, 'rounds': []}
        for round_index in range(config['rounds']):
            candidates, coverage = shortlist(current, width, policy, distances, config['per_round'],
                                              seed+1000*start_index+round_index)
            winner = min([current]+candidates, key=lambda q: (score(q), q))
            best = score(winner)
            accepted = best < value-config['tolerance']*max(1., abs(value))
            trace['rounds'].append({'round': round_index+1, 'legal_moves': coverage,
                                    'evaluated_proposals': len(candidates), 'accepted': accepted,
                                    'before': value, 'best': best})
            if not accepted:
                break
            current, value = winner, best
        trace.update(final=current, final_score=value)
        traces.append(trace)
        finalists.append(current)
    best = min(finalists, key=lambda q: (score(q), q))
    return best, cache, traces
