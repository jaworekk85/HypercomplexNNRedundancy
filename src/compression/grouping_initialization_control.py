"""Normalized fixed-group ridge controls; no claim of semantic identifiability."""
from copy import deepcopy

import torch

from src.compression.grouping_mechanism import raw,permute_moments,q_moments,solve,relative_permutation
from src.compression.component_reassembly import packing_indices
from src.compression.resnet_reassembly import original_quartets
from src.compression.direct_consumer import theta,sse


@torch.no_grad()
def consumer_targets(consumer,packs,device,batch,stop=lambda:None):
    result = {}
    for split,pack in packs.items():
        values = []
        for h in pack['features'].split(batch):
            stop()
            values.append(consumer(h.to(device)).cpu())
        result[split] = torch.cat(values)
    return result


def normalized_record(record,moments):
    result = deepcopy(record)
    for split,name in (('fit','fit'),('check_calibration','check')):
        m = moments[split]
        energy = m['yy']/m['count']
        if energy <= 0:
            raise ValueError('Zero target energy prevents normalization')
        result[name+'_target_mean_square'] = energy
        result[name+'_normalized_mse'] = result[name+'_mse']/energy
    return result


def fit_cell(packs,targets,assignment,shape,alpha,progress=lambda **kw:None,stop=lambda:None):
    kept,width = assignment['kept'],assignment['source_width']
    original = packing_indices([original_quartets(width)[i] for i in kept],width)
    moments = {}
    for split,pack in packs.items():
        stop()
        moments[split] = raw(pack['features'][:,original],targets[split])
    gram,_ = q_moments(moments['fit'],shape[1])
    penalty = alpha*float(gram.diag().mean())
    if penalty <= 0:
        raise ValueError('Degenerate original feature Gram')
    rows,kernels = [],[]
    for case,groups in assignment['groups'].items():
        stop()
        permutation = relative_permutation(groups,kept,width)
        changed = {s:permute_moments(m,permutation) for s,m in moments.items()}
        kernel,record = solve(changed['fit'],changed['check_calibration'],shape,penalty)
        rows.append({'case':case,**normalized_record(record,changed)})
        kernels.append(kernel)
        progress(case=case,completed=len(rows),total=len(assignment['groups']))
    # A coherent permutation of complete quaternion units is a true symmetry.
    groups = [original_quartets(width)[i] for i in reversed(kept)]
    permutation = relative_permutation(groups,kept,width)
    _,reorder = solve(permute_moments(moments['fit'],permutation),permute_moments(moments['check_calibration'],permutation),shape,penalty)
    reference = next(r for r in rows if r['case'] == 'original')
    if abs(reorder['regularized_fit_objective']-reference['regularized_fit_objective']) > 1e-8*max(1.,reference['regularized_fit_objective']):
        raise AssertionError('Whole-unit reorder failed')
    # Replay a saved bank through independently reconstructed moment objectives.
    replay = []
    for row,kernel in zip(rows,kernels):
        stop()
        p = relative_permutation(assignment['groups'][row['case']],kept,width)
        checked = permute_moments(moments['check_calibration'],p)
        g,b = q_moments(checked,shape[1])
        values = theta(kernel.numpy())
        mse = sse(g.numpy(),b.numpy(),checked['yy'],values)/checked['count']
        if abs(mse-row['check_mse']) > 1e-10*max(1.,mse):
            raise AssertionError('Coefficient bank check replay failed')
        replay.append(mse)
    return rows,kernels,{'lambda':penalty,'whole_unit_reorder':reorder,'coefficient_replay_mse':replay}


def policy_summary(rows):
    base = next(r for r in rows if r['case'] == 'original')
    result = {'original_normalized_mse':base['check_normalized_mse']}
    for policy in ('role','free'):
        selected = [r for r in rows if r['case'].startswith(policy+'_100_')]
        if not selected:
            raise ValueError('Missing randomized policy')
        value = sum(r['check_normalized_mse'] for r in selected)/len(selected)
        result[policy] = {'draws':len(selected),'mean_normalized_mse':value,
                          'delta_vs_original':value-base['check_normalized_mse'],
                          'mse_ratio_vs_original':value/base['check_normalized_mse'] if base['check_normalized_mse'] > 0 else None}
    return result


def crossed_summary(cells):
    summaries = {name:policy_summary(rows) for name,rows in cells.items()}
    result = {'cells':summaries,'contrasts':{}}
    for policy in ('role','free'):
        d = {name:summary[policy]['delta_vs_original'] for name,summary in summaries.items()}
        ii,it,ti,tt = (d[name] for name in ('initial__initial','initial__trained','trained__initial','trained__trained'))
        result['contrasts'][policy] = {'initial_penalty':ii,'trained_penalty':tt,'training_increment':tt-ii,
            'feature_increment_at_initial_consumer':ti-ii,'consumer_increment_at_initial_features':it-ii,
            'interaction':tt-ti-it+ii}
    return result
