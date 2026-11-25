"""Fixed-feature grouping interventions; standard ridge and Hamilton projection."""
from copy import deepcopy

import torch
from torch.nn import functional as F

from src.compression.component_reassembly import packing_indices
from src.compression.direct_consumer import q_normal_equations, kernel_of, theta, sse
from src.compression.resnet_reassembly import original_quartets, producer_features
from src.compression.structure_catalogue import raw_moments
from src.models.quaternion_conv_torch import hamilton_kernel


def kept_units(features):
    width = features.shape[1]//4
    energy = features.double().square().mean((0,2,3)).reshape(4,width).sum(0)
    return sorted(sorted(range(width),key=lambda i:(-float(energy[i]),i))[:width//2])


def regroup(kept,width,policy,fraction,seed):
    groups = original_quartets(width)
    result = [groups[i] for i in kept]
    if fraction == 0:
        return result
    generator = torch.Generator().manual_seed(seed)
    number = round(len(kept)*fraction)
    if number < 2 or number > len(kept):
        raise ValueError('Mixing needs at least two retained units')
    active = torch.randperm(len(kept),generator=generator)[:number].tolist()
    pool = [result[i] for i in active]
    if policy == 'role':
        first = torch.randperm(number,generator=generator).tolist()
        shift = int(torch.randint(1,number,(),generator=generator))
        orders = [first,first[shift:]+first[:shift]]
        orders += [torch.randperm(number,generator=generator).tolist() for _ in range(2)]
        assigned = [tuple(pool[orders[r][i]][r] for r in range(4)) for i in range(number)]
    elif policy == 'free':
        flat = [q[r] for r in range(4) for q in pool]
        for _ in range(10000):
            perm = torch.randperm(len(flat),generator=generator).tolist()
            assigned = [tuple(flat[perm[r*number+i]] for r in range(4)) for i in range(number)]
            if all(len({s%width for s in q}) > 1 for q in assigned) and any(s//width != r for q in assigned for r,s in enumerate(q)):
                break
        else:
            raise RuntimeError('Could not sample mixed groups')
    else:
        raise ValueError(policy)
    for i,q in zip(active,assigned):
        result[i] = q
    packing_indices(result,width)
    if sorted(s for q in result for s in q) != sorted(s for i in kept for s in groups[i]):
        raise AssertionError('Retained scalar set changed')
    if sum(len({s%width for s in q}) > 1 for q in result) != number:
        raise AssertionError('Wrong mixing degree')
    return result


def case_groups(kept,width,seed,block,config):
    result = {'original':regroup(kept,width,'role',0,seed)}
    base = seed+config['shuffle_seed_offset']+block*10000
    for policy,offset in (('role',0),('free',1000)):
        result[f'{policy}_50_0'] = regroup(kept,width,policy,.5,base+offset+500)
        for repeat in range(config['random_repeats']):
            result[f'{policy}_100_{repeat}'] = regroup(kept,width,policy,1,base+offset+repeat)
    return result


def raw(features,targets):
    g,b,yy,count = raw_moments(features,targets)
    return {'gram':g,'cross':b,'yy':yy,'count':count}


def permute_moments(moments,permutation):
    ids = torch.tensor([9*c+j for c in permutation for j in range(9)],dtype=torch.long)
    return {'gram':moments['gram'][ids][:,ids],'cross':moments['cross'][ids],
            'yy':moments['yy'],'count':moments['count']}


def q_moments(moments,outputs):
    g,b = q_normal_equations(moments['gram'].numpy(),moments['cross'].numpy(),outputs)
    return torch.from_numpy(g),torch.from_numpy(b)


def solve(fit,check,shape,penalty):
    g,b = q_moments(fit,shape[1])
    values = torch.linalg.solve(g+penalty*torch.eye(len(g),dtype=g.dtype),b)
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError('Nonfinite grouping ridge')
    residual = float(torch.linalg.vector_norm((g+penalty*torch.eye(len(g),dtype=g.dtype))@values-b)/max(1.,float(torch.linalg.vector_norm(b))))
    record = {'lambda':penalty,'zero_centered':True,'relative_normal_residual':residual,
              'coefficient_count':values.numel()}
    for name,m in (('fit',fit),('check',check)):
        mg,mb = q_moments(m,shape[1])
        record[name+'_mse'] = sse(mg.numpy(),mb.numpy(),m['yy'],values.numpy())/m['count']
    record['regularized_fit_objective'] = record['fit_mse']*fit['count']+penalty*float(values.square().sum())
    return torch.from_numpy(kernel_of(values.numpy(),shape)),record


def relative_permutation(quartets,kept,width):
    source = packing_indices([original_quartets(width)[i] for i in kept],width)
    lookup = {s:i for i,s in enumerate(source)}
    return [lookup[s] for s in packing_indices(quartets,width)]


@torch.no_grad()
def dense_permutation_check(source,features,kept,quartets):
    width = source.conv1.out_channels
    permutation = relative_permutation(quartets,kept,width)
    ids = packing_indices([original_quartets(width)[i] for i in kept],width)
    x = features[:2,ids].double()
    weights = hamilton_kernel(source.conv2.kernel.detach().cpu().double())[:,ids]
    expected = F.conv2d(x,weights,padding=1)
    actual = F.conv2d(x[:,permutation],weights[:,permutation],padding=1)
    error = float((actual-expected).abs().max())
    if error > 1e-8*max(1.,float(expected.abs().max())):
        raise AssertionError('Dense inverse-permutation identity failed')
    return error


@torch.no_grad()
def physical_fit(block,packs,penalty,device,batch,stop=lambda:None):
    moments,feature_error = {},{}
    for split,pack in packs.items():
        h = producer_features(block,pack['inputs'],device,batch,stop)
        moments[split] = raw(h,pack['targets'])
        if 'selected_features' in pack:
            feature_error[split] = float((h.double()-pack['selected_features'].double()).square().mean())
    kernel,record = solve(moments['fit'],moments['check_calibration'],tuple(block.conv2.kernel.shape),penalty)
    block.conv2.kernel.copy_(kernel.to(block.conv2.kernel))
    record['producer_feature_mse'] = feature_error
    return record


def fit_producer(block,pack,config,seed,device,stop=lambda:None):
    """Only producer kernel trains; fixed BN state and exact selected targets."""
    block = deepcopy(block).to(device).eval()
    for parameter in block.parameters():
        parameter.requires_grad_(False)
    block.conv1.kernel.requires_grad_(True)
    before = block.conv1.kernel.detach().clone()
    optimizer = torch.optim.Adam([block.conv1.kernel],lr=config['learning_rate'])
    generator = torch.Generator().manual_seed(seed+config['seed_offset'])
    order = torch.empty(0,dtype=torch.long)
    losses,gradient_norms = [],[]
    for step in range(config['steps']):
        stop()
        if len(order) < config['batch_size']:
            order = torch.cat((order,torch.randperm(len(pack['inputs']),generator=generator)))
        ids,order = order[:config['batch_size']],order[config['batch_size']:]
        optimizer.zero_grad(set_to_none=True)
        h = F.relu(block.bn1(block.conv1(pack['inputs'][ids].to(device))))
        loss = (h-pack['selected_features'][ids].to(device)).square().mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite producer feature loss')
        loss.backward()
        grad = block.conv1.kernel.grad
        if grad is None or not bool(torch.isfinite(grad).all()):
            raise RuntimeError('Missing/nonfinite producer gradient')
        if step == 0:
            gradient_norms = grad.flatten(1).norm(dim=1).tolist()
        optimizer.step()
        losses.append(float(loss.detach()))
    updates = (block.conv1.kernel.detach()-before).flatten(1).norm(dim=1).tolist()
    if any(v <= 0 for v in gradient_norms+updates):
        raise RuntimeError('A producer quaternion component did not train')
    for parameter in block.parameters():
        parameter.requires_grad_(True)
    return block.eval(),{'losses':losses,'first_component_gradient_norms':gradient_norms,'component_update_norms':updates,
                         'bn_state_fixed':True,'only_producer_kernel_optimized':True}
