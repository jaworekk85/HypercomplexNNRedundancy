"""Role-aware scalar reassembly into standard compact Hamilton ResNet blocks.

All projections, ridge solves and Schur identities are established linear
algebra. The catalogue/search is a bounded heuristic, not a global optimum.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from src.compression.component_reassembly import hamilton_basis, packing_indices, project_hamilton_kernel
from src.compression.direct_consumer import kernel_of, q_normal_equations, sse, theta
from src.compression.structure_catalogue import raw_moments
from src.models.quaternion_conv_torch import QuaternionConv2d, hamilton_kernel


def original_quartets(width):
    return [tuple(role*width+unit for role in range(4)) for unit in range(width)]


def energy_groups(features,count,role_fixed):
    energy = features.detach().double().cpu().square().mean((0,2,3))
    width = len(energy)//4
    if role_fixed:
        selected = [sorted(range(r*width,(r+1)*width),key=lambda s:(-float(energy[s]),s))[:count] for r in range(4)]
    else:
        values = sorted(sorted(range(4*width),key=lambda s:(-float(energy[s]),s))[:4*count])
        selected = [values[r*count:(r+1)*count] for r in range(4)]
    return [tuple(selected[r][i] for r in range(4)) for i in range(count)]


def make_catalogue(block,features,count,config):
    width = block.conv1.out_channels
    scalars = 4*width
    energy = features.detach().double().cpu().square().mean((0,2,3))
    rows = hamilton_kernel(block.conv1.kernel.detach().double().cpu()).reshape(scalars,4,-1)
    decoded = torch.einsum('cab,sbv->ascv',hamilton_basis(),rows).reshape(4*scalars,-1)
    norms = decoded.square().sum(1)
    distances = (norms[:,None]+norms[None,:]-2*decoded@decoded.T).clamp_min(0)

    def anchored(s,role,variant,fixed):
        picked = {role:s}
        for step,r in enumerate(x for x in range(4) if x != role):
            eligible = range(r*width,(r+1)*width) if fixed else range(scalars)
            eligible = [v for v in eligible if v not in picked.values()]
            def score(v):
                return (sum(float(distances[r*scalars+v,a*scalars+t]) for a,t in picked.items()),-float(energy[v]),v)
            ranked = sorted(eligible,key=score)
            picked[r] = ranked[min(variant,len(ranked)-1) if step == 0 else 0]
        return tuple(picked[r] for r in range(4))

    role_pool = []
    def add(pool,q):
        if q not in pool:
            pool.append(q)
    for q in original_quartets(width)+energy_groups(features,count,True):
        add(role_pool,q)
    anchors = sorted(range(scalars),key=lambda s:(-float(energy[s]),s))
    for variant in range(config['neighbour_variants']+1):
        for s in anchors:
            add(role_pool,anchored(s,s//width,variant,True))
    role_pool = role_pool[:max(config['role_catalogue_limit'],width+count)]
    pool = list(role_pool)
    for q in energy_groups(features,count,False):
        add(pool,q)
    required = len(pool)
    for variant in range(config['neighbour_variants']+1):
        for s in anchors:
            for role in range(4):
                add(pool,anchored(s,role,variant,False))
    pool = pool[:max(config['full_catalogue_limit'],required)]
    return pool,{'role_catalogue_count':len(role_pool),'total_candidates':len(pool),
                 'scalar_energy_groups':[list(q) for q in energy_groups(features,count,False)]}


@torch.no_grad()
def compile_block(block,quartets,allow_overlap=False):
    old,consumer = block.conv1,block.conv2
    if allow_overlap:
        for q in quartets:
            packing_indices([q],old.out_channels)
        indices = [q[r] for r in range(4) for q in quartets]
    else:
        indices = packing_indices(quartets,old.out_channels)
    result = deepcopy(block).eval()
    result.conv1 = QuaternionConv2d(old.in_channels,len(quartets),old.kernel.shape[-1],old.stride,old.padding).to(old.kernel)
    rows = hamilton_kernel(old.kernel.detach().double().cpu())[indices]
    coefficients = project_hamilton_kernel(rows)
    result.conv1.kernel.copy_(coefficients.to(old.kernel))
    bn = block.bn1
    result.bn1 = nn.BatchNorm2d(len(indices),eps=bn.eps,momentum=bn.momentum).to(bn.weight)
    for name in ('weight','bias','running_mean','running_var'):
        getattr(result.bn1,name).copy_(getattr(bn,name)[indices])
    result.bn1.num_batches_tracked.copy_(bn.num_batches_tracked)
    result.conv2 = QuaternionConv2d(len(quartets),consumer.out_channels,consumer.kernel.shape[-1],consumer.stride,consumer.padding).to(consumer.kernel)
    selected_columns = hamilton_kernel(consumer.kernel.detach().double().cpu())[:,indices]
    result.conv2.kernel.copy_(project_hamilton_kernel(selected_columns).to(consumer.kernel))
    result.eval()
    return result,{'producer_projection_sse':float((rows-hamilton_kernel(coefficients)).square().sum()),
                   'quartets':[list(q) for q in quartets],
                   'mixed_quartets':sum(len({s%old.out_channels for s in q})>1 for q in quartets),
                   'changed_roles':sum(s//old.out_channels != r for q in quartets for r,s in enumerate(q))}


@torch.no_grad()
def producer_features(block,inputs,device,batch_size,check_stop=lambda:None):
    outputs = []
    for x in inputs.split(batch_size):
        check_stop()
        outputs.append(F.relu(block.bn1(block.conv1(x.to(device)))).cpu())
    return torch.cat(outputs)


def positions_for(images,spatial,count,seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(spatial,generator=generator)[:min(count,spatial)] for _ in range(images)])


def sampled_design(features,targets,positions):
    batch,channels,h,w = features.shape
    units = channels//4
    patches = F.unfold(features,3,padding=1).reshape(batch,4,units,9,h*w).permute(0,4,1,2,3)
    where = positions.to(features.device)
    patches = patches[torch.arange(batch,device=features.device)[:,None],where].reshape(-1,4,units,9).double()
    design = torch.einsum('cab,nbqd->naqcd',hamilton_basis(features.dtype,features.device).double(),patches).reshape(4*len(patches),units*36)
    y = targets.reshape(batch,4,-1,h*w).permute(0,3,1,2)
    response = y[torch.arange(batch,device=features.device)[:,None],where].reshape(4*len(patches),-1).double()
    return design,response


@torch.no_grad()
def selection_statistics(block,bank,pack,positions,config,device,check_stop=lambda:None):
    bank_block,_ = compile_block(block,bank,allow_overlap=True)
    bank_block = bank_block.to(device).eval()
    designs,responses = [],[]
    source_energy,source_columns = 0.,4*block.conv1.out_channels*9
    for start in range(0,len(pack['inputs']),config['bank_batch_size']):
        check_stop()
        end = start+config['bank_batch_size']
        x = pack['inputs'][start:end].to(device)
        h = F.relu(bank_block.bn1(bank_block.conv1(x)))
        target = pack['targets'][start:end].to(device)
        design,y = sampled_design(h,target,positions[start:end])
        designs.append(design); responses.append(y)
        source_design,_ = sampled_design(pack['features'][start:end].to(device),target,positions[start:end])
        source_energy += float(source_design.square().sum())/source_columns
    d,y = torch.cat(designs),torch.cat(responses)
    penalty = config['ridge_alpha']*source_energy
    if penalty <= 0:
        raise ValueError('Degenerate source design')
    prior = bank_block.conv2.kernel.detach().double().permute(2,0,3,4,1).reshape(len(bank)*36,-1)
    gram,cross = d.T@d,d.T@y
    stats = {'gram':gram,'cross':cross,'prior':prior,'rhs':cross+penalty*prior,
             'yy':float(y.square().sum()),'penalty':penalty,'block_size':36,
             'scalar_observations':y.numel(),'sample_rows':len(y)}
    del designs,responses,d,y,bank_block
    return stats


def columns(selected,size,device):
    return torch.tensor([q*size+j for q in selected for j in range(size)],device=device,dtype=torch.long)


def objective(stats,selected):
    if not selected:
        return stats['yy']
    idx = columns(selected,stats['block_size'],stats['gram'].device)
    g = stats['gram'][idx][:,idx]
    rhs = stats['rhs'][idx]
    solved = torch.linalg.solve(g+stats['penalty']*torch.eye(len(g),dtype=g.dtype,device=g.device),rhs)
    return stats['yy']+stats['penalty']*float(stats['prior'][idx].square().sum())-float((rhs*solved).sum())


def conditional_gains(stats,selected):
    g,rhs = stats['gram'],stats['rhs']
    size,penalty = stats['block_size'],stats['penalty']
    count = len(g)//size
    bank_ids = torch.arange(count,device=g.device)
    schur = g.reshape(count,size,count,size).permute(0,2,1,3)[bank_ids,bank_ids].clone()
    schur += penalty*torch.eye(size,dtype=g.dtype,device=g.device)[None]
    residual = rhs.reshape(count,size,-1).clone()
    if selected:
        idx = columns(selected,size,g.device)
        system = g[idx][:,idx]+penalty*torch.eye(len(idx),dtype=g.dtype,device=g.device)
        chol = torch.linalg.cholesky(system)
        cross = g[idx].reshape(len(idx),count,size).permute(1,0,2)
        solved_cross = torch.cholesky_solve(cross.permute(1,0,2).reshape(len(idx),count*size),chol).reshape(len(idx),count,size).permute(1,0,2)
        schur -= torch.einsum('mbi,mbj->mij',cross,solved_cross)
        residual -= torch.einsum('mbi,bo->mio',cross,torch.cholesky_solve(rhs[idx],chol))
    schur = (schur+schur.transpose(1,2))/2
    factors,info = torch.linalg.cholesky_ex(schur)
    if bool((info != 0).any()):
        raise RuntimeError('Schur system is not positive definite; no automatic ridge change')
    solution = torch.cholesky_solve(residual,factors)
    gain = (residual*solution).sum((1,2))-penalty*stats['prior'].reshape(count,size,-1).square().sum((1,2))
    return gain.detach().cpu().tolist()


def select_groups(stats,bank,source_width,count,policy,config,progress=lambda **kw:None,check_stop=lambda:None):
    whole = set(original_quartets(source_width))
    allowed = [i for i,q in enumerate(bank) if policy == 'free_roles'
               or policy == 'whole_joint' and q in whole
               or policy == 'role_fixed' and all(s//source_width == r for r,s in enumerate(q))]
    chosen,trace = [],[]
    for step in range(count):
        check_stop()
        used = {s for q in chosen for s in bank[q]}
        feasible = []
        for i in allowed:
            if used.intersection(bank[i]):
                continue
            touched = {s%source_width for s in used.union(bank[i])}
            if source_width-len(touched) >= count-step-1:
                feasible.append(i)
        if not feasible:
            raise RuntimeError('No disjoint completion; original quartets should guarantee feasibility')
        gains = conditional_gains(stats,chosen)
        winner = min(feasible,key=lambda i:(-gains[i],bank[i]))
        chosen.append(winner)
        trace.append({'phase':'greedy','step':step+1,'gain':gains[winner],'candidate':winner})
        progress(phase='greedy',selected=step+1,target=count)
    for sweep in range(config['exchange_sweeps']):
        for position in range(count):
            check_stop()
            base = chosen[:position]+chosen[position+1:]
            used = {s for q in base for s in bank[q]}
            feasible = [i for i in allowed if not used.intersection(bank[i])]
            gains = conditional_gains(stats,base)
            winner = min(feasible,key=lambda i:(-gains[i],bank[i]))
            current = chosen[position]
            tolerance = 1e-9*max(1.,abs(objective(stats,chosen)))
            if gains[winner] > gains[current]+tolerance:
                chosen[position] = winner
                trace.append({'phase':'exchange','sweep':sweep,'position':position,'old':current,'candidate':winner,
                              'improvement':gains[winner]-gains[current]})
            progress(phase='exchange',selected=position+1,target=count)
    packing_indices([bank[i] for i in chosen],source_width)
    return [bank[i] for i in chosen],{'trace':trace,'sampled_regularized_objective':objective(stats,chosen),
                                     'sampled_lambda':stats['penalty']}


@torch.no_grad()
def final_consumer_fit(block,pack,penalty,device,batch_size,check_stop=lambda:None):
    h = producer_features(block,pack['inputs'],device,batch_size,check_stop)
    g,b,squares,count = raw_moments(h,pack['targets'])
    qg,qb = q_normal_equations(g.numpy(),b.numpy(),block.conv2.out_channels)
    qg,qb = torch.from_numpy(qg),torch.from_numpy(qb)
    prior = torch.from_numpy(theta(block.conv2.kernel.detach().cpu().numpy()))
    values = prior+torch.linalg.solve(qg+penalty*torch.eye(len(qg),dtype=qg.dtype),qb-qg@prior)
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError('Nonfinite final consumer fit')
    block.conv2.kernel.copy_(torch.as_tensor(kernel_of(values.numpy(),tuple(block.conv2.kernel.shape))).to(block.conv2.kernel))
    return {'lambda':penalty,'fit_mse_float64':sse(qg.numpy(),qb.numpy(),squares,values.numpy())/count}


@torch.no_grad()
def local_mse(block,pack,device,batch_size):
    error,count = 0.,0
    for start in range(0,len(pack['inputs']),batch_size):
        x = pack['inputs'][start:start+batch_size].to(device)
        y = block.conv2(F.relu(block.bn1(block.conv1(x)))).cpu().double()
        target = pack['targets'][start:start+batch_size].double()
        error += float((y-target).square().sum()); count += target.numel()
    return error/count
