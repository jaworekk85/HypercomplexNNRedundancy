"""BN/marginal-statistics interventions with frozen trainable parameters."""
from copy import deepcopy
from hashlib import sha256

import torch
from torch import nn

from src.cnn_training import prepare_batch


class ReachedLayer(Exception):
    pass


def parameter_hash(model):
    digest = sha256()
    for name,value in model.named_parameters():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def recalibrate_bn(model,images,mean,std,batch,stop=lambda:None,progress=lambda **kw:None):
    result = deepcopy(model).eval()
    before = parameter_hash(result)
    records = []
    for name,layer in result.named_modules():
        if not isinstance(layer,nn.BatchNorm2d):
            continue
        stop()
        total = torch.zeros(layer.num_features,dtype=torch.float64,device=mean.device)
        squares = torch.zeros_like(total)
        count,batches = 0,0
        def hook(module,args):
            nonlocal count,batches
            x = args[0].double()
            total.add_(x.sum((0,2,3)))
            squares.add_(x.square().sum((0,2,3)))
            count += x.shape[0]*x.shape[2]*x.shape[3]
            batches += 1
            raise ReachedLayer()
        handle = layer.register_forward_pre_hook(hook)
        try:
            for images_batch in images.split(batch):
                stop()
                try:
                    result(prepare_batch(images_batch,mean,std))
                except ReachedLayer:
                    pass
        finally:
            handle.remove()
        if not count:
            raise RuntimeError('BN layer was not reached')
        mu = total/count
        variance = (squares/count-mu.square()).clamp_min(0)
        if not bool(torch.isfinite(mu).all() and torch.isfinite(variance).all()):
            raise RuntimeError('Nonfinite BN moments')
        layer.running_mean.copy_(mu.to(layer.running_mean))
        layer.running_var.copy_(variance.to(layer.running_var))
        layer.num_batches_tracked.fill_(batches)
        records.append({'layer':name,'observations_per_channel':count,'batches':batches,
                        'mean':mu.cpu().tolist(),'population_variance':variance.cpu().tolist()})
        progress(layer=name,completed=len(records))
    if parameter_hash(result) != before:
        raise AssertionError('Recalibration changed a trainable parameter')
    return result,{'parameter_hash_before':before,'parameter_hash_after':parameter_hash(result),
                   'trainable_parameters_unchanged':True,'population_moments':True,'layers':records}


@torch.no_grad()
def collect_consumer(model,path,images,mean,std,batch,stop=lambda:None):
    layer = model.get_submodule(path)
    features,targets = [],[]
    def hook(module,args,out):
        features.append(args[0].detach().cpu())
        targets.append(out.detach().cpu())
        raise ReachedLayer()
    handle = layer.register_forward_hook(hook)
    try:
        model.eval()
        for x in images.split(batch):
            stop()
            try:
                model(prepare_batch(x,mean,std))
            except ReachedLayer:
                pass
    finally:
        handle.remove()
    if not features:
        raise RuntimeError('Consumer was not reached')
    return {'features':torch.cat(features),'targets':torch.cat(targets)}


def standardize_packs(packs):
    fit = packs['fit']['features'].double()
    mean = fit.mean((0,2,3),keepdim=True)
    variance = fit.var((0,2,3),correction=0,keepdim=True)
    scale = variance.sqrt().clamp_min(1e-6)
    result = {s:{'features':((p['features'].double()-mean)/scale).float()} for s,p in packs.items()}
    return result,{'fit_mean':mean.flatten().tolist(),'fit_scale':scale.flatten().tolist(),
                   'constant_channels':int((variance < 1e-12).sum()),'fit_only':True}


def add_centered_normalization(rows,targets):
    mu = targets['fit'].double().mean((0,2,3),keepdim=True)
    energies = {s:float((y.double()-mu).square().mean()) for s,y in targets.items()}
    if min(energies.values()) <= 0:
        raise ValueError('Degenerate centered target energy')
    result = deepcopy(rows)
    for row in result:
        row['fit_centered_normalized_mse'] = row['fit_mse']/energies['fit']
        row['check_centered_normalized_mse'] = row['check_mse']/energies['check_calibration']
    return result,{'fit_channel_target_mean':mu.flatten().tolist(),'centered_energy':energies,
                   'centering_fitted_on_fit_only':True,'regression_intercept_added':False}
