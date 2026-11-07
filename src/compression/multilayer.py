"""Uniform multi-site structured compression of the small CNN families, applied shallow to deep.

A site is the output of one convolution+BN+ReLU (block ends also pool and drop out). Removing units
physically narrows the producer convolution, its BN and the consumer inputs (the next convolution,
or the linear readout after global average pooling). Optional compensation replaces every removed
unit by a ridge reconstruction from the kept units of the same site and folds it into the consumer.
Unconstrained multi-source folding is prior art (GRAIL2026, Kim2020, Asiaee2026CMR); the quaternion
variant restricts every map to a right-Hamilton block, q_v ~= sum_k q_k*a_k, so a convolutional
consumer stays a QuaternionConv2d. The readout is a real linear layer, so any 4x4 map folds exactly
there. Fits use uncentered second moments of the site features, which give the same ridge solutions
as src.compression.multisource on the raw values. Greedy backward elimination by reconstruction
residual and BN-folded magnitude are standard heuristics. No new method is claimed.
"""

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from src.compression.bn_magnitude import bn_group_scores
from src.compression.internal_merging import BASIS, expand_right, outgoing_addition
from src.compression.multisource import substitute_multi
from src.compression.quaternion_channels import component_indices
from src.models.quaternion_conv_torch import QuaternionConv2d


def sites(model):
    """(block, 0) feeds the block's second convolution; (block, 1) feeds the next block or the readout."""
    return [(block, layer) for block in range(len(model.blocks)) for layer in (0, 1)]


def producer(model, site):
    block, layer = site
    return model.blocks[block][3*layer], model.blocks[block][3*layer+1]


def consumer(model, site):
    block, layer = site
    if layer == 0:
        return model.blocks[block][3]
    return model.blocks[block+1][0] if block+1 < len(model.blocks) else model.readout


def _set_consumer(model, site, module):
    block, layer = site
    if layer == 0:
        model.blocks[block][3] = module
    elif block+1 < len(model.blocks):
        model.blocks[block+1][0] = module
    else:
        model.readout = module


def output_module(model, site):
    """Module whose output is exactly the consumer input in eval mode (dropout is the identity)."""
    block, layer = site
    return model.blocks[block][2] if layer == 0 else model.blocks[block]


def units(model, site):
    return producer(model, site)[0].out_channels


def kept_width(width, fraction):
    return width-round(fraction*width)


@torch.no_grad()
def compact_site(model, site, kept):
    """Separate compact model; kept indices refer to the site's current units."""
    kept = [int(unit) for unit in kept]
    width = units(model, site)
    if not kept or kept != sorted(set(kept)) or kept[0] < 0 or kept[-1] >= width:
        raise ValueError("Keep a nonempty sorted unique subset of the site's units")
    quaternion = model.kind == "quaternion"
    conv, bn = producer(model, site)
    old = consumer(model, site)
    if conv.bias is not None or (not isinstance(old, nn.Linear) and old.bias is not None):
        raise ValueError("Expected bias-free convolutions")
    if not bn.affine or not bn.track_running_stats:
        raise ValueError("Expected affine BN with running statistics")
    channels = component_indices(kept, width) if quaternion else kept
    result = deepcopy(model)
    block, layer = site
    if quaternion:
        narrow = QuaternionConv2d(conv.in_channels, len(kept), conv.kernel.shape[-1], conv.stride, conv.padding).to(conv.kernel)
        narrow.kernel.copy_(conv.kernel[:, kept])
    else:
        narrow = nn.Conv2d(conv.in_channels, len(kept), conv.kernel_size, conv.stride, conv.padding, bias=False).to(conv.weight)
        narrow.weight.copy_(conv.weight[kept])
    compact_bn = nn.BatchNorm2d(len(channels), eps=bn.eps, momentum=bn.momentum).to(bn.weight)
    for name in ("weight", "bias", "running_mean", "running_var"):
        getattr(compact_bn, name).copy_(getattr(bn, name)[channels])
    compact_bn.num_batches_tracked.copy_(bn.num_batches_tracked)
    result.blocks[block][3*layer], result.blocks[block][3*layer+1] = narrow, compact_bn
    if isinstance(old, nn.Linear):
        new = nn.Linear(len(channels), old.out_features, bias=old.bias is not None).to(old.weight)
        new.weight.copy_(old.weight[:, channels])
        if old.bias is not None:
            new.bias.copy_(old.bias)
    elif quaternion:
        new = QuaternionConv2d(len(kept), old.out_channels, old.kernel.shape[-1], old.stride, old.padding).to(old.kernel)
        new.kernel.copy_(old.kernel[:, :, kept])
    else:
        new = nn.Conv2d(len(kept), old.out_channels, old.kernel_size, old.stride, old.padding, bias=False).to(old.weight)
        new.weight.copy_(old.weight[:, kept])
    _set_consumer(result, site, new)
    return result.train(model.training)


@torch.no_grad()
def site_gram(model, site, inputs, batch_size=256):
    """Uncentered second moments (channels x channels, float64) of the site features over all positions."""
    model.eval()
    state = {"gram": None, "rows": 0}

    def hook(module, args, output):
        values = output.detach().permute(0, 2, 3, 1).reshape(-1, output.shape[1]).double()
        state["gram"] = values.T@values if state["gram"] is None else state["gram"]+values.T@values
        state["rows"] += len(values)
    handle = output_module(model, site).register_forward_hook(hook)
    try:
        for start in range(0, len(inputs), batch_size):
            model(inputs[start:start+batch_size])
    finally:
        handle.remove()
    return state["gram"].cpu().numpy(), state["rows"]


class SiteMoments:
    """Ridge fits from uncentered moments; lambda = alpha * mean Gram diagonal, as in multisource.ridge."""

    def __init__(self, gram, rows, quaternion):
        gram = np.asarray(gram, dtype=np.float64)
        self.quaternion, self.rows = quaternion, rows
        if quaternion:
            width = len(gram)//4
            # [unit, component, unit, component] from component-blocked channels c*width+u.
            self.moments = gram.reshape(4, width, 4, width).transpose(1, 0, 3, 2)
        else:
            self.moments = gram
        self.units = len(self.moments)

    @staticmethod
    def _solve(gram, rhs, alpha):
        penalty = alpha*float(np.mean(np.diag(gram))) if alpha > 0 else 0.
        return np.linalg.solve(gram+penalty*np.eye(len(gram)), rhs)

    def fit(self, victim, kept, alpha, mode):
        """Returns (map, residual mean squared error per scalar equation) for one victim."""
        kept, m = list(kept), self.moments
        if mode == "real":
            if self.quaternion:
                raise ValueError("Scalar fits need real features")
            gram, rhs = m[np.ix_(kept, kept)], m[kept, victim]
            solution = self._solve(gram, rhs, alpha)
            return solution, float(m[victim, victim]-2*solution@rhs+solution@gram@solution)/self.rows
        if not self.quaternion:
            raise ValueError("Quaternion fits need quaternion features")
        sources, cross = m[kept][:, :, kept], m[kept][:, :, victim]  # (k,4,k,4) and [k,i,o]
        target = float(np.trace(m[victim, :, victim]))
        if mode == "quaternion_right":
            gram = np.einsum("coi,doj,kilj->kcld", BASIS, BASIS, sources).reshape(4*len(kept), -1)
            rhs = np.einsum("coi,kio->kc", BASIS, cross).reshape(-1)
            solution = self._solve(gram, rhs, alpha)
            sse = target-2*solution@rhs+solution@gram@solution
            return expand_right(solution.reshape(len(kept), 4)), float(sse)/(4*self.rows)
        if mode == "full_real":
            gram, rhs = sources.reshape(4*len(kept), -1), cross.reshape(4*len(kept), 4)
            solution = self._solve(gram, rhs, alpha)
            sse = target-2*np.sum(solution*rhs)+np.trace(solution.T@gram@solution)
            return solution.reshape(len(kept), 4, 4).transpose(0, 2, 1), float(sse)/(4*self.rows)
        raise ValueError("Unknown fit family")


def site_maps(moments, victims, alpha, mode):
    """Maps in the src.compression.multisource format; sources are all non-victim units."""
    kept = sorted(set(range(moments.units))-set(victims))
    result = []
    for victim in sorted(victims):
        value = moments.fit(victim, kept, alpha, mode)[0]
        result.append({"victim": int(victim), "kept": kept, "mode": mode,
                       ("matrices" if moments.quaternion else "coefficients"): value.tolist()})
    return result


def greedy_victims(moments, count, energy, alpha, mode):
    """Backward elimination: repeatedly drop the unit whose ridge reconstruction from the remaining
    units has the smallest residual MSE times outgoing consumer energy (multi-source analogue of the
    pairwise ranking). Exact ties favor lower indices."""
    remaining, victims = list(range(moments.units)), []
    if not 0 <= count < len(remaining):
        raise ValueError("Keep at least one unit")
    for _ in range(count):
        scores = [(moments.fit(unit, [other for other in remaining if other != unit], alpha, mode)[1]*energy[unit], unit) for unit in remaining]
        victims.append(min(scores)[1])
        remaining.remove(victims[-1])
    return sorted(victims)


def consumer_energy(model, site):
    """Squared outgoing weight norm per unit of the site."""
    old, width = consumer(model, site), units(model, site)
    if isinstance(old, nn.Linear):
        weight = np.square(old.weight.detach().cpu().double().numpy())
        return weight.reshape(len(weight), 4, width).sum(axis=(0, 1)) if model.kind == "quaternion" else weight.sum(0)
    if model.kind == "quaternion":
        return np.square(old.kernel.detach().cpu().double().numpy()).sum(axis=(0, 1, 3, 4))
    return np.square(old.weight.detach().cpu().double().numpy()).sum(axis=(0, 2, 3))


@torch.no_grad()
def magnitude_scores(model, site):
    """BN-folded kernel magnitude per unit: src.compression.bn_magnitude for quaternions, its scalar analogue for real CNNs."""
    conv, bn = producer(model, site)
    if model.kind == "quaternion":
        return np.asarray(bn_group_scores(conv, bn)["bn_folded_group_l2"])
    if bn.training:
        raise ValueError("Use evaluation-mode BN")
    scale = bn.weight.detach().cpu().double()*torch.rsqrt(bn.running_var.detach().cpu().double()+bn.eps)
    return (conv.weight.detach().cpu().double().square().sum(dim=(1, 2, 3)).sqrt()*scale.abs()).numpy()


def magnitude_victims(model, site, count):
    scores = magnitude_scores(model, site)
    return sorted(sorted(range(len(scores)), key=lambda unit: (scores[unit], unit))[:count])


@torch.no_grad()
def fold(source, compact, site, maps):
    """Add every map into the compact consumer; returns the energy fraction lost to Hamilton projection."""
    old, new, width = consumer(source, site), consumer(compact, site), units(source, site)
    quaternion = source.kind == "quaternion"
    discarded = energy = 0.
    if isinstance(old, nn.Linear):
        weight = old.weight.detach().double()
        for row in maps:
            if quaternion:
                outgoing = weight[:, component_indices([row["victim"]], width)]
                grouped = new.weight.view(len(new.weight), 4, len(row["kept"]))
                for position, matrix in enumerate(row["matrices"]):
                    grouped[:, :, position].add_((outgoing@torch.as_tensor(matrix, dtype=torch.float64, device=weight.device)).to(new.weight.dtype))
            else:
                coefficients = torch.as_tensor(row["coefficients"], dtype=torch.float64, device=weight.device)
                new.weight.add_((weight[:, row["victim"]][:, None]*coefficients[None]).to(new.weight.dtype))
    elif quaternion:
        kernel = old.kernel.detach().cpu().double().numpy()
        for row in maps:
            for position, matrix in enumerate(np.asarray(row["matrices"])):
                addition, diagnostic = outgoing_addition(kernel, row["victim"], matrix)
                new.kernel[:, :, position].add_(torch.as_tensor(addition, dtype=new.kernel.dtype, device=new.kernel.device))
                discarded += diagnostic["discarded_squared_norm"]
                energy += diagnostic["full_squared_norm"]
    else:
        weight = old.weight.detach()
        for row in maps:
            coefficients = torch.as_tensor(row["coefficients"], dtype=weight.dtype, device=weight.device)
            new.weight.add_(coefficients[None, :, None, None]*weight[:, row["victim"]][:, None])
    return discarded/energy if energy > 1e-20 else 0.


def compress_site(model, site, victims, maps=None):
    """Remove victims at one site, optionally folding their maps; returns (compact, kept, projection loss)."""
    victims = sorted(int(unit) for unit in victims)
    kept = sorted(set(range(units(model, site)))-set(victims))
    compact = compact_site(model, site, kept)
    lost = 0.
    if maps:
        if any(row["kept"] != kept for row in maps) or sorted(row["victim"] for row in maps) != victims:
            raise ValueError("Maps must cover exactly the victims, with the kept units as sources")
        lost = fold(model, compact, site, maps)
    return compact.eval(), kept, lost


@torch.no_grad()
def substituted_logits(model, site, maps, inputs, batch_size=256):
    """Functional reference: the uncompressed network with victim features replaced by their reconstruction."""
    model.eval()
    quaternion = model.kind == "quaternion"
    handle = output_module(model, site).register_forward_hook(lambda module, args, output: substitute_multi(output, maps, quaternion))
    try:
        return torch.cat([model(inputs[start:start+batch_size]) for start in range(0, len(inputs), batch_size)])
    finally:
        handle.remove()
