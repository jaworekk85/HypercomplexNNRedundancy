"""Physical QNN versus unrestricted-consumer compaction, with optional BN.

Four-component feature groups retain component-blocked order. Unrestricted
maps are folded into ordinary dense Conv2d consumers, never projected back.
The first convolution stays quaternion: it is not a consumer of any removed
hidden feature. This module makes no new-method claim (GRAIL2026/Kim2020).
"""
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from src.compression.internal_merging import outgoing_addition
from src.compression import multilayer as old
from src.compression.quaternion_channels import component_indices
from src.models.quaternion_conv_torch import QuaternionConv2d, hamilton_kernel
from src.models.cifar_cnn_torch import CifarCNN
from src.models.eurosat_cnn import FourBandCNN


def group_width(layer):
    if isinstance(layer, QuaternionConv2d):
        return layer.out_channels
    if not isinstance(layer, nn.Conv2d) or layer.out_channels % 4:
        raise ValueError("Expected four-component output groups")
    return layer.out_channels // 4


@torch.no_grad()
def dense_conv(layer):
    if isinstance(layer, nn.Conv2d):
        return deepcopy(layer)
    if not isinstance(layer, QuaternionConv2d) or layer.bias is not None:
        raise ValueError("Expected bias-free quaternion convolution")
    result = nn.Conv2d(4*layer.in_channels, 4*layer.out_channels, layer.kernel.shape[-1],
                       stride=layer.stride, padding=layer.padding, bias=False).to(layer.kernel)
    result.weight.copy_(hamilton_kernel(layer.kernel))
    return result.train(layer.training)


def release_consumers(model):
    """Same function initially; independent real weights at all five conv consumers."""
    result = deepcopy(model)
    for block, layer in old.sites(result)[1:]:
        index = 3*layer
        result.blocks[block][index] = dense_conv(result.blocks[block][index])
    return result


def make_model(config, dataset, normalization, family):
    cls = CifarCNN if dataset == "cifar" else FourBandCNN
    model = cls("quaternion", config["quaternion_widths"], config["real_widths"], config["dropout"])
    if normalization == "none":
        for block in model.blocks:
            for index in (1, 4):
                block[index] = nn.Identity()
    elif normalization != "bn":
        raise ValueError(normalization)
    if family == "free":
        model = release_consumers(model)
    elif family != "quaternion":
        raise ValueError(family)
    return model


@torch.no_grad()
def compact(model, site, victims, maps=None, family="quaternion"):
    producer, norm = old.producer(model, site)
    width = group_width(producer)
    victims = sorted(int(i) for i in victims)
    if len(victims) != len(set(victims)) or any(i < 0 or i >= width for i in victims):
        raise ValueError("Invalid victims")
    kept = sorted(set(range(width))-set(victims))
    if not kept: raise ValueError("Keep at least one unit")
    channels = component_indices(kept, width)
    consumer = old.consumer(model, site)
    if producer.bias is not None or (not isinstance(consumer, nn.Linear) and consumer.bias is not None):
        raise ValueError("Expected bias-free convolutions")
    result = deepcopy(model)
    block, layer = site
    if isinstance(producer, QuaternionConv2d):
        narrow = QuaternionConv2d(producer.in_channels, len(kept), producer.kernel.shape[-1], producer.stride, producer.padding).to(producer.kernel)
        narrow.kernel.copy_(producer.kernel[:, kept])
    else:
        narrow = nn.Conv2d(producer.in_channels, len(channels), producer.kernel_size, producer.stride, producer.padding, bias=False).to(producer.weight)
        narrow.weight.copy_(producer.weight[channels])
    result.blocks[block][3*layer] = narrow
    if isinstance(norm, nn.BatchNorm2d):
        new_norm = nn.BatchNorm2d(len(channels), eps=norm.eps, momentum=norm.momentum).to(norm.weight)
        for name in ("weight", "bias", "running_mean", "running_var"):
            getattr(new_norm, name).copy_(getattr(norm, name)[channels])
        new_norm.num_batches_tracked.copy_(norm.num_batches_tracked)
    elif isinstance(norm, nn.Identity):
        new_norm = nn.Identity()
    else:
        raise ValueError("Only component BN or Identity supported")
    result.blocks[block][3*layer+1] = new_norm
    if maps is not None and (sorted(r["victim"] for r in maps) != victims or any(r["kept"] != kept for r in maps)):
        raise ValueError("Maps must cover all victims and exactly the kept sources")
    if isinstance(consumer, nn.Linear):
        new = nn.Linear(len(channels), consumer.out_features, bias=consumer.bias is not None).to(consumer.weight)
        new.weight.copy_(consumer.weight[:, channels])
        if consumer.bias is not None: new.bias.copy_(consumer.bias)
        for row in maps or []:
            outgoing = consumer.weight[:, component_indices([row["victim"]], width)].double()
            for pos, matrix in enumerate(row["matrices"]):
                columns = component_indices([pos], len(kept))
                addition = outgoing @ torch.as_tensor(matrix, dtype=torch.float64, device=outgoing.device)
                new.weight[:, columns] += addition.to(new.weight.dtype)
    elif family == "free":
        expanded = dense_conv(consumer)
        new = nn.Conv2d(len(channels), expanded.out_channels, expanded.kernel_size, expanded.stride, expanded.padding, bias=False).to(expanded.weight)
        new.weight.copy_(expanded.weight[:, channels])
        for row in maps or []:
            outgoing = expanded.weight[:, component_indices([row["victim"]], width)].double()
            for pos, matrix in enumerate(row["matrices"]):
                addition = torch.einsum("oihw,ij->ojhw", outgoing, torch.as_tensor(matrix, dtype=torch.float64, device=outgoing.device))
                new.weight[:, component_indices([pos], len(kept))] += addition.to(new.weight.dtype)
    elif family == "quaternion" and isinstance(consumer, QuaternionConv2d):
        new = QuaternionConv2d(len(kept), consumer.out_channels, consumer.kernel.shape[-1], consumer.stride, consumer.padding).to(consumer.kernel)
        new.kernel.copy_(consumer.kernel[:, :, kept])
        kernel = consumer.kernel.detach().cpu().double().numpy()
        for row in maps or []:
            for pos, matrix in enumerate(row["matrices"]):
                addition, check = outgoing_addition(kernel, row["victim"], matrix)
                if check["discarded_energy_fraction"] > 1e-20:
                    raise ValueError("Map cannot be folded without breaking quaternion structure")
                new.kernel[:, :, pos] += torch.as_tensor(addition, dtype=new.kernel.dtype, device=new.kernel.device)
    else:
        raise ValueError("Quaternion path requires quaternion consumer")
    old._set_consumer(result, site, new)
    return result.eval()


def describe(model):
    convs = [old.producer(model, site)[0] for site in old.sites(model)]
    return {"real_parameters": sum(p.numel() for p in model.parameters()),
            "quaternion_convolutions": sum(isinstance(c, QuaternionConv2d) for c in convs),
            "dense_real_convolutions": sum(isinstance(c, nn.Conv2d) for c in convs),
            "output_group_widths": [group_width(c) for c in convs],
            "bn_layers": sum(isinstance(m, nn.BatchNorm2d) for m in model.modules())}
