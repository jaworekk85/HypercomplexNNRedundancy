"""Small image-CNN training helpers with explicit seeds and augmentation."""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import numpy as np
import torch
from torch.nn import functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def prepare_batch(images, mean, std, augment=False, generator=None):
    images = images.float() / 255.0
    if augment:
        n, _, height, width = images.shape
        padded = F.pad(images, (4, 4, 4, 4)).permute(0, 2, 3, 1)
        row = torch.randint(9, (n, 1, 1), device=images.device, generator=generator) + torch.arange(height, device=images.device)[None, :, None]
        col = torch.randint(9, (n, 1, 1), device=images.device, generator=generator) + torch.arange(width, device=images.device)[None, None, :]
        images = padded[torch.arange(n, device=images.device)[:, None, None], row, col].permute(0, 3, 1, 2)
        flip = torch.rand((n, 1, 1, 1), device=images.device, generator=generator) < 0.5
        images = torch.where(flip, images.flip(-1), images)
    return (images - mean) / std


@torch.no_grad()
def evaluate(model, images, labels, mean, std, batch_size=256):
    model.eval()
    total_loss, correct, logits = 0.0, 0, []
    for start in range(0, len(images), batch_size):
        outputs = model(prepare_batch(images[start:start + batch_size], mean, std))
        target = labels[start:start + batch_size]
        total_loss += float(F.cross_entropy(outputs, target, reduction="sum"))
        correct += int((outputs.argmax(1) == target).sum())
        logits.append(outputs.cpu().numpy())
    return {"cross_entropy": total_loss / len(images), "accuracy": correct / len(images)}, np.concatenate(logits)


def first_update_audit(model, inputs, labels, optimizer):
    from src.models.quaternion_conv_torch import QuaternionConv2d
    model.train()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    rows = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"Missing/nonfinite gradient: {name}")
        rows.append({"name": name, "shape": list(parameter.shape), "gradient_norm": float(parameter.grad.norm())})
    quaternion = {}
    for name, layer in model.named_modules():
        if isinstance(layer, QuaternionConv2d):
            quaternion[name] = {"component_gradient_norms": layer.kernel.grad.flatten(1).norm(dim=1).tolist()}
    optimizer.step()
    for row, (name, parameter) in zip(rows, model.named_parameters()):
        row["max_abs_update"] = float((parameter.detach() - before[name]).abs().max())
    for name, layer in model.named_modules():
        if isinstance(layer, QuaternionConv2d):
            quaternion[name]["component_update_norms"] = (layer.kernel.detach() - before[f"{name}.kernel"]).flatten(1).norm(dim=1).tolist()
    checks = {"all_variables_updated": all(row["max_abs_update"] > 0 for row in rows),
              "all_quaternion_components_have_gradients_and_updates": all(value > 0 for item in quaternion.values() for values in item.values() for value in values)}
    if not all(checks.values()):
        raise RuntimeError(f"First-update audit failed: {checks}")
    return {"parameters": rows, "quaternion_layers": quaternion, "checks": checks,
            "loss_before": float(loss.detach()), "correct_before": int((outputs.detach().argmax(1) == labels).sum())}
