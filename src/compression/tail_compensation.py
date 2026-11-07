"""One late Q-consumer calibration pilot; functional distillation is prior art."""
from copy import deepcopy
import time

import torch
from torch.nn import functional as F

from src.compression import multilayer as old
from src.models.quaternion_conv_torch import QuaternionConv2d


def logits_from_consumer(model, values):
    """Tail after the sixth/last convolution in the existing three-block CNN."""
    for module in list(model.blocks[2].children())[4:]:
        values = module(values)
    return model.readout(values.mean(dim=(2, 3)))


@torch.no_grad()
def capture(reference, student, images, batch_size=32):
    if len(student.blocks) != 3 or not isinstance(old.consumer(student, (2, 0)), QuaternionConv2d):
        raise ValueError("This bounded pilot expects the existing last Q consumer")
    reference.eval(); student.eval()
    rows = {"inputs": [], "local": [], "logits": []}
    state = {}
    target = old.consumer(reference, (2, 0))
    consumer = old.consumer(student, (2, 0))
    handles = [target.register_forward_hook(lambda m, x, y: state.__setitem__("local", y.detach())),
               consumer.register_forward_pre_hook(lambda m, x: state.__setitem__("inputs", x[0].detach()))]
    discrepancies = []
    try:
        for x in images.split(batch_size):
            rows["logits"].append(reference(x).detach())
            expected = student(x)
            rows["inputs"].append(state["inputs"])
            rows["local"].append(state["local"])
            replay = logits_from_consumer(student, consumer(state["inputs"]))
            discrepancies.append(float((expected-replay).abs().max()))
    finally:
        for handle in handles: handle.remove()
    error = max(discrepancies)
    if error > 1e-4: raise AssertionError("Cached tail does not replay the full model")
    return {**{name: torch.cat(values) for name, values in rows.items()}, "images": len(images), "tail_replay_max_error": error}


@torch.no_grad()
def errors(model, pack, batch_size=32):
    local_sse = logits_sse = 0.
    local_count = logits_count = 0
    consumer = old.consumer(model, (2, 0))
    for start in range(0, pack["images"], batch_size):
        stop = start+batch_size
        z = consumer(pack["inputs"][start:stop])
        logits = logits_from_consumer(model, z)
        local_sse += float((z-pack["local"][start:stop]).double().square().sum())
        logits_sse += float((logits-pack["logits"][start:stop]).double().square().sum())
        local_count += z.numel(); logits_count += logits.numel()
    return {"local_mse": local_sse/local_count, "logit_mse": logits_sse/logits_count}


def refine(student, fit, check, mode, config):
    if mode not in ("local_extra", "tail_logits"): raise ValueError(mode)
    result = deepcopy(student).eval()
    for parameter in result.parameters(): parameter.requires_grad_(False)
    kernel = old.consumer(result, (2, 0)).kernel
    kernel.requires_grad_(True)
    initial = kernel.detach().clone()
    denominator = initial.square().sum().clamp_min(1e-12)
    before = {name: value.detach().clone() for name, value in result.state_dict().items()}
    initial_metrics = {"fit": errors(result, fit, config["batch_size"]), "check": errors(result, check, config["batch_size"])}
    optimizer = torch.optim.Adam([kernel], lr=config["learning_rate"])
    generator = torch.Generator().manual_seed(config["order_seed"])
    batches = []
    while len(batches) < config["steps"]:
        batches.extend(torch.randperm(fit["images"], generator=generator).split(config["batch_size"]))
    history, first = [], None
    start_time = time.perf_counter()
    for step, indices in enumerate(batches[:config["steps"]], 1):
        indices = indices.to(kernel.device)
        optimizer.zero_grad(set_to_none=True)
        z = old.consumer(result, (2, 0))(fit["inputs"][indices])
        predicted = z if mode == "local_extra" else logits_from_consumer(result, z)
        targets = fit["local" if mode == "local_extra" else "logits"][indices]
        reconstruction = F.mse_loss(predicted, targets)
        penalty = (kernel-initial).square().sum()/denominator
        loss = reconstruction+config["relative_ridge"]*penalty
        if not bool(torch.isfinite(loss)): raise ValueError("Nonfinite calibration objective")
        loss.backward()
        if kernel.grad is None or not bool(torch.isfinite(kernel.grad).all()): raise ValueError("Missing/nonfinite Q gradient")
        if step == 1:
            gradient_norms = kernel.grad.flatten(1).norm(dim=1).tolist()
        optimizer.step()
        if step == 1:
            updates = (kernel.detach()-initial).flatten(1).norm(dim=1).tolist()
            if not all(v > 0 for v in gradient_norms+updates): raise AssertionError("A Q component did not receive a gradient/update")
            first = {"gradient_component_norms": gradient_norms, "update_component_norms": updates, "passed": True}
        if step == 1 or step % 25 == 0 or step == config["steps"]:
            history.append({"step": step, "minibatch_reconstruction_mse": float(reconstruction.detach()), "relative_displacement": float(penalty.detach()), "objective": float(loss.detach())})
    seconds = time.perf_counter()-start_time
    changed = [name for name, value in result.state_dict().items() if not torch.equal(value, before[name])]
    if changed != ["blocks.2.3.kernel"]: raise AssertionError(f"Unexpected state changes: {changed}")
    final_metrics = {"fit": errors(result, fit, config["batch_size"]), "check": errors(result, check, config["batch_size"])}
    for parameter in result.parameters(): parameter.requires_grad_(True)
    return result, {"mode": mode, "initial": initial_metrics, "final": final_metrics, "history": history,
                    "first_update": first, "only_target_kernel_changed": True, "bn_buffers_and_head_unchanged": True,
                    "steps": config["steps"], "successful_optimizer_steps": config["steps"], "seconds": seconds,
                    "fit_images": fit["images"], "check_images": check["images"], "labels_used": False,
                    "teacher": "Current pre-site4 reference; not the pristine full source", "checkpoint_selection": "Fixed final calibration step"}
