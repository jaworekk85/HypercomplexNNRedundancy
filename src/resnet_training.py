"""Fixed-recipe SGD baseline training with exact completed-epoch resume."""
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.cnn_training import evaluate, prepare_batch, seed_everything
from src.trainable_map_repair import atomic_save


def learning_rate(config, epoch):
    warmup = config["warmup_epochs"]
    if epoch <= warmup:
        return config["warmup_start_lr"]+(config["learning_rate"]-config["warmup_start_lr"])*(epoch-1)/max(1, warmup-1)
    return config["learning_rate"]*config["gamma"]**sum(epoch >= milestone for milestone in config["milestones"])


def train(model, tensors, labels, mean, std, config, checkpoint, stop=None, on_epoch=None, after_batch=None):
    device = mean.device
    seed_everything(config["seed"])
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler(device.type, enabled=config["mixed_precision"] and device.type == "cuda")
    order = torch.Generator().manual_seed(config["seed"]+config["order_seed_offset"])
    augmentation = torch.Generator(device=device).manual_seed(config["seed"]+config["augment_seed_offset"])
    history, counts = [], {"attempted": 0, "successful": 0}
    best = {"epoch": None, "validation": {"cross_entropy": float("inf")}, "model": None}
    optimizer.register_step_post_hook(lambda *args: counts.__setitem__("successful", counts["successful"]+1))

    def save():
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                     "history": history, "counts": counts, "best": best, "config": config,
                     "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                             "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                             "order": order.get_state(), "augmentation": augmentation.get_state()}}, checkpoint)

    if Path(checkpoint).exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["config"] != config: raise ValueError("ResNet resume recipe changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        history, best = saved["history"], saved["best"]
        counts.update(saved["counts"])
        rng = saved["rng"]
        random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if device.type == "cuda": torch.cuda.set_rng_state_all(rng["cuda"])
        order.set_state(rng["order"]); augmentation.set_state(rng["augmentation"])
    else: save()
    for epoch in range(len(history)+1, config["epochs"]+1):
        if stop is not None and Path(stop).exists(): return None, {"history": history, "counts": counts}, False
        start = time.perf_counter()
        lr = learning_rate(config, epoch)
        for group in optimizer.param_groups: group["lr"] = lr
        model.train()
        loss_sum, correct, successful_before = 0., 0, counts["successful"]
        for index in torch.randperm(len(tensors["train"]), generator=order).split(config["batch_size"]):
            index = index.to(device)
            x = prepare_batch(tensors["train"][index], mean, std, True, augmentation)
            y = labels["train"][index]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=scaler.is_enabled()):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            if not bool(torch.isfinite(loss)): raise RuntimeError("Nonfinite ResNet loss; no automatic recipe change")
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            counts["attempted"] += 1
            loss_sum += float(loss.detach())*len(index)
            correct += int((logits.detach().argmax(1) == y).sum())
            if after_batch is not None: after_batch(counts["attempted"])
            if stop is not None and Path(stop).exists(): return None, {"history": history, "counts": counts}, False
        if counts["successful"] == successful_before: raise RuntimeError("All updates in epoch skipped")
        validation = evaluate(model, tensors["validation"], labels["validation"], mean, std, config["batch_size"])[0]
        if not np.isfinite(validation["cross_entropy"]): raise RuntimeError("Nonfinite validation loss")
        if validation["cross_entropy"] < best["validation"]["cross_entropy"]:
            best = {"epoch": epoch, "validation": validation,
                    "model": {n: value.detach().cpu().clone() for n, value in model.state_dict().items()}}
        row = {"epoch": epoch, "train_augmented_loss": loss_sum/len(tensors["train"]),
               "train_augmented_accuracy": correct/len(tensors["train"]), "validation": validation,
               "learning_rate": lr, "epoch_seconds": time.perf_counter()-start,
               "successful_optimizer_steps": counts["successful"], "attempted_optimizer_steps": counts["attempted"]}
        history.append(row)
        save()
        if on_epoch is not None: on_epoch(row, best)
    model.load_state_dict(best["model"])
    return model.eval(), {"history": history, "counts": counts, "selected_epoch": best["epoch"],
                          "selected_validation": best["validation"], "final_validation": history[-1]["validation"]}, True
