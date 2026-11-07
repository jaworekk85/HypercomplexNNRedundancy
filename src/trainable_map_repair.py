"""Epoch-resumable version of the existing supervised compact-model recipe."""
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.cnn_training import evaluate, prepare_batch, seed_everything


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def repair(model, tensors, labels, mean, std, config, seed, checkpoint,
           stop_path=None, on_epoch=None, after_batch=None):
    """Discard an interrupted partial epoch; restore completed-epoch RNG/state.

    after_batch is solely a technical interruption-test hook. Schedule advances
    on attempted steps, matching the frozen recipe; successful AMP steps logged.
    """
    device = mean.device
    seed_everything(seed)
    batches = (len(tensors["train"]) + config["batch_size"] - 1) // config["batch_size"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=batches*config["epochs"], eta_min=config["minimum_learning_rate"])
    scaler = torch.amp.GradScaler(device.type, enabled=config["mixed_precision"] and device.type == "cuda")
    order = torch.Generator().manual_seed(seed + config["order_seed_offset"])
    augmentation = torch.Generator(device=device).manual_seed(seed + config["augment_seed_offset"])
    history, count = [], {"successful": 0, "attempted": 0}
    optimizer.register_step_post_hook(lambda *args: count.__setitem__("successful", count["successful"] + 1))

    def save():
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "history": history, "count": count, "config": config, "seed": seed,
                     "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                             "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                             "order": order.get_state(), "augmentation": augmentation.get_state()}}, checkpoint)

    if Path(checkpoint).exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["config"] != config or saved["seed"] != seed:
            raise ValueError("Resume recipe changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        history, counts = saved["history"], saved["count"]
        count.update(counts)
        rng = saved["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if device.type == "cuda": torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        order.set_state(rng["order"].cpu())
        augmentation.set_state(rng["augmentation"].cpu())
    else:
        save()
    for epoch in range(len(history) + 1, config["epochs"] + 1):
        if stop_path is not None and Path(stop_path).exists(): return model, history, count, False
        start = time.perf_counter()
        model.train()
        loss_sum, correct = 0., 0
        for index in torch.randperm(len(tensors["train"]), generator=order).split(config["batch_size"]):
            index = index.to(device)
            x = prepare_batch(tensors["train"][index], mean, std, True, augmentation)
            y = labels["train"][index]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=scaler.is_enabled()):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            if not bool(torch.isfinite(loss)): raise RuntimeError("Nonfinite repair loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count["attempted"] += 1
            loss_sum += float(loss.detach()) * len(index)
            correct += int((logits.detach().argmax(1) == y).sum())
            if after_batch is not None: after_batch(count["attempted"])
            if stop_path is not None and Path(stop_path).exists(): return model, history, count, False
        row = {"epoch": epoch, "train_augmented_loss": loss_sum/len(tensors["train"]),
               "train_augmented_accuracy": correct/len(tensors["train"]),
               "learning_rate_end": optimizer.param_groups[0]["lr"],
               "epoch_seconds": time.perf_counter()-start}
        for split in ("validation", "check_calibration"):
            row[split] = evaluate(model, tensors[split], labels[split], mean, std, config["batch_size"])[0]
        history.append(row)
        save()
        if on_epoch is not None: on_epoch(row, count)
    return model.eval(), history, count, True
