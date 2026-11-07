"""Equal-budget fine-tuning of compact models with a shared batch/augmentation stream."""

import time

import torch
from torch.nn import functional as F

from src.cnn_training import evaluate, prepare_batch, seed_everything


def finetune(model, tensors, labels, mean, std, config, seed, evaluation_splits=("validation",)):
    """Train all parameters for config["epochs"] epochs; every policy of one training seed sees the same batches.

    Per-step cosine schedule from learning_rate to minimum_learning_rate. BN statistics update in train mode.
    No checkpoint selection: the final epoch is the prespecified outcome; per-epoch held-out values are logged only.
    """
    device = mean.device
    seed_everything(seed)
    steps = config["epochs"]*((len(tensors["train"])+config["batch_size"]-1)//config["batch_size"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=config["minimum_learning_rate"])
    scaler = torch.amp.GradScaler(device.type, enabled=config["mixed_precision"] and device.type == "cuda")
    order_rng = torch.Generator().manual_seed(seed+config["order_seed_offset"])
    augment_rng = torch.Generator(device=device).manual_seed(seed+config["augment_seed_offset"])
    history = []
    for epoch in range(1, config["epochs"]+1):
        start = time.perf_counter()
        model.train()
        loss_sum, correct = 0., 0
        for index in torch.randperm(len(tensors["train"]), generator=order_rng).split(config["batch_size"]):
            index = index.to(device)
            x = prepare_batch(tensors["train"][index], mean, std, True, augment_rng)
            y = labels["train"][index]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=scaler.is_enabled()):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite fine-tuning loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.detach())*len(index)
            correct += int((logits.detach().argmax(1) == y).sum())
        row = {"epoch": epoch, "train_augmented_loss": loss_sum/len(tensors["train"]), "train_augmented_accuracy": correct/len(tensors["train"]),
               "learning_rate_end": optimizer.param_groups[0]["lr"], "epoch_seconds": time.perf_counter()-start}
        for split in evaluation_splits:
            row[split] = evaluate(model, tensors[split], labels[split], mean, std, config["batch_size"])[0]
        history.append(row)
    return model.eval(), history
