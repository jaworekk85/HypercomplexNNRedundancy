"""Train a small CIFAR CNN on GPU, with checkpoints and held-out evaluation."""

import argparse
import csv
import gc
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

from src.cnn_training import evaluate, first_update_audit, prepare_batch, seed_everything
from src.data.cifar_cached import channel_statistics, file_sha256, load_cached_cifar
from src.models.cifar_cnn_torch import CifarCNN

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["scripts/cifar_cnn.py", "src/cnn_training.py", "src/data/cifar_cached.py",
           "src/models/quaternion_conv_torch.py", "src/models/cifar_cnn_torch.py", "src/evaluation/cnn_report.py",
           "src/activations/diagnostics.py", "src/activations/similarity.py"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_model(config, kind):
    return CifarCNN(kind, config["quaternion_widths"], config["real_widths"], config["dropout"])


def source_hashes():
    return {name: file_sha256(ROOT / name) for name in SOURCES}


def benchmark(config, output_dir):
    device = torch.device(config["device"])
    results = {}
    for kind in config["models"]:
        seed_everything(config["seed"])
        model = make_model(config, kind).to(device)
        x = torch.randn(config["batch_size"], 3, 32, 32, device=device)
        y = torch.arange(config["batch_size"], device=device) % 10
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        audit = first_update_audit(model, x, y, optimizer)
        scaler = torch.amp.GradScaler(device.type, enabled=config["mixed_precision"] and device.type == "cuda")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=scaler.is_enabled()):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = (time.perf_counter() - start) / 20
        results[kind] = {"seconds_per_training_batch": seconds, "real_trainable_parameters": sum(p.numel() for p in model.parameters()),
                         "estimated_training_seconds_per_48000_image_epoch": seconds * (48000 / config["batch_size"]),
                         "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                         "audit": audit}
        print(f"{kind}: {seconds:.4f}s/batch, {results[kind]['real_trainable_parameters']} parameters", flush=True)
        del model, optimizer, x, y
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(output_dir / "benchmark.json", results)


@torch.no_grad()
def collect(model, images, mean, std, batch_size=128):
    model.eval()
    result = []
    for start in range(0, len(images), batch_size):
        values = model.forward_features(prepare_batch(images[start:start + batch_size], mean, std))
        if model.kind == "quaternion":
            n, channels, height, width = values.shape
            values = values.reshape(n, 4, channels // 4, height, width).permute(0, 3, 4, 2, 1).reshape(n, height * width, channels // 4, 4)
        else:
            values = values.permute(0, 2, 3, 1).reshape(len(values), -1, values.shape[1])
        result.append(values.cpu().numpy())
    return np.concatenate(result)


def train_one(config, kind, tensors, labels, mean, std, folder, resume):
    device = mean.device
    folder.mkdir(parents=True, exist_ok=True)
    sources = source_hashes()
    if resume and (folder / "metrics.json").exists():
        report = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        if report["source_hashes"] != sources:
            raise ValueError("Sources changed since completed model training")
        print(f"Using completed {kind} training", flush=True)
        return report
    seed_everything(config["seed"])
    print(f"Training {kind} CNN on {device}; {len(tensors['train'])} images, {config['epochs']} epochs", flush=True)
    model = make_model(config, kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["minimum_learning_rate"])
    scaler = torch.amp.GradScaler(device.type, enabled=config["mixed_precision"] and device.type == "cuda")
    order_rng = torch.Generator().manual_seed(config["seed"] + 40000)
    augment_rng = torch.Generator(device=device).manual_seed(config["seed"] + 50000)
    initial = collect(model, tensors["calibration"], mean, std)
    initial_validation, _ = evaluate(model, tensors["validation"], labels["validation"], mean, std)
    best_loss, best_epoch, best_validation = initial_validation["cross_entropy"], 0, initial_validation
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history, audit, first_epoch, updates = [], None, 1, 0
    last_path = folder / "last.pt"
    if resume and last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved["config"] != config or saved["source_hashes"] != sources:
            raise ValueError("Resume config/source mismatch; use a new output directory")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        best_state, best_loss, best_epoch = saved["best_state"], saved["best_loss"], saved["best_epoch"]
        best_validation, history, audit, updates = saved["best_validation"], saved["history"], saved["audit"], saved["updates"]
        order_rng.set_state(saved["order_rng"])
        augment_rng.set_state(saved["augment_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        first_epoch = saved["epoch"] + 1
        print(f"Resuming {kind} at epoch {first_epoch}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(first_epoch, config["epochs"] + 1):
        start = time.perf_counter()
        model.train()
        loss_sum, correct = 0.0, 0
        for index in torch.randperm(len(tensors["train"]), generator=order_rng).split(config["batch_size"]):
            index = index.to(device)
            x = prepare_batch(tensors["train"][index], mean, std, True, augment_rng)
            y = labels["train"][index]
            if audit is None:
                audit = first_update_audit(model, x, y, optimizer)
                loss_sum += audit["loss_before"] * len(index)
                correct += audit["correct_before"]
            else:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, enabled=scaler.is_enabled()):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite image training loss")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach()) * len(index)
                correct += int((logits.detach().argmax(1) == y).sum())
            updates += 1
        validation, _ = evaluate(model, tensors["validation"], labels["validation"], mean, std)
        if validation["cross_entropy"] < best_loss:
            best_loss, best_epoch, best_validation = validation["cross_entropy"], epoch, validation
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        row = {"epoch": epoch, "train_augmented_loss": loss_sum / len(tensors["train"]), "train_augmented_accuracy": correct / len(tensors["train"]),
               "validation_loss": validation["cross_entropy"], "validation_accuracy": validation["accuracy"],
               "learning_rate": optimizer.param_groups[0]["lr"], "epoch_seconds": time.perf_counter() - start}
        history.append(row)
        scheduler.step()
        saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "config": config, "source_hashes": sources, "best_state": best_state, "best_loss": best_loss,
                 "best_epoch": best_epoch, "best_validation": best_validation, "history": history, "audit": audit, "updates": updates,
                 "order_rng": order_rng.get_state(), "augment_rng": augment_rng.get_state(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        torch.save(saved, folder / "last.tmp")
        (folder / "last.tmp").replace(last_path)
        with (folder / "training.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"{kind} epoch {epoch}/{config['epochs']}: train_aug={row['train_augmented_accuracy']:.3f}, val={validation['accuracy']:.3f}, loss={validation['cross_entropy']:.4f}, {row['epoch_seconds']:.1f}s", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    trained = collect(model, tensors["calibration"], mean, std)
    np.savez_compressed(folder / "calibration_activations.npz", initial=initial, trained=trained)
    checkpoint = {"model": {name: value.detach().cpu() for name, value in model.state_dict().items()}, "kind": kind, "config": config, "best_epoch": best_epoch}
    torch.save(checkpoint, folder / "model.pt")
    restored = make_model(config, kind).to(device).eval()
    restored.load_state_dict(torch.load(folder / "model.pt", map_location=device, weights_only=True)["model"])
    probe = prepare_batch(tensors["validation"][:32], mean, std)
    with torch.no_grad():
        reload_error = float((model(probe) - restored(probe)).abs().max())
    selected_train, _ = evaluate(model, tensors["train"], labels["train"], mean, std)
    report = {"kind": kind, "best_epoch": best_epoch, "selected_validation": best_validation, "selected_train": selected_train,
              "real_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "nontrainable_buffer_elements": sum(b.numel() for b in model.buffers()), "first_update": audit,
              "checkpoint_bytes": (folder / "model.pt").stat().st_size, "checkpoint_reload_max_error": reload_error,
              "optimizer_updates_attempted": updates, "activation_shape": list(trained.shape), "source_hashes": sources,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
              "test_evaluated": False, "masked_parameters": 0, "removed_units": 0, "compact_architecture_savings": 0}
    if reload_error > 1e-6:
        raise RuntimeError("CNN checkpoint reload mismatch")
    write_json(folder / "metrics.json", report)
    del model, restored, optimizer, scaler, checkpoint, saved, probe
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def run(config, output_dir, resume=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    if (output_dir / "summary.json").exists():
        raise FileExistsError("Finished results exist; use a new output directory")
    if config_path.exists() and (not resume or json.loads(config_path.read_text()) != config):
        raise ValueError("Output config exists; use --resume with identical settings or a new output directory")
    write_json(config_path, config)
    (images, targets), (test_images, test_targets) = load_cached_cifar(ROOT / config["archive"])
    with np.load(ROOT / config["split_source"]) as previous:
        validation, calibration = previous["validation"], previous["calibration"]
    train = np.setdiff1d(np.arange(len(images)), np.r_[validation, calibration])
    indices = {"train": train, "validation": validation, "calibration": calibration}
    assert len(train) == 48000 and len(np.unique(np.concatenate(list(indices.values())))) == 50000
    np.savez_compressed(output_dir / "split_indices.npz", **indices, test=np.arange(len(test_images)))
    statistics = channel_statistics(images[train])
    write_json(output_dir / "data.json", {"statistics": statistics, "counts": {key: len(value) for key, value in indices.items()},
               "test_samples": len(test_images), "source_split_sha256": file_sha256(ROOT / config["split_source"]),
               "archive_sha256": file_sha256(ROOT / config["archive"]), "disjoint_splits": True})
    device = torch.device(config["device"])
    tensors = {key: torch.as_tensor(images[index], device=device) for key, index in indices.items()}
    labels = {key: torch.as_tensor(targets[index], device=device) for key, index in indices.items()}
    mean = torch.tensor(statistics["mean"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(statistics["std"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    del images
    records = [train_one(config, kind, tensors, labels, mean, std, output_dir / kind, resume) for kind in config["models"]]
    test_x = torch.as_tensor(test_images, device=device)
    test_y = torch.as_tensor(test_targets, device=device)
    for report in records:
        folder = output_dir / report["kind"]
        model = make_model(config, report["kind"]).to(device).eval()
        model.load_state_dict(torch.load(folder / "model.pt", map_location=device, weights_only=True)["model"])
        report["test"], logits = evaluate(model, test_x, test_y, mean, std)
        report["test_evaluated"] = True
        report["target_80_percent_achieved"] = report["test"]["accuracy"] >= config["target_accuracy"]
        np.savez_compressed(folder / "test_predictions.npz", logits=logits, labels=test_targets)
        write_json(folder / "metrics.json", report)
        print(f"FINAL {report['kind']}: test accuracy {report['test']['accuracy']:.4f}", flush=True)
        del model
    environment = {"python": sys.version, "torch": torch.__version__, "cuda_build": torch.version.cuda,
                   "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                   "device": str(device), "seed": config["seed"], "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "versions": {name: metadata.version(name) for name in ("numpy", "matplotlib", "pytest")}}
    from src.evaluation.cnn_report import save_cnn_report
    save_cnn_report(records, config, output_dir)
    summary = {"config": config, "environment": environment, "models": records, "source_hashes": source_hashes(),
               "summary_status": "completed", "primary_target_achieved": next(r for r in records if r["kind"] == "quaternion")["target_80_percent_achieved"]}
    write_json(output_dir / "summary.json", summary)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=True)
    (output_dir / "environment.freeze.txt").write_text(freeze.stdout, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "cifar_cnn.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "cifar_cnn")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(config["cpu_threads"])
    if config["device"] == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use the verified .venv-cuda environment or explicitly choose CPU in a new config")
    if args.benchmark:
        benchmark(config, args.output_dir)
    else:
        run(config, args.output_dir, args.resume)
