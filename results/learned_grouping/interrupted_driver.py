"""One-seed CIFAR learned-assignment pilot with hardening and resumable states."""

import argparse
from copy import deepcopy
import csv
import gc
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts.cifar_cnn import collect, make_model as make_original, source_hashes, write_json
from src.cnn_training import evaluate, first_update_audit, prepare_batch, seed_everything
from src.data.cifar_cached import file_sha256, load_cached_cifar
from src.models.channel_routing import LearnableChannelPermutation, RoutedCifarCNN

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["scripts/learned_grouping.py", "src/models/channel_routing.py"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_model(config, mode):
    return RoutedCifarCNN(mode, config["seed"], config["quaternion_widths"], config["dropout"],
                          config["initial_strength"], config["temperature_start"], config["sinkhorn_iterations"])


def load_export(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(checkpoint["config"], "identity").to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def save_last(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train_mode(config, mode, tensors, labels, mean, std, folder, hashes):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "metrics.json").exists():
        result = read(folder / "metrics.json")
        if result["source_hashes"] != hashes:
            raise ValueError("Completed pilot model sources changed")
        return result
    seed_everything(config["seed"])
    device = mean.device
    model = make_model(config, mode).to(device)
    backbone = [p for name,p in model.named_parameters() if name != "router.logits"]
    groups = [{"params": backbone, "lr": config["learning_rate"], "weight_decay": config["weight_decay"]}]
    if mode == "learned":
        groups.append({"params": [model.router.logits], "lr": config["router_learning_rate"], "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["minimum_learning_rate"])
    scaler = torch.amp.GradScaler(device.type, enabled=config["mixed_precision"] and device.type == "cuda")
    order_rng = torch.Generator().manual_seed(config["seed"]+40000)
    augment_rng = torch.Generator(device=device).manual_seed(config["seed"]+50000)
    initial_parameters = sum(p.numel() for p in model.parameters())
    initial = collect(model, tensors["calibration"], mean, std)
    history, audit, hardening, best_state, best_validation, best_epoch = [], None, None, None, None, None
    updates, first_epoch = 0, 1
    last_path = folder / "last.pt"
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved["config"] != config or saved["mode"] != mode or saved["source_hashes"] != hashes:
            raise ValueError("Pilot resume source/config mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        history, audit, hardening = saved["history"], saved["audit"], saved["hardening"]
        best_state, best_validation, best_epoch = saved["best_state"], saved["best_validation"], saved["best_epoch"]
        updates, first_epoch = saved["updates"], saved["epoch"]+1
        order_rng.set_state(saved["order_rng"])
        augment_rng.set_state(saved["augment_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        print(f"Resuming {mode} at epoch {first_epoch}", flush=True)
    else:
        print(f"Training {mode}: {initial_parameters} training real parameters", flush=True)
    for epoch in range(first_epoch, config["epochs"]+1):
        start = time.perf_counter()
        if mode == "learned" and epoch <= config["soft_epochs"]:
            fraction = (epoch-1)/max(1, config["soft_epochs"]-1)
            model.router.temperature.fill_(config["temperature_start"]*(config["temperature_end"]/config["temperature_start"])**fraction)
        model.train()
        loss_sum, correct = 0.0, 0
        for index in torch.randperm(len(tensors["train"]), generator=order_rng).split(config["batch_size"]):
            index = index.to(device)
            x = prepare_batch(tensors["train"][index], mean, std, True, augment_rng)
            y = labels["train"][index]
            if audit is None:
                audit = first_update_audit(model, x, y, optimizer)
                loss_sum += audit["loss_before"]*len(index)
                correct += audit["correct_before"]
            else:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, enabled=scaler.is_enabled()):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite routing training loss")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach())*len(index)
                correct += int((logits.detach().argmax(1) == y).sum())
            updates += 1
        validation, logits = evaluate(model, tensors["validation"], labels["validation"], mean, std)
        if mode == "learned" and epoch == config["soft_epochs"]:
            before = validation
            before_logits = logits
            diagnostics = model.router.diagnostics()
            model.router.harden()
            validation, logits = evaluate(model, tensors["validation"], labels["validation"], mean, std)
            hardening = {"epoch": epoch, "soft_validation": before, "immediate_hard_validation": validation,
                         "accuracy_drop": before["accuracy"]-validation["accuracy"],
                         "max_abs_logit_shift": float(np.max(np.abs(before_logits-logits))), "routing": diagnostics}
            write_json(folder / "hardening.json", hardening)
            print(f"HARDENING: soft val={before['accuracy']:.3f}, immediate hard val={validation['accuracy']:.3f}", flush=True)
        eligible = epoch >= config["soft_epochs"]
        if eligible and (best_validation is None or validation["cross_entropy"] < best_validation["cross_entropy"]):
            best_state = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
            best_validation, best_epoch = validation, epoch
        phase = "hard" if mode != "learned" or bool(model.router.hard_enabled) else "soft"
        history.append({"epoch": epoch, "train_augmented_loss": loss_sum/len(tensors["train"]),
                        "train_augmented_accuracy": correct/len(tensors["train"]), "validation_loss": validation["cross_entropy"],
                        "validation_accuracy": validation["accuracy"], "phase_after_epoch": phase,
                        "learning_rate": optimizer.param_groups[0]["lr"], "epoch_seconds": time.perf_counter()-start})
        scheduler.step()
        state = {"config": config, "mode": mode, "source_hashes": hashes, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "history": history, "audit": audit, "hardening": hardening, "best_state": best_state,
                 "best_validation": best_validation, "best_epoch": best_epoch, "updates": updates,
                 "order_rng": order_rng.get_state(), "augment_rng": augment_rng.get_state(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        save_last(last_path, state)
        with (folder / "training.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(f"{mode} epoch {epoch}/{config['epochs']} ({phase}): val={validation['accuracy']:.3f}, loss={validation['cross_entropy']:.4f}, {history[-1]['epoch_seconds']:.1f}s", flush=True)
    if best_state is None:
        raise RuntimeError("No eligible hard checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    routing = model.router.diagnostics() if mode == "learned" else {"matching": model.router.indices.cpu().tolist()}
    exported = model.export_hard()
    probe = prepare_batch(tensors["validation"][:32], mean, std)
    with torch.no_grad():
        error = float((model(probe)-exported(probe)).abs().max())
    if error != 0:
        raise AssertionError("Hard-export equivalence failed")
    torch.save({"model": {name:value.cpu() for name,value in exported.state_dict().items()}, "config": config,
                "mode": mode, "best_epoch": best_epoch}, folder / "model.pt")
    restored = load_export(folder / "model.pt", device)
    with torch.no_grad():
        reload_error = float((restored(probe)-exported(probe)).abs().max())
    if reload_error != 0:
        raise AssertionError("Export reload equivalence failed")
    selected_train, _ = evaluate(exported, tensors["train"], labels["train"], mean, std)
    trained = collect(exported, tensors["calibration"], mean, std)
    np.savez_compressed(folder / "calibration_activations.npz", initial=initial, trained=trained)
    result = {"mode": mode, "seed": config["seed"], "best_epoch": best_epoch, "selected_validation": best_validation,
              "selected_train": selected_train, "training_real_parameters": initial_parameters,
              "exported_real_parameters": sum(p.numel() for p in exported.parameters()),
              "exported_buffer_elements": sum(b.numel() for b in exported.buffers()),
              "checkpoint_bytes": (folder / "model.pt").stat().st_size, "optimizer_updates_attempted": updates,
              "routing": routing, "hardening": hardening, "first_update": audit,
              "hard_export_max_abs_error": error, "reload_max_abs_error": reload_error,
              "source_hashes": hashes, "test_evaluated": False, "compact_channel_savings": 0}
    write_json(folder / "metrics.json", result)
    del model, exported, restored, optimizer, state
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(config, folder):
    if (folder / "summary.json").exists():
        print("Learned grouping pilot already completed", flush=True)
        return
    original_folder = ROOT / config["baseline_source"]
    original = read(original_folder / "summary.json")
    if original["source_hashes"] != source_hashes() or read(ROOT / "results" / "learned_grouping" / "smoke.json")["status"] != "passed":
        raise ValueError("Frozen backbone source or prototype gate failed")
    if not 1 <= config["soft_epochs"] < config["epochs"]:
        raise ValueError("Require both soft and hard training phases")
    old_config = original["config"]
    for key in ("seed", "quaternion_widths", "dropout", "epochs", "batch_size", "learning_rate", "minimum_learning_rate", "weight_decay"):
        if config[key] != old_config[key]:
            raise ValueError(f"Identity reference protocol mismatch: {key}")
    hashes = {**source_hashes(), **{name:file_sha256(ROOT / name) for name in SOURCES}}
    protocol = {"config": config, "source_hashes": hashes, "original_summary_sha256": file_sha256(original_folder / "summary.json")}
    if (folder / "protocol.json").exists() and read(folder / "protocol.json") != protocol:
        raise ValueError("Pilot source/config changed; use another output directory")
    write_json(folder / "protocol.json", protocol)
    torch.set_num_threads(config["cpu_threads"])
    seed_everything(config["seed"])
    device = torch.device(config["device"])
    (images, labels), (test_images, test_labels) = load_cached_cifar(ROOT / config["archive"])
    with np.load(original_folder / "split_indices.npz") as split:
        indices = {key:split[key] for key in ("train", "validation", "calibration")}
    statistics = read(original_folder / "data.json")["statistics"]
    for name in ("split_indices.npz", "data.json", "environment.freeze.txt"):
        if not (folder / name).exists():
            shutil.copyfile(original_folder / name, folder / name)
    tensors = {key:torch.as_tensor(images[index], device=device) for key,index in indices.items()}
    targets = {key:torch.as_tensor(labels[index], device=device) for key,index in indices.items()}
    mean = torch.tensor(statistics["mean"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(statistics["std"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    baseline = deepcopy(next(r for r in original["models"] if r["kind"] == "quaternion"))
    base = make_original(old_config, "quaternion").to(device).eval()
    base.load_state_dict(torch.load(original_folder / "quaternion" / "model.pt", map_location=device, weights_only=True)["model"])
    identity = make_model(config, "identity").to(device).eval()
    identity.load_state_dict(base.state_dict(), strict=False)
    probe = prepare_batch(tensors["validation"][:32], mean, std)
    with torch.no_grad():
        reference_error = float((base(probe)-identity(probe)).abs().max())
    if reference_error != 0 or baseline["best_epoch"] < config["soft_epochs"]:
        raise AssertionError("Reused identity reference is not equivalent/eligible")
    write_json(folder / "identity_reference.json", {"mode": "identity", "seed": config["seed"], "reused": True,
               "reference_forward_max_abs_error": reference_error, "source_checkpoint_sha256": file_sha256(original_folder / "quaternion" / "model.pt"),
               "best_epoch": baseline["best_epoch"], "selected_validation": baseline["selected_validation"], "test": baseline["test"]})
    del base, identity, probe, images
    records = [train_mode(config, mode, tensors, targets, mean, std, folder / mode, hashes) for mode in config["new_modes"]]
    random, learned = (next(r for r in records if r["mode"] == mode) for mode in ("random", "learned"))
    control_accuracy = max(baseline["selected_validation"]["accuracy"], random["selected_validation"]["accuracy"])
    gate = {"useful_hardened_quality": learned["selected_validation"]["accuracy"] >= control_accuracy-config["maximum_validation_accuracy_deficit_for_compression_gate"]-1e-12,
            "assignment_actually_changed": learned["routing"]["changed_output_positions"] > 0}
    gate["supports_compression_followup"] = all(gate.values())
    write_json(folder / "selection_frozen.json", {"records": records, "compression_gate": gate, "all_choices_frozen_before_new_test": True})
    del tensors, targets
    gc.collect()
    torch.cuda.empty_cache()
    test_x, test_y = torch.as_tensor(test_images, device=device), torch.as_tensor(test_labels, device=device)
    for record in records:
        model_folder = folder / record["mode"]
        if not record["test_evaluated"]:
            model = load_export(model_folder / "model.pt", device)
            record["test"], logits = evaluate(model, test_x, test_y, mean, std)
            record["test_evaluated"] = True
            np.savez_compressed(model_folder / "test_predictions.npz", logits=logits, labels=test_labels)
            write_json(model_folder / "metrics.json", record)
            print(f"FINAL {record['mode']}: test={record['test']['accuracy']:.4f}", flush=True)
            del model
    summary = {"status": "completed", "config": config, "identity_reference": baseline, "models": records,
               "compression_gate": gate, "protocol": protocol, "selection_freeze_sha256": file_sha256(folder / "selection_frozen.json"),
               "environment": {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
                               "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}}
    write_json(folder / "summary.json", summary)
    render_report(summary, folder)
    print(f"Learned assignment pilot completed; compression gate: {gate['supports_compression_followup']}", flush=True)


def render_report(summary, folder):
    base = summary["identity_reference"]
    lines = ["# Learned group-assignment pilot", "", "## Hypotheses", "",
             "Learning scalar-channel membership may help a fixed constrained CNN. One seed cannot establish a stable gain, novelty or compression value. Canonical context: Mena2018, Wang2019FLGC, Vecchi2020, Singh2026.", "", "## Fixed protocol", "",
             "One router between blocks 1 and 2; same 48k/1k/1k split and backbone initialization, dedicated sample-order/augmentation streams. Reuse the equivalent original identity seed-2026 reference. New random and learned runs use 50 epochs, 40 soft epochs then 10 hard epochs in the learned condition. Checkpoint choice uses minimum validation CE within epochs 40-50, never soft-model quality or test. All choices and the compression gate were frozen before the two new tests. The test set was used in earlier pilots.",
             "Learned training adds 1024 logits and dense mixing computation. Export retains only permutation indices and the original backbone parameters. Training budgets are not parameter/compute matched; final routing does not itself remove channels. No Gumbel noise, new multiplication rules or latency gain is claimed.", "", "## Measured results", "",
             "| Mode | Training real parameters | Exported real parameters | Selected epoch | Validation accuracy | Test accuracy |", "| --- | ---: | ---: | ---: | ---: | ---: |",
             f"| identity (reused) | 73898 | 73898 | {base['best_epoch']} | {100*base['selected_validation']['accuracy']:.2f}% | {100*base['test']['accuracy']:.2f}% |"]
    for record in summary["models"]:
        lines.append(f"| {record['mode']} | {record['training_real_parameters']} | {record['exported_real_parameters']} | {record['best_epoch']} | {100*record['selected_validation']['accuracy']:.2f}% | {100*record['test']['accuracy']:.2f}% |")
    learned = next(r for r in summary["models"] if r["mode"] == "learned")
    hardening = learned["hardening"]
    lines.extend(["", f"At hardening: soft validation {100*hardening['soft_validation']['accuracy']:.2f}%, immediate hard validation {100*hardening['immediate_hard_validation']['accuracy']:.2f}% (drop {100*hardening['accuracy_drop']:.2f} pp). Selected exported assignment changes {learned['routing']['changed_output_positions']}/32 output positions from the initial random assignment.",
                  f"Validation-only compression gate: {summary['compression_gate']}.", "", "## Interpretation and limitations", "",
                  "Retain any soft-to-hard collapse and failed gate. A soft model can exploit extra linear mixing, so soft quality does not prove a useful fixed grouping. A fixed hard model passing the gate would only justify a subsequent matched physical-compression comparison, not establish compression benefit. One seed and one boundary are exploratory; no first-in-literature or publishable novelty claim follows. A learned permutation is not inherently invariant to the chosen coordinate basis or uniquely quaternion-specific."])
    (folder / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "learned_grouping.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "learned_grouping")
    args = parser.parse_args()
    run(read(args.config), args.output_dir)
