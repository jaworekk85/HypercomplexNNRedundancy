"""Sequential real/Q ResNet20 baselines. --prepare checks; --run starts/resumes."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from scripts.paper_a_controls import now, rel
from scripts.run_activation_queue import lock
from src.cnn_training import evaluate, first_update_audit, seed_everything
from src.data.cifar_cached import load_cached_cifar, file_sha256
from src.models.cifar_resnet import CifarResNet20
from src.resnet_training import train
from src.trainable_map_repair import atomic_save

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/resnet20_baselines.json"
NEW = [CONFIG, "src/models/cifar_resnet.py", "src/resnet_training.py", "scripts/resnet20_baselines.py", "tests_torch/test_cifar_resnet.py"]


def make_model(study, kind):
    return CifarResNet20(kind, study["real_widths"], study["quaternion_widths"], study["classes"])


def register(study):
    parent = curve.read(ROOT/study["parent_protocol"])
    for name, digest in {**parent["source_hashes"], **parent["input_hashes"]}.items():
        if file_sha256(ROOT/name) != digest: raise ValueError(f"Frozen parent changed: {name}")
    inputs = [ROOT/study[p] for p in ("parent_protocol", "split_source", "data_statistics", "archive")]
    definition = {"config": study, "source_hashes": {**parent["source_hashes"], **{p: file_sha256(ROOT/p) for p in NEW}},
                  "input_hashes": {**parent["input_hashes"], **{rel(p): file_sha256(p) for p in inputs}},
                  "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda},
                  "prior_art": study["prior_art"], "test_evaluated": False}
    output = ROOT/study["output"]
    output.mkdir(parents=True, exist_ok=True)
    path = output/"protocol.json"
    if path.exists() and curve.read(path) != definition: raise ValueError("ResNet baseline protocol changed")
    write_json(path, definition)
    return definition


def prepare(study):
    device = torch.device(study["device"])
    checks = {}
    for kind in study["models"]:
        seed_everything(study["seed"])
        model = make_model(study, kind).to(device)
        x = torch.randn(study["batch_size"], 3, 32, 32, device=device)
        y = torch.arange(len(x), device=device) % study["classes"]
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=study["momentum"], weight_decay=study["weight_decay"])
        audit = first_update_audit(model, x[:8], y[:8], optimizer)
        scaler = torch.amp.GradScaler(device.type, enabled=study["mixed_precision"] and device.type == "cuda")
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
        start = None
        for step in range(15):
            if step == 5:
                if device.type == "cuda": torch.cuda.synchronize()
                start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=scaler.is_enabled()):
                loss = torch.nn.functional.cross_entropy(model(x), y)
            if not bool(torch.isfinite(loss)): raise AssertionError("Nonfinite preflight loss")
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        if device.type == "cuda": torch.cuda.synchronize()
        seconds = (time.perf_counter()-start)/10
        model.eval()
        replica = make_model(study, kind).to(device).eval()
        replica.load_state_dict(model.state_dict())
        with torch.no_grad(): reload_error = float((model(x[:8])-replica(x[:8])).abs().max())
        if reload_error > 1e-6: raise AssertionError("ResNet reload failed")
        checks[kind] = {"real_trainable_parameters": sum(p.numel() for p in model.parameters()),
                        "nontrainable_buffer_elements": sum(b.numel() for b in model.buffers()),
                        "first_update_audit": audit, "reload_max_abs_error": reload_error,
                        "seconds_per_training_batch": seconds, "rough_training_seconds_per_epoch": seconds*375,
                        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
        print(f"READY {kind}: {checks[kind]['real_trainable_parameters']} real trainables; {seconds:.4f}s/batch", flush=True)
        del model, replica, optimizer, scaler, x, y
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    definition = register(study)
    output = ROOT/study["output"]
    write_json(output/"readiness.json", {"status": "passed", "source_hashes": definition["source_hashes"],
              "tests": "4 CPU checks passed: both architectures/gradients/physical widths, Q shortcut grouping, exact SGD interruption replay",
              "checks": checks, "scientific_results": False, "updated_utc": now()})
    if not (output/"state.json").exists(): write_json(output/"state.json", {"status": "ready_not_started", "updated_utc": now()})


def report(study, results, status, progress=None):
    output = ROOT/study["output"]
    write_json(output/"summary.json", {"status": status, "config": study, "results": results, "progress": progress, "test_evaluated": False})
    lines = ["# CIFAR ResNet20 baseline quality gate", "", f"Status: {status}. One seed2026 per family; sequential200ep SGD.", "",
             "## Hypothesis", "", "A stronger residual backbone may reach approximately90percent validation accuracy and support meaningful Q/real compression validation. This is a target, not a measured guarantee or novelty claim.", "",
             "## Measured results", "", "| Family | Real parameters | Selected epoch | Selected validation (%) | Final validation (%) |", "|---|---:|---:|---:|---:|"]
    for row in results: lines.append(f"| {row['kind']} | {row['real_trainable_parameters']} | {row['selected_epoch']} | {100*row['selected_validation']['accuracy']:.2f} | {100*row['final_validation']['accuracy']:.2f} |")
    if progress: lines += ["", f"Current family {progress['kind']}, completed epoch {progress['epoch']}/{study['epochs']}, current validation {100*progress['validation']['accuracy']:.2f}%; current epoch seconds {progress['epoch_seconds']:.1f}."]
    lines += ["", "## Interpretation and limits", "", study["recipe"], "", study["checkpoint_selection"], "", study["scope"], "", study["resume"], "",
              "Actual counts distinguish real trainables, Q units and BN buffers. Approximate parameter matching is not equal width/compute or an isolated quaternion-algebra experiment. No pruning or compressed-model savings have been measured in this baseline stage. Same train/validation/calibration split reused; no calibration/test inference, validation selection disclosed. He2016ResNet original real ResNet20 test91.25percent is background, not our outcome. Literature: references.bib. GPU preflight uses synthetic discarded models and is not scientific evidence."]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(study):
    definition = register(study)
    output = ROOT/study["output"]
    ready = curve.read(output/"readiness.json")
    if ready["status"] != "passed" or ready["source_hashes"] != definition["source_hashes"]: raise ValueError("Baseline readiness changed")
    with lock("compression_curve_runner.lock"):
        results = []
        try:
            if (output/"STOP").exists():
                write_json(output/"state.json", {"status": "paused", "updated_utc": now()})
                return
            device = torch.device(study["device"])
            (images, labels), unused_test = load_cached_cifar(ROOT/study["archive"])
            del unused_test
            with np.load(ROOT/study["split_source"]) as split: ids = {k: split[k].copy() for k in ("train", "validation", "calibration")}
            sets = [set(v) for v in ids.values()]
            if any(a & b for i, a in enumerate(sets) for b in sets[i+1:]) or sum(map(len, ids.values())) != 50000: raise AssertionError("Invalid data split")
            stats = curve.read(ROOT/study["data_statistics"])["statistics"]
            mean = torch.tensor(stats["mean"], device=device).reshape(1, 3, 1, 1)
            std = torch.tensor(stats["std"], device=device).reshape(1, 3, 1, 1)
            tensors = {s: torch.as_tensor(images[ids[s]], device=device) for s in ("train", "validation")}
            targets = {s: torch.as_tensor(labels[ids[s]], device=device) for s in tensors}
            for kind in study["models"]:
                folder = output/kind
                folder.mkdir(parents=True, exist_ok=True)
                if (folder/"metrics.json").exists():
                    row = curve.read(folder/"metrics.json")
                    if file_sha256(folder/"model.pt") != row["checkpoint_sha256"]: raise ValueError("Completed ResNet changed")
                    results.append(row)
                    continue
                seed_everything(study["seed"])
                model = make_model(study, kind).to(device)
                def epoch_done(row, best):
                    progress = {"kind": kind, **row, "best_epoch": best["epoch"], "best_validation": best["validation"]}
                    write_json(output/"state.json", {"status": "running", "pid": os.getpid(), "completed_models": len(results), **progress, "updated_utc": now()})
                    with (folder/"training.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps(row)+"\n")
                    report(study, results, "running", progress)
                    print(f"{kind} epoch{row['epoch']}/{study['epochs']} val={100*row['validation']['accuracy']:.2f}% CE={row['validation']['cross_entropy']:.4f} time={row['epoch_seconds']:.1f}s", flush=True)
                write_json(output/"state.json", {"status": "running", "kind": kind, "pid": os.getpid(), "completed_models": len(results), "updated_utc": now()})
                print(f"START {kind}: {sum(p.numel() for p in model.parameters())} real trainables,200ep", flush=True)
                model, details, done = train(model, tensors, targets, mean, std, study, folder/"last.pt", output/"STOP", epoch_done)
                if not done:
                    write_json(output/"state.json", {"status": "paused", "kind": kind, "completed_epoch": len(details["history"]), "updated_utc": now()})
                    report(study, results, "paused")
                    return
                atomic_save({"model": {n: v.detach().cpu() for n, v in model.state_dict().items()}, "kind": kind, "config": study,
                             "selected_epoch": details["selected_epoch"]}, folder/"model.pt")
                replica = make_model(study, kind).to(device).eval()
                replica.load_state_dict(torch.load(folder/"model.pt", map_location=device, weights_only=True)["model"])
                replay = evaluate(replica, tensors["validation"], targets["validation"], mean, std, study["batch_size"])[0]
                if abs(replay["cross_entropy"]-details["selected_validation"]["cross_entropy"]) > 1e-6 or replay["accuracy"] != details["selected_validation"]["accuracy"]:
                    raise AssertionError("Selected checkpoint validation replay failed")
                row = {"kind": kind, **details, "real_trainable_parameters": sum(p.numel() for p in model.parameters()),
                       "nontrainable_buffer_elements": sum(b.numel() for b in model.buffers()), "checkpoint_sha256": file_sha256(folder/"model.pt"),
                       "checkpoint_bytes": (folder/"model.pt").stat().st_size, "validation_reload_replay": replay,
                       "target_met": replay["accuracy"] >= study["target_validation_accuracy"], "test_evaluated": False}
                write_json(folder/"metrics.json", row)
                results.append(row)
                report(study, results, "running")
                del model, replica
                gc.collect()
                if device.type == "cuda": torch.cuda.empty_cache()
            register(study)
            report(study, results, "completed")
            write_json(output/"state.json", {"status": "completed", "completed_models": len(results), "quality_target_met_by_both": all(r["target_met"] for r in results), "completed_utc": now()})
        except BaseException as error:
            write_json(output/"state.json", {"status": "failed", "error": f"{type(error).__name__}: {error}", "updated_utc": now()})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    study = curve.read(ROOT/CONFIG)
    torch.set_num_threads(study["cpu_threads"])
    if args.run: run(study)
    else: prepare(study)
