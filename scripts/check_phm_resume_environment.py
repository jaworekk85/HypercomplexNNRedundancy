"""Audit a relocated PHM training environment without changing frozen studies.

Synthetic optimizer checks use a discarded model. Validation is replayed without
fitting; calibration and official test images are not used for model selection.
"""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.cnn_training import evaluate, first_update_audit, seed_everything
from src.data.cifar_cached import file_sha256, load_cached_cifar
from src.models.cifar_phm_resnet import CifarPHMResNet20

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/resnet20_qstem_phm4_baseline.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(output):
    config = read(CONFIG)
    study = ROOT / config["output"]
    protocol = read(study / "protocol.json")
    result = {
        "status": "checking", "started_utc": datetime.now(timezone.utc).isoformat(),
        "audit_script_sha256": file_sha256(Path(__file__)),
        "protocol_sha256": file_sha256(study / "protocol.json"),
        "config": config, "test_evaluated": False,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda,
        },
        "python_executable": sys.executable, "platform": platform.platform(),
    }
    output.mkdir(parents=True, exist_ok=True)
    try:
        if result["environment"] != protocol["environment"]:
            raise ValueError("Environment versions differ from the frozen protocol")
        if config != protocol["config"]:
            raise ValueError("Configuration differs from the frozen protocol")
        checked, problems = {}, []
        for name, expected in {**protocol["source_hashes"], **protocol["input_hashes"]}.items():
            path = ROOT / name
            actual = file_sha256(path) if path.is_file() else None
            checked[name] = actual
            if actual != expected:
                problems.append({"path": name, "expected": expected, "actual": actual})
        result["frozen_hashes"] = {"checked": len(checked), "problems": problems}
        if problems:
            raise ValueError(f"Frozen-source/input audit failed: {len(problems)} files")
        print(f"Frozen hashes passed: {len(checked)}", flush=True)
        result["pip_check"] = subprocess.run(
            [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=True
        ).stdout.strip()
        result["pip_freeze"] = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
        ).stdout.splitlines()
        result["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        result["gpu"] = torch.cuda.get_device_name(0)
        torch.set_num_threads(config["cpu_threads"])
        seed_everything(config["seed"])
        checkpoint = study / "qstem_phm4/last.pt"
        result["checkpoint_sha256"] = file_sha256(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        required = {"model", "optimizer", "scaler", "history", "counts", "best", "config", "rng"}
        if not required <= saved.keys() or saved["config"] != config:
            raise ValueError("Incomplete checkpoint or mismatched resume recipe")
        if not saved["history"] or not saved["optimizer"]["state"] or not saved["scaler"]:
            raise ValueError("Checkpoint lacks completed history, optimizer, or AMP state")
        epochs = len(saved["history"])
        if [row["epoch"] for row in saved["history"]] != list(range(1, epochs + 1)):
            raise ValueError("Checkpoint epoch history is discontinuous")
        torch.set_rng_state(saved["rng"]["torch"])
        torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        torch.Generator().set_state(saved["rng"]["order"])
        torch.Generator(device="cuda").set_state(saved["rng"]["augmentation"])
        model = CifarPHMResNet20(config["quaternion_widths"], config["classes"], config["phm_dimensions"]).cuda()
        count = sum(p.numel() for p in model.parameters())
        if count != 272746:
            raise ValueError(f"Unexpected real parameter count: {count}")
        (images, labels), unused_test = load_cached_cifar(ROOT / config["archive"])
        del unused_test
        with np.load(ROOT / config["split_source"]) as split:
            ids = {key: split[key].copy() for key in ("train", "validation", "calibration")}
        groups = [set(v.tolist()) for v in ids.values()]
        if sorted(set.union(*groups)) != list(range(50000)) or sum(map(len, groups)) != 50000:
            raise ValueError("Split indices overlap, are duplicated, or omit training images")
        stats = read(ROOT / config["data_statistics"])["statistics"]
        mean = torch.tensor(stats["mean"], device="cuda").reshape(1, 3, 1, 1)
        std = torch.tensor(stats["std"], device="cuda").reshape(1, 3, 1, 1)
        x = torch.as_tensor(images[ids["validation"]], device="cuda")
        y = torch.as_tensor(labels[ids["validation"]], device="cuda")
        result["validation_replays"] = {}
        for name, state, expected in (
            ("last_completed_epoch", saved["model"], saved["history"][-1]["validation"]),
            ("best_ce_checkpoint", saved["best"]["model"], saved["best"]["validation"]),
        ):
            model.load_state_dict(state)
            actual = evaluate(model, x, y, mean, std, config["batch_size"])[0]
            delta = abs(actual["cross_entropy"] - expected["cross_entropy"])
            result["validation_replays"][name] = {"expected": expected, "actual": actual, "ce_abs_difference": delta}
            if actual["accuracy"] != expected["accuracy"] or delta > 1e-6:
                raise ValueError(f"Checkpoint validation replay failed: {name}")
        del model, images, labels, x, y
        seed_everything(config["seed"])
        model = CifarPHMResNet20(config["quaternion_widths"], config["classes"], config["phm_dimensions"]).cuda()
        x = torch.randn(config["batch_size"], 3, 32, 32, device="cuda")
        y = torch.arange(len(x), device="cuda") % config["classes"]
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=config["momentum"], weight_decay=config["weight_decay"])
        result["discarded_copy_update_audit"] = first_update_audit(model, x[:8], y[:8], optimizer)
        scaler = torch.amp.GradScaler("cuda", enabled=config["mixed_precision"])
        updates = []
        optimizer.register_step_post_hook(lambda *args: updates.append(True))
        torch.cuda.reset_peak_memory_stats()
        for step in range(15):
            if step == 5:
                torch.cuda.synchronize()
                start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = torch.nn.functional.cross_entropy(model(x), y)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite synthetic AMP loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        if not updates:
            raise ValueError("All synthetic AMP optimizer updates were skipped")
        result["synthetic_gpu_preflight"] = {
            "batch_size": len(x), "attempted_updates": 15, "successful_updates": len(updates),
            "seconds_per_batch_after_warmup": (time.perf_counter() - start) / 10,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "real_trainable_parameters": count, "scientific_training": False,
        }
        backup = output / f"checkpoint_epoch{epochs}.pt"
        if not backup.exists():
            shutil.copy2(checkpoint, backup)
        if file_sha256(backup) != result["checkpoint_sha256"]:
            raise ValueError("Pre-resume checkpoint backup differs")
        result["completed_epoch"] = epochs
        result["best_epoch"] = saved["best"]["epoch"]
        result["optimizer_counts"] = saved["counts"]
        result["checkpoint_backup"] = str(backup.relative_to(ROOT))
        result["status"] = "passed"
        print(f"PASS: epoch {epochs}, {count} real trainables; CUDA updates and validation replays passed", flush=True)
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "preflight.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.output.resolve())
