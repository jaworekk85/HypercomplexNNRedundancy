"""Synthetic CPU end-to-end check; never a scientific result."""
from pathlib import Path
from hashlib import sha256
from unittest.mock import patch

import numpy as np
import torch

from scripts import resnet_structure_catalogue as stage
from scripts.cifar_cnn import write_json
from src.models.cifar_resnet import CifarResNet20


def run():
    torch.set_num_threads(2)
    torch.manual_seed(6)
    root = stage.ROOT
    hashes = {p: stage.file_sha256(root/p) for p in stage.NEW}
    suffix = sha256(str(hashes).encode()).hexdigest()[:12]
    output = root/".cache/structure_catalogue_end_to_end"/suffix
    output.mkdir(parents=True, exist_ok=True)
    cfg = {"real_widths": [4, 8, 16], "quaternion_widths": [2, 4, 8], "classes": 10,
           "split_source": str((output/"splits.npz").relative_to(root)).replace("\\", "/")}
    write_json(output/"baseline.json", cfg)
    np.savez(output/"splits.npz", train=np.arange(16), validation=np.arange(16, 24), calibration=np.arange(24, 32))
    torch.save({"model": CifarResNet20("quaternion", **{k: v for k, v in cfg.items() if k != "split_source"}).state_dict()}, output/"model.pt")
    write_json(output/"readiness.json", {"status": "passed", "source_hashes": {}})
    base = stage.curve.read(root/stage.CONFIG)
    study = {**base, "output": str(output.relative_to(root)).replace("\\", "/"), "device": "cpu",
             "baseline_config": str((output/"baseline.json").relative_to(root)).replace("\\", "/"),
             "baseline_checkpoint": str((output/"model.pt").relative_to(root)).replace("\\", "/"),
             "finetune": {**base["finetune"], "epochs": 1, "batch_size": 8, "mixed_precision": False}}
    images = np.random.default_rng(6).integers(0, 256, (32, 3, 8, 8), dtype=np.uint8)
    labels = np.arange(32, dtype=np.int64) % 10
    data = (images, labels, torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1), [])
    original_lock = stage.lock
    with patch.object(stage, "register", return_value={"source_hashes": {}}), patch.object(stage, "check_dependencies"), \
         patch.object(stage.curve, "load_data", return_value=data), \
         patch.object(stage, "lock", side_effect=lambda unused: original_lock("structure_catalogue_cpu_smoke.lock")):
        stage.run(study)
    summary = stage.curve.read(output/"summary.json")
    if summary["status"] != "completed" or len(summary["results"]) != 5: raise AssertionError("Synthetic pipeline incomplete")
    technical = {"status": "passed", "scientific_results": False, "source_hashes": hashes,
              "policies_completed": len(summary["results"]), "actual_budget": summary["global_budget"],
              "all_counts_within_budget": all(r["real_parameters"] <= summary["global_budget"] for r in summary["results"]),
              "synthetic_images": 32, "test_evaluated": False, "dedicated_CPU_test_lock": True,
              "earlier_check": "Initial smoke correctly refused the active shared GPU-study lock; CPU-only harness now uses a separate test lock. Scientific run retains the original exclusive lock."}
    write_json(output/"technical_result.json", technical)
    write_json(root/".cache/structure_catalogue_end_to_end/last_readiness.json", technical)
    print("SMOKE PASSED:5 policies, synthetic CPU data ONLY", flush=True)


if __name__ == "__main__": run()
