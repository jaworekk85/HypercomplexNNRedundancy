"""End-to-end CPU check on synthetic data; outputs are NOT scientific results."""
from hashlib import sha256
from unittest.mock import patch

import numpy as np
import torch

from scripts import resnet20_compression as stage
from scripts.cifar_cnn import write_json
from src.models.cifar_resnet import CifarResNet20


def run():
    torch.set_num_threads(2)
    torch.manual_seed(32)
    root = stage.ROOT
    sources = {p: stage.file_sha256(root/p) for p in stage.NEW}
    folder = root/".cache/resnet20_compression_smoke"/sha256(str(sources).encode()).hexdigest()[:12]
    folder.mkdir(parents=True, exist_ok=True)
    baseline = {"real_widths": [4, 8, 16], "quaternion_widths": [2, 4, 8], "classes": 10,
                "split_source": str((folder/"split.npz").relative_to(root)).replace("\\", "/")}
    write_json(folder/"baseline.json", baseline)
    np.savez(folder/"split.npz", train=np.arange(32), validation=np.arange(32, 40), calibration=np.arange(40, 48))
    fake = {kind: {"model": CifarResNet20(kind, baseline["real_widths"], baseline["quaternion_widths"]).state_dict()}
            for kind in ("real", "quaternion")}
    images = np.random.default_rng(5).integers(0, 256, (48, 3, 8, 8), dtype=np.uint8)
    labels = np.arange(48, dtype=np.int64) % 10
    data = images, labels, torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1), []
    original = stage.curve.read(root/stage.CONFIG)
    study = {**original, "output": str(folder.relative_to(root)).replace("\\", "/"), "device": "cpu", "fractions": [.5],
             "baseline_config": str((folder/"baseline.json").relative_to(root)).replace("\\", "/"),
             "calibration_batch": 4, "finetune": {**original["finetune"], "epochs": 1, "batch_size": 8, "mixed_precision": False}}
    write_json(folder/"readiness.json", {"status": "passed", "source_hashes": {}})
    original_load, original_lock = torch.load, stage.lock
    def fake_load(path, *args, **kwargs):
        p = str(path).replace("\\", "/")
        for kind in fake:
            if p.endswith("results/resnet20_baselines/"+kind+"/model.pt"): return fake[kind]
        return original_load(path, *args, **kwargs)
    with patch.object(stage, "register", return_value={"source_hashes": {}}), patch.object(stage.curve, "load_data", return_value=data), \
         patch.object(stage.torch, "load", side_effect=fake_load), patch.object(stage, "lock", side_effect=lambda unused: original_lock("resnet20_compression_cpu_smoke.lock")):
        stage.run(study)
    result = stage.curve.read(folder/"summary.json")
    if result["status"] != "completed" or len(result["results"]) != 6: raise AssertionError("Synthetic pipeline incomplete")
    record = {"status": "passed", "scientific_results": False, "source_hashes": sources, "cases": 6,
              "split_sizes": {"train": 32, "validation": 8, "fit": 4, "check": 4}, "test_evaluated": False}
    write_json(root/".cache/resnet20_compression_smoke/last_readiness.json", record)
    print("SMOKE PASSED:6 cases,9 sites, synthetic CPU data ONLY", flush=True)


if __name__ == "__main__": run()
