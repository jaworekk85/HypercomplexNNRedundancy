"""Run one exact frozen seed/model job before the main resumable runner reaches it."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.cifar_cnn import source_hashes, train_one, write_json
from src.data.cifar_cached import file_sha256, load_cached_cifar

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--kind", required=True, choices=("quaternion", "real"))
    args = parser.parse_args()
    source = ROOT / "results" / "cifar_cnn"
    original = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if original["source_hashes"] != source_hashes():
        raise ValueError("Frozen training implementation changed")
    config = {**original["config"], "seed": args.seed}
    torch.set_num_threads(config["cpu_threads"])
    (images, targets), unused_test = load_cached_cifar(ROOT / config["archive"])
    del unused_test
    with np.load(source / "split_indices.npz") as saved:
        indices = {key: saved[key] for key in ("train", "validation", "calibration")}
    statistics = json.loads((source / "data.json").read_text(encoding="utf-8"))["statistics"]
    device = torch.device(config["device"])
    tensors = {key: torch.as_tensor(images[index], device=device) for key, index in indices.items()}
    labels = {key: torch.as_tensor(targets[index], device=device) for key, index in indices.items()}
    mean = torch.tensor(statistics["mean"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(statistics["std"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    target = ROOT / "results" / "cnn_multiseed" / f"seed_{args.seed}" / args.kind
    write_json(target / "worker.json", {"seed": args.seed, "kind": args.kind, "config": config,
               "worker_sha256": file_sha256(Path(__file__)), "source_hashes": source_hashes(),
               "test_inference": False, "purpose": "Independent identical-protocol job; main runner reuses completed metrics."})
    train_one(config, args.kind, tensors, labels, mean, std, target, resume=True)
