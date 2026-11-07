"""Resume the bounded replay, BN intervention, and artifact verification."""

import csv

import torch

from scripts.cifar_cnn import write_json
from scripts.replay_routing_boundary import ROOT, read, run as replay
from scripts.diagnose_routing_bn import run as diagnose
from scripts.verify_routing_bn import verify
from scripts.summarize_research import run as summarize


if __name__ == "__main__":
    config = read(ROOT / "configs/routing_bn_diagnostic.json")
    checkpoint = ROOT / config["output"] / "replay" / "last.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["epoch"] == config["boundary_epoch"]:
            # Repair derived files if interrupted after the atomic checkpoint.
            # The frozen trainer/replay and all numeric states stay unchanged.
            with (checkpoint.parent / "training.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(state["history"][0]))
                writer.writeheader()
                writer.writerows(state["history"])
            write_json(checkpoint.parent / "hardening.json", state["hardening"])
    replay(config)
    diagnose(config)
    verify(ROOT / config["output"])
    summarize()
