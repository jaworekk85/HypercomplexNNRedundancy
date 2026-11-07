"""Plot saved routing matrices and full validation curves; no new inference."""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.models.channel_routing import LearnableChannelPermutation, log_sinkhorn

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = ROOT / "results" / "learned_grouping_pilot"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    config = summary["config"]
    saved = torch.load(folder / "learned" / "last.pt", map_location="cpu", weights_only=True)["best_state"]
    initial = LearnableChannelPermutation(32, config["seed"]+60000, config["initial_strength"],
                                          config["temperature_start"], config["sinkhorn_iterations"])
    with torch.no_grad():
        initial_matrix = initial.soft_matrix().numpy()
        final_matrix = log_sinkhorn(saved["router.logits"], saved["router.temperature"], config["sinkhorn_iterations"]).numpy()
    indices = saved["router.indices"].numpy()
    final_hard = np.zeros((32, 32))
    final_hard[np.arange(32), indices] = 1
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for axis, matrix, title in zip(axes, (initial_matrix, final_matrix, final_hard),
                                    ("Initial soft assignment", "Final soft assignment (selected state)", "Exported one-to-one assignment")):
        artist = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        axis.set(title=title, xlabel="Input scalar channel", ylabel="Output component position")
    fig.colorbar(artist, ax=axes.tolist(), shrink=.8)
    fig.savefig(folder / "routing_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    paths = {"identity (reused)": ROOT / config["baseline_source"] / "quaternion" / "training.csv",
             "fixed random": folder / "random" / "training.csv", "learned": folder / "learned" / "training.csv"}
    for name, path in paths.items():
        history = list(csv.DictReader(path.open(encoding="utf-8")))
        epochs = [int(row["epoch"]) for row in history]
        axes[0].plot(epochs, [100*float(row["validation_accuracy"]) for row in history], label=name)
        axes[1].plot(epochs, [float(row["validation_loss"]) for row in history], label=name)
    for axis in axes:
        axis.axvline(config["soft_epochs"], color="gray", linestyle=":", label="Hardening / eligible window starts")
        axis.set_xlabel("Epoch")
        axis.legend(fontsize=8)
    axes[0].set(ylabel="Validation accuracy (%)", title="Full curves, one training seed")
    axes[1].set(ylabel="Validation cross-entropy", title="Soft/hard boundary recorded without tuning")
    fig.savefig(folder / "learning_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved routing/learning plots generated", flush=True)
