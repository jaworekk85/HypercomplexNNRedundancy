"""Visualize saved membership and training trajectories; no new inference."""

import csv
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))


def plot(output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    output = Path(output)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    reference = ROOT / summary["protocol"]["study"]["pilot_reference"]
    trace = json.loads((output / "learned" / "assignment_trace.json").read_text(encoding="utf-8"))
    histories = {}
    for name, path in (("Learned hard assignment", output / "learned" / "training_discrete.csv"),
                       ("Fixed random assignment", reference / "random" / "training.csv")):
        with path.open(encoding="utf-8") as handle:
            histories[name] = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for name, history in histories.items():
        axes[0].plot([int(row["epoch"]) for row in history], [100*float(row["validation_accuracy"]) for row in history], label=name)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[0].set_title("Original BN, before final refit")
    axes[0].legend(fontsize=8)
    axes[1].plot([row["epoch"] for row in trace], [row["new_unordered_groups"] for row in trace], color="#55a077")
    axes[1].set_ylim(-.2, 8.3)
    axes[1].set_ylabel("New unordered groups vs initial (of 8)")
    axes[1].set_title("Actual membership changes")
    axes[2].plot([row["epoch"] for row in trace], [row["assignment_switch_events"] for row in trace], color="#bb5545")
    axes[2].set_ylabel("Cumulative matching-switch events")
    axes[2].set_title("Adaptive forwards, including validation")
    for ax in axes:
        ax.axvline(40, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Epoch (assignment frozen after 40)")
        ax.grid(alpha=.15)
    fig.suptitle("Discrete-forward grouping pilot, seed 2026; fixed control reused")
    fig.tight_layout()
    fig.savefig(output / "learning_and_membership.png", dpi=180)
    plt.close(fig)
    initial = np.asarray(summary["membership"]["initial_matching"]).reshape(4, 8).T
    final = np.asarray(summary["membership"]["exported_matching"]).reshape(4, 8).T
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    colors = ListedColormap(["#c7dbec", "#bde0ce", "#f5d5b4", "#e2cce9"])
    for ax, matrix, title in zip(axes, (initial, final), ("Initial fixed-random groups", "Selected learned groups")):
        ax.imshow(matrix//8, cmap=colors, vmin=0, vmax=3, aspect="auto")
        for row in range(8):
            for col in range(4):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=12)
        ax.set_xticks(range(4), ["r position", "i position", "j position", "k position"])
        ax.set_yticks(range(8), [f"Group {row+1}" for row in range(8)])
        ax.set_title(title)
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
    fig.suptitle("Cells: original scalar channel IDs (0-31); color: original component block")
    fig.tight_layout()
    fig.savefig(output / "group_membership.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot(ROOT / "results/hard_forward_grouping")
