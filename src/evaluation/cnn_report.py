"""Held-out spatial activation diagnostics and report for the small CNN."""

import csv
import json

import numpy as np

from src.activations.diagnostics import effective_rank, pair_stability, pair_summary, scalar_pearson
from src.activations.similarity import grouped_cosine


def spatial_diagnostics(values, quaternion):
    # Split by whole images before pooling spatial observations: no image sharing between halves.
    def pooled(array):
        return array.reshape(-1, *array.shape[2:])
    flattened = pooled(values)
    scalar, valid = scalar_pearson(flattened)
    first, _ = scalar_pearson(pooled(values[::2]))
    second, _ = scalar_pearson(pooled(values[1::2]))
    report = {"input_shape": list(values.shape), "spatial_positions_pooled": values.shape[1],
              "scalar_pairs": pair_summary(scalar), "constant_scalar_features": int((~valid).sum()),
              "scalar_stability_across_disjoint_image_halves": pair_stability(first, second),
              "rank": effective_rank(flattened)}
    matrices = {"scalar": scalar}
    if quaternion:
        grouped, valid = grouped_cosine(flattened)
        first, _ = grouped_cosine(pooled(values[::2]))
        second, _ = grouped_cosine(pooled(values[1::2]))
        report["grouped_pairs"] = pair_summary(grouped)
        report["constant_quaternion_units"] = int((~valid).sum())
        report["grouped_stability_across_disjoint_image_halves"] = pair_stability(first, second)
        matrices["grouped"] = grouped
    return report, matrices


def save_cnn_report(records, config, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    colors = {"quaternion": "tab:blue", "real": "tab:green"}
    for record in records:
        kind = record["kind"]
        folder = output_dir / kind
        with (folder / "training.csv").open(encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        epoch = [int(row["epoch"]) for row in history]
        axes[0].plot(epoch, [float(row["train_augmented_accuracy"]) for row in history], color=colors[kind], alpha=0.5, linestyle="--", label=f"{kind}: train (augmented)")
        axes[0].plot(epoch, [float(row["validation_accuracy"]) for row in history], color=colors[kind], label=f"{kind}: validation")
        axes[1].plot(epoch, [float(row["validation_loss"]) for row in history], color=colors[kind], label=kind)
        diagnostics, matrices = {}, {}
        with np.load(folder / "calibration_activations.npz") as arrays:
            for stage in ("initial", "trained"):
                diagnostics[stage], current = spatial_diagnostics(arrays[stage], kind == "quaternion")
                matrices.update({f"{stage}_{name}": matrix for name, matrix in current.items()})
        record["activation_diagnostics"] = diagnostics
        np.savez_compressed(folder / "calibration_similarity.npz", **matrices)
        (folder / "metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    axes[0].axhline(config["target_accuracy"], color="gray", linestyle=":", label="80% target")
    axes[0].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1), title="Online augmented train and clean validation")
    axes[1].set(xlabel="Epoch", ylabel="Cross-entropy", title="Clean held-out validation")
    axes[0].legend(fontsize=7)
    axes[1].legend()
    fig.savefig(output_dir / "learning_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for record in records:
        with np.load(output_dir / record["kind"] / "calibration_similarity.npz") as arrays:
            for axis, key in zip(axes, ("trained_scalar", "trained_grouped")):
                if key not in arrays:
                    continue
                matrix = arrays[key]
                pairs = matrix[np.triu_indices(len(matrix), 1)]
                pairs = pairs[np.isfinite(pairs)]
                if len(pairs):
                    axis.hist(pairs, bins=np.linspace(-1, 1, 41), histtype="step", density=True, linewidth=2, label=record["kind"], color=colors[record["kind"]])
    axes[0].set(title="Scalar Pearson, pooled spatial positions", xlabel="Correlation", ylabel="Pair density")
    axes[1].set(title="Whole quaternion channels", xlabel="Centered grouped cosine", ylabel="Pair density")
    for axis in axes:
        axis.legend()
    fig.savefig(output_dir / "redundancy_histograms.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    lines = ["# CIFAR CNN quality gate", "", "## Hypotheses and goal", "",
             "Target: at least 80% CIFAR-10 test accuracy for a correctly trained quaternion CNN. The target is not a guaranteed result. H1-H3 about redundancy/compression remain open.", "",
             "## Protocol", "",
             "CIFAR-10 (Krizhevsky2009): 48,000 training images and the same disjoint 1000 validation/1000 calibration images as the Dense pilot. Use all 10,000 official test images only after both trainings and checkpoint choices finish. This official test set was already used by the earlier Dense pilot; it is not a newly collected blind benchmark.",
             "One seed, 2026. Six 3x3 convolutions in three blocks, each Conv-BatchNorm-ReLU twice, MaxPool and scalar-channel Dropout2d. Global average pooling and a real ten-logit readout. Quaternion widths 8/16/32 (32/64/128 scalar channels); real widths 16/32/64. Ordinary per-component BatchNorm is used, with no claimed quaternion invariance.",
             "Quaternion input is (0,R,G,B), component-blocked within every feature map. Kernels implement the standard right Hamilton product, following the tensor formulation motivation of NiemczynowiczKycia2026. Each connection has four learned real kernels; the expanded convolution matrix is derived, not independently trained. This is an existing model family, not a proposed novel architecture.",
             f"AdamW: lr {config['learning_rate']}, weight decay {config['weight_decay']}, cosine schedule to {config['minimum_learning_rate']}, {config['epochs']} epochs, batch {config['batch_size']}. Train-only random crop from four zero-padded pixels and horizontal flip. RGB statistics fit only on 48,000 training images. GPU mixed precision; the first-update audit uses float32. Dedicated augmentation and batch generators give the same sample order/augmentation stream across the two models.",
             "Checkpoint selection minimizes clean validation cross-entropy. The calibration set is not used to optimize weights, choose epochs, or select the architecture. Last-epoch checkpoints include optimizer/scheduler/scaler and RNG states for resuming; selected inference checkpoints contain model weights and BatchNorm buffers.", "",
             "## Measured results", "",
             "| Model | Real trainable parameters | BN buffer elements | Best epoch | Selected train accuracy | Validation accuracy | Test accuracy | Test cross-entropy |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in records:
        lines.append(f"| {r['kind']} | {r['real_trainable_parameters']} | {r['nontrainable_buffer_elements']} | {r['best_epoch']} | {100*r['selected_train']['accuracy']:.2f}% | {100*r['selected_validation']['accuracy']:.2f}% | {100*r['test']['accuracy']:.2f}% | {r['test']['cross_entropy']:.4f} |")
    lines.extend(["", "| Model | Calibration tensor | Effective scalar rank (initial → selected) | Scalar pairs ≥0.9 | Grouped pairs ≥0.9 |",
                  "| --- | --- | ---: | ---: | ---: |"])
    for r in records:
        a, b = r["activation_diagnostics"]["initial"], r["activation_diagnostics"]["trained"]
        grouped = b.get("grouped_pairs", {}).get("fraction_ge_0_9")
        group_text = f"{100*grouped:.3f}%" if grouped is not None else "not applicable"
        scalar = b["scalar_pairs"]["fraction_ge_0_9"]
        scalar_text = f"{100*scalar:.3f}%" if scalar is not None else "undefined"
        lines.append(f"| {r['kind']} | {r['activation_shape']} | {a['rank']['effective_rank']:.2f} → {b['rank']['effective_rank']:.2f} | {scalar_text} | {group_text} |")
    primary = next(r for r in records if r["kind"] == "quaternion")
    lines.extend(["", f"Primary 80% target achieved: **{primary['target_80_percent_achieved']}**.", "",
                  "## Checks and interpretation", "",
                  "All expected parameters, every quaternion convolution component's gradients/updates, and checkpoint output equivalence were checked before interpreting the run. Explicit audits and full shapes are in per-model JSON. Nontrainable BatchNorm running statistics are separated from real trainable parameters.",
                  "Similarity pools spatial positions and calibration examples. Stability comparisons split by disjoint whole-image halves before pooling, avoiding comparisons of different pixels from the same image as if they were held-out images. Constant features are excluded; 0.9 is a descriptive threshold rather than proof of removable channels.",
                  "This is a one-seed quality check. The real/quaternion budgets are close but not identical; compute and effective width differ. A higher accuracy than the Dense pilot cannot be attributed solely to architecture: training-set size, augmentation, optimization and framework also changed. No general quaternion performance gain is established. No units were pruned, no weights masked, and actual compact-architecture savings are zero.",
                  "## Reproducibility", "",
                  "config.json, data.json and split_indices.npz record choices and held-out indices. summary.json records source hashes, package versions, device and seed; environment.freeze.txt records the full GPU environment. Histories, selected/last checkpoints, calibration features and final test logits are saved per model."])
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
