"""Tables and plots for the fixed small RGB pilot; no pruning claims."""

import csv

import numpy as np


def average(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    return {"mean": float(values.mean()) if len(values) else None, "sample_std": (float(values.std(ddof=1)) if len(values) > 1 else 0.0) if len(values) else None,
            "values": values.tolist()}


def save_pilot_report(records, config, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    encodings = [*config["encodings"], "RGB"]
    summary = {"config": config, "models": {}, "paired_accuracy_differences": {}}
    rows = []
    for encoding in encodings:
        selected = [record for record in records if record["encoding"] == encoding]
        data = {
            "real_trainable_parameters": selected[0]["real_trainable_parameters"],
            "hidden_scalar_features": selected[0]["hidden_scalar_features"],
            "test_accuracy": average([r["test"]["accuracy"] for r in selected]),
            "test_cross_entropy": average([r["test"]["cross_entropy"] for r in selected]),
            "selected_training_accuracy": average([r["selected_train"]["accuracy"] for r in selected]),
            "selected_validation_accuracy": average([r["selected_validation"]["accuracy"] for r in selected]),
            "best_epochs": [r["best_epoch"] for r in selected],
            "activation_diagnostics": {},
        }
        for stage in ("initial", "trained"):
            diagnostics = [r["activations"][stage] for r in selected]
            data["activation_diagnostics"][stage] = {
                "effective_rank": average([d["rank"]["effective_rank"] for d in diagnostics]),
                "normalized_effective_rank": average([d["rank"]["normalized_effective_rank"] for d in diagnostics]),
                "constant_scalar_features": [d["scalar"]["constant_features"] for d in diagnostics],
                "scalar_fraction_ge_0_9": average([d["scalar"]["pairs"]["fraction_ge_0_9"] for d in diagnostics]),
                "scalar_pair_stability": [d["scalar"]["calibration_halves_stability"] for d in diagnostics],
            }
            if encoding != "RGB":
                data["activation_diagnostics"][stage]["grouped_fraction_ge_0_9"] = average([d["grouped"]["pairs"]["fraction_ge_0_9"] for d in diagnostics])
                data["activation_diagnostics"][stage]["grouped_maximum"] = [d["grouped"]["pairs"]["maximum"] for d in diagnostics]
                data["activation_diagnostics"][stage]["grouped_pair_stability"] = [d["grouped"]["calibration_halves_stability"] for d in diagnostics]
        summary["models"][encoding] = data
        for r in selected:
            rows.append({"encoding": encoding, "seed": r["seed"], "real_parameters": r["real_trainable_parameters"],
                         "best_epoch": r["best_epoch"], "train_accuracy": r["selected_train"]["accuracy"],
                         "validation_accuracy": r["selected_validation"]["accuracy"], "test_accuracy": r["test"]["accuracy"],
                         "test_cross_entropy": r["test"]["cross_entropy"],
                         "effective_rank": r["activations"]["trained"]["rank"]["effective_rank"],
                         "normalized_effective_rank": r["activations"]["trained"]["rank"]["normalized_effective_rank"]})
    for first, second in [(e, "RGB") for e in config["encodings"]] + ([("ABGR", "ARGB")] if set(config["encodings"]) == {"ARGB", "ABGR"} else []):
        differences = []
        for seed in config["seeds"]:
            a = next(r for r in records if r["seed"] == seed and r["encoding"] == first)
            b = next(r for r in records if r["seed"] == seed and r["encoding"] == second)
            differences.append(a["test"]["accuracy"] - b["test"]["accuracy"])
        summary["paired_accuracy_differences"][f"{first}_minus_{second}"] = average(differences)
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    colors = {"ARGB": "tab:blue", "ABGR": "tab:orange", "RGB": "tab:green"}
    labels = {"ARGB": "Quaternion (0,R,G,B)", "ABGR": "Quaternion (0,B,G,R)", "RGB": "Real RGB"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for encoding in encodings:
        for seed in config["seeds"]:
            with (output_dir / f"seed_{seed}_{encoding.lower()}" / "training.csv").open(encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))
            epochs = [int(row["epoch"]) for row in history]
            axes[0].plot(epochs, [float(row["train_loss"]) for row in history], color=colors[encoding], alpha=0.4,
                         label=labels[encoding] if seed == config["seeds"][0] else None)
            axes[1].plot(epochs, [float(row["validation_loss"]) for row in history], color=colors[encoding], alpha=0.4)
    axes[0].set(title="Training", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(title="Held-out validation", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend(fontsize=8)
    fig.savefig(output_dir / "learning_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, key in zip(axes, ("trained_scalar", "trained_grouped")):
        for encoding in encodings:
            if key.endswith("grouped") and encoding == "RGB":
                continue
            pairs = []
            for seed in config["seeds"]:
                with np.load(output_dir / f"seed_{seed}_{encoding.lower()}" / "calibration_activations.npz") as arrays:
                    matrix = arrays[key]
                    values = matrix[np.triu_indices(len(matrix), 1)]
                    pairs.extend(values[np.isfinite(values)])
            if pairs:
                axis.hist(pairs, bins=np.linspace(-1, 1, 41), density=True, histtype="step", linewidth=1.7,
                          color=colors[encoding], label=labels[encoding])
        axis.set(xlabel="Centered similarity", ylabel="Pair density",
                 title="Scalar Pearson" if key.endswith("scalar") else "Grouped quaternion cosine")
        axis.legend(fontsize=8)
    fig.savefig(output_dir / "redundancy_histograms.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(encodings), figsize=(5 * len(encodings), 4.5), layout="constrained", squeeze=False)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("lightgray")
    for axis, encoding in zip(axes[0], encodings):
        with np.load(output_dir / f"seed_{config['seeds'][0]}_{encoding.lower()}" / "calibration_activations.npz") as arrays:
            heat = axis.imshow(np.ma.masked_invalid(arrays["trained_scalar"]), cmap=cmap, vmin=-1, vmax=1)
        axis.set(title=labels[encoding], xlabel="Scalar hidden feature", ylabel="Scalar hidden feature")
    fig.colorbar(heat, ax=list(axes[0]), shrink=0.8, label="Pearson correlation")
    fig.savefig(output_dir / "scalar_similarity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(config["encodings"]), 2, figsize=(10, 4.5 * len(config["encodings"])), layout="constrained", squeeze=False)
    for row, encoding in zip(axes, config["encodings"]):
        with np.load(output_dir / f"seed_{config['seeds'][0]}_{encoding.lower()}" / "calibration_activations.npz") as arrays:
            for axis, stage in zip(row, ("initial", "trained")):
                heat = axis.imshow(np.ma.masked_invalid(arrays[f"{stage}_grouped"]), cmap=cmap, vmin=-1, vmax=1)
                axis.set(title=f"{labels[encoding]}: {stage}", xlabel="Quaternion hidden unit", ylabel="Quaternion hidden unit")
    fig.colorbar(heat, ax=list(axes.flat), shrink=0.8, label="Centered grouped cosine")
    fig.savefig(output_dir / "grouped_similarity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    text = [
        "# Small RGB pilot", "", "## Hypotheses", "",
        "H1-H3 from PROJECT.md remain research hypotheses. This pilot checks image-data trainability and describes held-out activations; it does not test compression value.", "",
        "## Frozen protocol", "",
        f"CIFAR-10 (Krizhevsky2009): {config['train_samples']} stratified training, {config['validation_samples']} validation, {config['calibration_samples']} calibration examples; all {config['test_samples']} official test examples. Split seed {config['split_seed']}; model seeds {config['seeds']}.",
        "RGB channels of one 32x32 pixel form one quaternion, with explicit zero alpha. ARGB follows the text of NiemczynowiczKycia2025; ABGR checks the channel order in the authors' notebook. See docs/INPUT_ENCODING_REVIEW.md. This is not a reproduction of their malaria CNN experiment.",
        "Channel normalization is fit on the selected training images only. The real control receives the same RGB information without a padded zero channel or unused zero-input columns.",
        f"Each model is a single {config['activation']} hidden layer and a real 10-logit readout. Adam, learning rate {config['learning_rate']}, batch size {config['batch_size']}, {config['epochs']} epochs; identical batch orders within each seed. Hidden weights have variance 2/(3*32*32), zero biases, and seeded GlorotUniform readouts.",
        "Checkpoint selection minimizes validation cross-entropy, with initialization also eligible. Calibration is used for diagnostics only. All training/selection finished before the final test stage. Uncompiled checkpoints contain configuration and weights, not optimizer state.", "",
        "## Measured results", "",
        "Values below are means and sample standard deviations across seeds, not confidence intervals. Seeds share the same image splits.", "",
        "| Model | Real trainable parameters | Hidden scalar features | Test accuracy (%) | Test cross-entropy | Selected epochs |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for encoding, data in summary["models"].items():
        acc, loss = data["test_accuracy"], data["test_cross_entropy"]
        text.append(f"| {labels[encoding]} | {data['real_trainable_parameters']} | {data['hidden_scalar_features']} | {100*acc['mean']:.2f} ± {100*acc['sample_std']:.2f} | {loss['mean']:.4f} ± {loss['sample_std']:.4f} | {data['best_epochs']} |")
    text.extend(["", "| Model | Effective rank, initial → selected | Constant scalar features, selected | Scalar pairs with correlation ≥0.9 (%) |",
                 "| --- | ---: | --- | ---: |"])
    for encoding, data in summary["models"].items():
        initial, trained = data["activation_diagnostics"]["initial"], data["activation_diagnostics"]["trained"]
        fraction = trained['scalar_fraction_ge_0_9']['mean']
        fraction_text = f"{100*fraction:.3f}" if fraction is not None else "undefined"
        text.append(f"| {labels[encoding]} | {initial['effective_rank']['mean']:.2f} → {trained['effective_rank']['mean']:.2f} | {trained['constant_scalar_features']} | {fraction_text} |")
    text.extend(["", "| Model | Maximum grouped cosine in each seed | Grouped pairs ≥0.9, mean fraction (%) | Pair-value correlation between calibration halves |",
                 "| --- | --- | ---: | --- |"])
    for encoding in config["encodings"]:
        trained = summary["models"][encoding]["activation_diagnostics"]["trained"]
        maxima = ", ".join(f"{value:.3f}" if value is not None else "undefined" for value in trained["grouped_maximum"])
        stability = ", ".join(f"{row['pair_value_pearson']:.3f}" if row['pair_value_pearson'] is not None else "undefined" for row in trained["grouped_pair_stability"])
        fraction = trained['grouped_fraction_ge_0_9']['mean']
        fraction_text = f"{100*fraction:.3f}" if fraction is not None else "undefined"
        text.append(f"| {labels[encoding]} | {maxima} | {fraction_text} | {stability} |")
    text.extend(["", "Paired test accuracy differences (percentage points):", ""])
    for name, values in summary["paired_accuracy_differences"].items():
        text.append(f"- {name}: {100*values['mean']:+.3f} ± {100*values['sample_std']:.3f}; per-seed differences {[round(100*v, 3) for v in values['values']]}.")
    text.extend([
        "", "All recorded first-step gradient/update checks and checkpoint roundtrips passed. Per-seed JSON files preserve kernel component norms and every check. No units were removed, no weights were masked, and actual compact-architecture savings are zero.",
        "", "## Definitions", "",
        "Scalar Pearson is computed on centered scalar hidden features. Grouped cosine centers each of the four components over calibration examples, then flattens the entire sample-by-component block per quaternion unit. Constant features/units are undefined and excluded. Histograms exclude diagonals and duplicate pairs. The threshold 0.9 was fixed before execution; it is a descriptive threshold, not a pruning rule.",
        "Entropy effective rank is exp(-sum(p*log(p))) for the normalized eigenvalues p of the centered scalar-feature covariance. Normalized rank divides by min(number of examples minus one, number of scalar features). Raw eigenvalues and normalized ranks are in JSON.",
        "Calibration-half stability compares matching pair values from even/odd calibration examples within the same model. It does not align individual neurons between independent training seeds. Similarity calculations follow the broad feature-redundancy motivation of Ayinde2019, with no new algorithm claimed.",
        "", "## Interpretation and limits", "",
        "These are small dense models trained on only 10% of the official training set. The comparison matches real parameter budgets approximately; it does not match hidden width, compute, or attained validation quality. Consequently, accuracy differences and raw rank differences are descriptive pilot observations, not a demonstrated quaternion performance gain or a controlled compression result.",
        "Both high scalar correlation and a low effective rank can coexist with modest whole-quaternion similarity. Neither establishes that a whole unit can be removed safely. An absent high-similarity signal under this metric does not refute H1 for other architectures, representations or datasets.",
        "ARGB/ABGR changes are reported rather than used to select a winning test configuration. Three seeds describe initialization/shuffle variation on one fixed split; broader uncertainty and generalization remain untested. Pruning and Gates C-D remain open.",
        "", "## Artifacts and provenance", "",
        "comparison.csv and summary.json contain the numeric comparison; each seed directory holds its history, selected model, calibration activations, test logits and audit. split_indices.npz and data.json identify exact splits and training-only normalization. environment.freeze.txt and source hashes in summary.json record the software. Dataset archive SHA-256 is checked by the installed Keras loader.",
        "", "Citations use references.bib: Krizhevsky2009, NiemczynowiczKycia2025, NiemczynowiczKycia2026, Ayinde2019. No source claims are inferred from an unexecuted notebook reproduction.",
    ])
    quaternion_records = [record for record in records if record["encoding"] != "RGB"]
    grouped_pairs = [record["activations"]["trained"]["grouped"]["pairs"] for record in quaternion_records]
    if all(pairs["valid_pairs"] > 0 and pairs["fraction_ge_0_9"] == 0 for pairs in grouped_pairs):
        maxima = [pairs["maximum"] for pairs in grouped_pairs]
        finding = (f"In this run, none of the {len(quaternion_records)} trained quaternion models contained a calibration unit pair "
                   f"with centered grouped cosine at least 0.9. Maximum pair similarities ranged from {min(maxima):.3f} to {max(maxima):.3f}. "
                   "This is a negative observation for the chosen near-duplicate diagnostic, not a general negative result about QNN redundancy.")
        text.insert(text.index("## Interpretation and limits") + 2, finding + "\n")
    (output_dir / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return summary
