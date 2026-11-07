"""Run upstream audit, a tiny synthetic classifier, and calibration diagnostics.

Run from the project root: python -m scripts.phase0
This is a Phase 0 infrastructure check, not a dataset benchmark or pruning study.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.runtime import (
    ROOT, configure, environment, np, real_parameter_count, source_hashes, tf, write_json,
)
from scripts.check_upstream import run_audit
from src.activations.similarity import grouped_cosine
from src.models.quaternion_dense import QuaternionDense


def make_data(config):
    # Separate deterministic RNG streams; calibration/test never enter training.
    streams = np.random.SeedSequence(config["seed"]).spawn(4)
    width = 4 * config["input_quaternion_units"]
    teacher = np.random.default_rng(streams[0]).normal(size=(width, config["classes"]))
    data = {"teacher": teacher}
    for split, stream in zip(("train", "calibration", "test"), streams[1:]):
        x = np.random.default_rng(stream).normal(size=(config[f"{split}_samples"], width)).astype("float32")
        data[f"{split}_x"] = x.reshape(-1, config["input_quaternion_units"], 4)
        data[f"{split}_y"] = np.argmax(x @ teacher, axis=1).astype("int32")
    return data


def build_classifier(config):
    inputs = tf.keras.Input((config["input_quaternion_units"], 4))
    activations = QuaternionDense(
        config["hidden_quaternion_units"], activation=config["activation"],
        kernel_initializer=config["kernel_initializer"], name="quaternion_hidden",
    )(inputs)
    logits = tf.keras.layers.Dense(config["classes"], name="real_readout")(
        tf.keras.layers.Flatten()(activations)
    )
    return tf.keras.Model(inputs, logits), tf.keras.Model(inputs, activations)


def evaluate(model, inputs, labels):
    logits = model(inputs, training=False)
    return {
        "cross_entropy": float(tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))),
        "accuracy": float(np.mean(np.argmax(logits.numpy(), axis=1) == labels)),
    }


def audit_first_step(model, inputs, labels, optimizer):
    variables = model.trainable_variables
    before = [v.numpy().copy() for v in variables]
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
    gradients = tape.gradient(loss, variables)
    if any(g is None or not np.isfinite(g.numpy()).all() for g in gradients):
        raise RuntimeError("A model parameter has missing or nonfinite gradients")
    optimizer.apply_gradients(zip(gradients, variables))
    rows = []
    for variable, old, gradient in zip(variables, before, gradients):
        rows.append({
            "name": variable.path,
            "shape": list(variable.shape),
            "real_parameters": int(np.prod(variable.shape)),
            "gradient_norm": float(tf.linalg.global_norm([gradient])),
            "max_abs_update": float(np.max(np.abs(variable.numpy() - old))),
        })
    kernel_index = next(i for i, v in enumerate(variables) if v is model.get_layer("quaternion_hidden").kernel)
    kernel = variables[kernel_index]
    return {
        "variables": rows,
        "all_variables_updated": all(row["max_abs_update"] > 0 for row in rows),
        "kernel_component_gradient_norms": np.linalg.norm(gradients[kernel_index].numpy().reshape(-1, 4), axis=0).tolist(),
        "kernel_component_update_norms": np.linalg.norm((kernel.numpy() - before[kernel_index]).reshape(-1, 4), axis=0).tolist(),
    }


def save_diagnostics(output_dir, initial, trained):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrices, summary = {}, {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.3), layout="constrained")
    fig_hist, ax_hist = plt.subplots(figsize=(6, 4), layout="constrained")
    for axis, (name, activations) in zip(axes, (("initial", initial), ("trained", trained))):
        matrix, valid = grouped_cosine(activations)
        matrices[name] = matrix
        pairs = matrix[np.triu_indices(len(matrix), k=1)]
        pairs = pairs[np.isfinite(pairs)]  # Exclude diagonals and constant units.
        summary[name] = {
            "valid_units": int(valid.sum()),
            "zero_variance_units": np.flatnonzero(~valid).tolist(),
            "finite_off_diagonal_pairs": len(pairs),
            "mean_pairwise_similarity": float(pairs.mean()) if len(pairs) else None,
            "max_pairwise_similarity": float(pairs.max()) if len(pairs) else None,
        }
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad("lightgray")
        heat = axis.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-1, vmax=1)
        axis.set(title=f"{name.capitalize()} (synthetic calibration)", xlabel="Quaternion unit", ylabel="Quaternion unit")
        if len(pairs):
            ax_hist.hist(pairs, bins=np.linspace(-1, 1, 17), histtype="step", linewidth=2, label=name)
    fig.colorbar(heat, ax=axes, label="Centered grouped cosine", shrink=0.8)
    fig.savefig(output_dir / "similarity_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    ax_hist.set(xlabel="Centered grouped cosine (off-diagonal pairs)", ylabel="Number of pairs", title="Synthetic calibration diagnostic; no pruning")
    ax_hist.legend()
    fig_hist.savefig(output_dir / "redundancy_histogram.png", dpi=160, bbox_inches="tight")
    plt.close(fig_hist)
    np.savez_compressed(output_dir / "activations.npz", initial=initial, trained=trained,
                        initial_similarity=matrices["initial"], trained_similarity=matrices["trained"])
    return summary


def run(config, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    upstream = run_audit(config["seed"])
    write_json(output_dir / "upstream_smoke.json", upstream)
    print(f"Unmodified upstream gate: {upstream['status']} (recorded separately)", flush=True)
    configure(config["seed"])
    data = make_data(config)
    np.savez_compressed(output_dir / "synthetic_data.npz", **data)
    model, collector = build_classifier(config)
    initial_activations = collector(data["calibration_x"], training=False).numpy()
    initial_train = evaluate(model, data["train_x"], data["train_y"])
    initial_kernel = model.get_layer("quaternion_hidden").kernel.numpy().copy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    first_step = audit_first_step(model, data["train_x"], data["train_y"], optimizer)

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
        gradients = tape.gradient(loss, model.trainable_variables)
        checked = [tf.debugging.check_numerics(g, "Nonfinite gradient") for g in gradients]
        optimizer.apply_gradients(zip(checked, model.trainable_variables))
        return loss

    history = [{"step": 0, **initial_train}, {"step": 1, **evaluate(model, data["train_x"], data["train_y"])}]
    for step in range(2, config["steps"] + 1):
        train_step(data["train_x"], data["train_y"])
        if step % 20 == 0 or step == config["steps"]:
            metrics = evaluate(model, data["train_x"], data["train_y"])
            history.append({"step": step, **metrics})
            print(f"Step {step}: training loss={metrics['cross_entropy']:.4f}", flush=True)
    with (output_dir / "training.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "cross_entropy", "accuracy"])
        writer.writeheader()
        writer.writerows(history)
    final_train = evaluate(model, data["train_x"], data["train_y"])
    # No early stopping or tuning on calibration/test. Evaluate test only now.
    test_metrics = evaluate(model, data["test_x"], data["test_y"])
    trained_activations = collector(data["calibration_x"], training=False).numpy()
    similarities = save_diagnostics(output_dir, initial_activations, trained_activations)
    model_path = output_dir / "model.keras"
    model.save(model_path)
    restored = tf.keras.models.load_model(model_path)
    reload_error = float(np.max(np.abs(restored(data["calibration_x"]).numpy() - model(data["calibration_x"]).numpy())))
    hidden = model.get_layer("quaternion_hidden")
    expected_hidden = 4 * (config["input_quaternion_units"] + 1) * config["hidden_quaternion_units"]
    expected_readout = (4 * config["hidden_quaternion_units"] + 1) * config["classes"]
    checks = {
        "all_expected_parameters_tracked": real_parameter_count(model.trainable_variables) == expected_hidden + expected_readout,
        "quaternion_kernel_is_tracked": any(v is hidden.kernel for v in model.trainable_variables),
        "all_variables_have_finite_nonzero_gradients": all(row["gradient_norm"] > 0 for row in first_step["variables"]),
        "all_variables_updated": first_step["all_variables_updated"],
        "all_four_kernel_components_receive_gradients": all(v > 0 for v in first_step["kernel_component_gradient_norms"]),
        "all_four_kernel_components_update": all(v > 0 for v in first_step["kernel_component_update_norms"]),
        "kernel_components_not_identically_initialized": not bool(np.all(initial_kernel == initial_kernel[..., 0:1])),
        "classifier_output_shape": list(model(data["test_x"]).shape) == [config["test_samples"], config["classes"]],
        "grouped_activation_shape": list(trained_activations.shape) == [config["calibration_samples"], config["hidden_quaternion_units"], 4],
        "training_loss_decreased": final_train["cross_entropy"] < initial_train["cross_entropy"],
        "checkpoint_roundtrip": reload_error <= 1e-6,
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "environment": environment(config["seed"]),
        "config": config,
        "upstream_status": upstream["status"],
        "implementation": "Local QuaternionDense fallback; upstream failure is preserved in upstream_smoke.json.",
        "parameter_counts": {
            "hidden_quaternion_units": config["hidden_quaternion_units"],
            "quaternion_layer_real_trainable": hidden.count_params(),
            "real_readout_trainable": expected_readout,
            "total_real_trainable": real_parameter_count(model.trainable_variables),
            "masked_parameters": 0,
            "removed_units": 0,
            "compact_architecture_savings": 0,
        },
        "first_optimizer_step": first_step,
        "metrics": {"initial_train": initial_train, "final_train": final_train, "held_out_synthetic_test": test_metrics},
        "calibration": {
            "layout": "(samples, quaternion_units, 4); component order (real, i, j, k)",
            "shape": list(trained_activations.shape),
            "metric": config["similarity"],
            "diagnostics": similarities,
            "uses_test_data": False,
        },
        "model_checkpoint_bytes": model_path.stat().st_size,
        "checkpoint_max_abs_reload_error": reload_error,
        "checks": checks,
        "source_hashes": {name: source_hashes(ROOT / name) for name in (
            "scripts/phase0.py", "scripts/check_upstream.py", "scripts/runtime.py",
            "src/models/quaternion_dense.py", "src/activations/similarity.py",
        )},
        "data_sha256": source_hashes(output_dir / "synthetic_data.npz")["sha256"],
        "limitations": [
            "One seed and synthetic data only; this is a trainability check, not evidence for H1-H3.",
            "No parameter-matched real control, pruning, compression gain, compute or latency comparison.",
            "Pairwise activation similarity does not establish removable units or novelty.",
            "CPU only; GPU and legacy TensorFlow/Keras combinations were not tested.",
        ],
    }
    write_json(output_dir / "local_smoke.json", report)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=True)
    (output_dir / "environment.freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    write_json(output_dir / "config.json", config)
    print(f"Local Gate A: {report['status']}; real trainable parameters={model.count_params()}")
    print(f"Training cross-entropy: {initial_train['cross_entropy']:.4f} -> {final_train['cross_entropy']:.4f}")
    print(f"Artifacts: {output_dir}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase0.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "phase0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["steps"] < 1 or config["optimizer"] != "Adam":
        parser.error("Phase 0 requires at least one update and supports Adam only")
    report = run(config, args.output_dir)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
