"""Small, held-out RGB pilot: real MLP versus quaternion ARGB and ABGR MLPs."""

import argparse
import csv
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts.runtime import ROOT, configure, environment, np, real_parameter_count, source_hashes, tf, write_json
from scripts.download_cifar10 import ARCHIVE_SHA256
from src.data.rgb import encode_pixels, fit_channel_statistics, normalize_rgb, stratified_indices
from src.models.rgb_baselines import build_rgb_model
from src.activations.diagnostics import activation_diagnostics


def batched_output(model, inputs, batch_size=256):
    return np.concatenate([model(inputs[i:i + batch_size], training=False).numpy()
                           for i in range(0, len(inputs), batch_size)])


def evaluate(model, inputs, labels):
    logits = batched_output(model, inputs)
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True).numpy()
    return {"cross_entropy": float(loss.mean()), "accuracy": float(np.mean(np.argmax(logits, axis=1) == labels))}


def first_update(model, inputs, labels, optimizer):
    variables = model.trainable_variables
    before = [v.numpy().copy() for v in variables]
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, model(inputs, training=True), from_logits=True))
    gradients = tape.gradient(loss, variables)
    if any(g is None or not np.isfinite(g.numpy()).all() for g in gradients):
        raise RuntimeError("Missing or nonfinite gradient before training")
    optimizer.apply_gradients(zip(gradients, variables))
    rows = [{"name": v.path, "shape": list(v.shape), "gradient_norm": float(tf.linalg.global_norm([g])),
             "max_abs_update": float(np.max(np.abs(v.numpy() - old)))}
            for v, old, g in zip(variables, before, gradients)]
    result = {"variables": rows, "all_variables_updated": all(row["max_abs_update"] > 0 for row in rows)}
    if any(layer.name == "quaternion_hidden" for layer in model.layers):
        kernel = model.get_layer("quaternion_hidden").kernel
        index = next(i for i, variable in enumerate(variables) if variable is kernel)
        result["kernel_component_gradient_norms"] = np.linalg.norm(gradients[index].numpy().reshape(-1, 4), axis=0).tolist()
        result["kernel_component_update_norms"] = np.linalg.norm((kernel.numpy() - before[index]).reshape(-1, 4), axis=0).tolist()
        if not all(v > 0 for v in result["kernel_component_gradient_norms"] + result["kernel_component_update_norms"]):
            raise RuntimeError("At least one quaternion kernel component is not training")
    if not result["all_variables_updated"]:
        raise RuntimeError("A trainable variable did not change after the first update")
    return result


def train_one(config, encoding, seed, rgb, labels, output_dir):
    """Receives training/validation/calibration only; test is a separate later stage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.backend.clear_session()
    configure(seed)
    model, collector, budgets = build_rgb_model(config, encoding, seed)
    inputs = {split: encode_pixels(images, encoding) for split, images in rgb.items()}
    initial_activations = batched_output(collector, inputs["calibration"])
    initial_train = evaluate(model, inputs["train"], labels["train"])
    initial_validation = evaluate(model, inputs["validation"], labels["validation"])
    best_validation_loss = initial_validation["cross_entropy"]
    best_weights = model.get_weights()
    best_epoch = 0
    history = [{"epoch": 0, "train_loss": initial_train["cross_entropy"], "train_accuracy": initial_train["accuracy"],
                "validation_loss": initial_validation["cross_entropy"], "validation_accuracy": initial_validation["accuracy"]}]
    optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
    rng = np.random.default_rng(seed + 40000)  # Same batches for all three models within a seed.
    audit = None

    @tf.function(input_signature=[tf.TensorSpec((None,) + inputs["train"].shape[1:], tf.float32), tf.TensorSpec((None,), tf.int32)])
    def step(x, y):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, model(x, training=True), from_logits=True))
        gradients = tape.gradient(loss, model.trainable_variables)
        gradients = [tf.debugging.check_numerics(g, "Nonfinite gradient") for g in gradients]
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    for epoch in range(1, config["epochs"] + 1):
        indices = rng.permutation(len(inputs["train"]))
        for start in range(0, len(indices), config["batch_size"]):
            batch = indices[start:start + config["batch_size"]]
            if audit is None:
                audit = first_update(model, inputs["train"][batch], labels["train"][batch], optimizer)
            else:
                step(inputs["train"][batch], labels["train"][batch])
        train_metrics = evaluate(model, inputs["train"], labels["train"])
        validation_metrics = evaluate(model, inputs["validation"], labels["validation"])
        history.append({"epoch": epoch, "train_loss": train_metrics["cross_entropy"], "train_accuracy": train_metrics["accuracy"],
                        "validation_loss": validation_metrics["cross_entropy"], "validation_accuracy": validation_metrics["accuracy"]})
        if validation_metrics["cross_entropy"] < best_validation_loss:
            best_validation_loss = validation_metrics["cross_entropy"]
            best_weights = model.get_weights()
            best_epoch = epoch
        if epoch == 1 or epoch % 5 == 0:
            print(f"{encoding} seed={seed} epoch={epoch}: train={train_metrics['accuracy']:.3f}, validation={validation_metrics['accuracy']:.3f}", flush=True)

    last_epoch_train = train_metrics
    model.set_weights(best_weights)
    selected_train = evaluate(model, inputs["train"], labels["train"])
    selected_validation = evaluate(model, inputs["validation"], labels["validation"])
    trained_activations = batched_output(collector, inputs["calibration"])
    before_report, before_matrices = activation_diagnostics(initial_activations)
    after_report, after_matrices = activation_diagnostics(trained_activations)
    np.savez_compressed(output_dir / "calibration_activations.npz", initial=initial_activations, trained=trained_activations,
                        **{f"initial_{key}": value for key, value in before_matrices.items()},
                        **{f"trained_{key}": value for key, value in after_matrices.items()})
    with (output_dir / "training.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    model.save(output_dir / "model.keras")
    restored = tf.keras.models.load_model(output_dir / "model.keras")
    reload_error = float(np.max(np.abs(model(inputs["validation"][:32]).numpy() - restored(inputs["validation"][:32]).numpy())))
    checks = {
        "all_expected_real_parameters_tracked": real_parameter_count(model.trainable_variables) == model.count_params(),
        "all_variables_updated_on_actual_rgb": audit["all_variables_updated"],
        "finite_selected_train_and_validation_metrics": all(np.isfinite(v) for d in (selected_train, selected_validation) for v in d.values()),
        "checkpoint_roundtrip": reload_error <= 1e-6,
        "finite_calibration_activations": bool(np.isfinite(trained_activations).all()),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed", "encoding": encoding, "seed": seed,
        "parameter_budgets": budgets, "real_trainable_parameters": real_parameter_count(model.trainable_variables),
        "hidden_quaternion_units": config["quaternion_hidden_units"] if encoding != "RGB" else None,
        "hidden_scalar_features": int(np.prod(trained_activations.shape[1:])),
        "first_optimizer_step": audit, "best_epoch": best_epoch,
        "initial_train": initial_train, "last_epoch_train": last_epoch_train,
        "selected_train": selected_train, "selected_validation": selected_validation,
        "activations": {"initial": before_report, "trained": after_report},
        "checkpoint_bytes": (output_dir / "model.keras").stat().st_size,
        "checkpoint_max_abs_reload_error": reload_error, "checks": checks,
        "calibration_used_for_training_or_selection": False, "test_evaluated": False,
        "masked_parameters": 0, "removed_units": 0, "compact_architecture_savings": 0,
    }
    write_json(output_dir / "metrics.json", report)
    del model, restored, collector, inputs, step
    gc.collect()
    return report


def run(config, output_dir):
    if (output_dir / "summary.json").exists():
        raise FileExistsError("Results already exist; use --output-dir for a new run")
    tf.config.threading.set_intra_op_parallelism_threads(config["intra_op_threads"])
    tf.config.threading.set_inter_op_parallelism_threads(config["inter_op_threads"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)  # Freeze choices before loading any test data.
    (raw, y), (test_raw, test_y) = tf.keras.datasets.cifar10.load_data()
    if list(raw.shape[1:]) != config["image_shape"] or len(test_raw) != config["test_samples"]:
        raise ValueError("Dataset shape differs from the frozen configuration")
    y, test_y = y.reshape(-1).astype(np.int32), test_y.reshape(-1).astype(np.int32)
    indices = stratified_indices(y, {name: config[f"{name}_samples"] for name in ("train", "validation", "calibration")}, config["split_seed"], config["classes"])
    joined = np.concatenate(list(indices.values()))
    if len(np.unique(joined)) != len(joined):
        raise RuntimeError("Training/validation/calibration overlap")
    statistics = fit_channel_statistics(raw[indices["train"]])
    rgb = {split: normalize_rgb(raw[index], statistics) for split, index in indices.items()}
    labels = {split: y[index] for split, index in indices.items()}
    np.savez_compressed(output_dir / "split_indices.npz", **indices, test=np.arange(len(test_y)))
    write_json(output_dir / "data.json", {
        "dataset": "CIFAR-10", "reference_key": "Krizhevsky2009", "homepage": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "expected_archive_sha256": ARCHIVE_SHA256, "keras_loader_file_hash_verification": True,
        "statistics": statistics, "split_seed": config["split_seed"],
        "class_counts": {split: np.bincount(values, minlength=config["classes"]).tolist() for split, values in labels.items()},
        "unused_official_training_samples": len(raw) - len(joined),
        "selected_raw_rgb_sha256": {split: hashlib.sha256(raw[index].tobytes()).hexdigest() for split, index in indices.items()},
        "disjoint_index_splits": True, "test_source": "Official separate CIFAR-10 test batch, all 10000 images",
    })
    del raw
    records = []
    for seed in config["seeds"]:
        for encoding in [*config["encodings"], "RGB"]:
            print(f"Training {encoding}, seed {seed}", flush=True)
            folder = output_dir / f"seed_{seed}_{encoding.lower()}"
            record = train_one(config, encoding, seed, rgb, labels, folder)
            if record["status"] != "passed":
                raise RuntimeError(f"Training integrity check failed: {folder}")
            records.append(record)

    # All model/epoch choices are now frozen. Test data is first normalized and evaluated here.
    print("All training finished; starting the final held-out test evaluation.", flush=True)
    test_rgb = normalize_rgb(test_raw, statistics)
    for record in records:
        folder = output_dir / f"seed_{record['seed']}_{record['encoding'].lower()}"
        model = tf.keras.models.load_model(folder / "model.keras")
        inputs = encode_pixels(test_rgb, record["encoding"])
        logits = batched_output(model, inputs)
        losses = tf.keras.losses.sparse_categorical_crossentropy(test_y, logits, from_logits=True).numpy()
        record["test"] = {"cross_entropy": float(losses.mean()), "accuracy": float(np.mean(logits.argmax(axis=1) == test_y))}
        record["test_evaluated"] = True
        np.savez_compressed(folder / "test_predictions.npz", logits=logits, labels=test_y)
        write_json(folder / "metrics.json", record)
        print(f"Final test {record['encoding']} seed={record['seed']}: {record['test']['accuracy']:.4f}", flush=True)
        del model, inputs
        gc.collect()

    from src.evaluation.rgb_report import save_pilot_report
    summary = save_pilot_report(records, config, output_dir)
    summary["environment"] = environment(config["seeds"])
    summary["environment"]["tensorflow_threads"] = {"intra_op": config["intra_op_threads"], "inter_op": config["inter_op_threads"]}
    summary["source_hashes"] = {name: source_hashes(ROOT / name) for name in (
        "scripts/rgb_pilot.py", "scripts/runtime.py", "src/data/rgb.py", "src/models/rgb_baselines.py",
        "src/models/quaternion_dense.py", "src/activations/similarity.py", "src/activations/diagnostics.py", "src/evaluation/rgb_report.py",
    )}
    summary["split_indices_sha256"] = source_hashes(output_dir / "split_indices.npz")["sha256"]
    summary["status"] = "passed"
    write_json(output_dir / "summary.json", summary)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=True)
    (output_dir / "environment.freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    print(f"RGB pilot complete: {output_dir}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "rgb_pilot.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "rgb_pilot")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["epochs"] < 1 or len(set(config["seeds"])) != len(config["seeds"]) or not config["seeds"]:
        parser.error("Use positive epochs and distinct nonempty seeds")
    if not config["encodings"] or any(e not in ("ARGB", "ABGR") for e in config["encodings"]) or len(set(config["encodings"])) != len(config["encodings"]):
        parser.error("Choose distinct quaternion encodings ARGB and/or ABGR")
    run(config, args.output_dir)


if __name__ == "__main__":
    main()
