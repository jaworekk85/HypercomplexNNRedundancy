"""Audit the unmodified installed package; a failed gate returns exit code 1."""

import argparse
import importlib.metadata as metadata
import inspect

from scripts.runtime import (
    ROOT, configure, environment, np, real_parameter_count, source_hashes, tf, write_json,
)


UPSTREAM = {
    "repository": "https://github.com/rkycia/HypercomplexKeras",
    "commit": "566f887a7039637860089863b5abfa13b1948267",
    "hyperdense_git_blob_sha1": "d076199079add76fd69badcd64feb96ef3f6c316",
    "khnn_repository": "https://github.com/rkycia/KHNN",
    "khnn_commit": "d2c532dbca8fe6c18355dbbc1f5749e7e0e4af32",
    "softwarex_repository": "https://github.com/ElsevierSoftwareX/SOFTX-D-25-00004",
    "softwarex_commit": "d2c532dbca8fe6c18355dbbc1f5749e7e0e4af32",
    "license": "Apache-2.0",
    "citation_key": "NiemczynowiczKycia2025",
}


def run_audit(seed=2026):
    configure(seed)
    from HypercomplexKeras.Hyperdense import HyperDense

    inputs = tf.random.normal((16, 8))  # two quaternions; component-blocked layout
    targets = tf.random.normal((16, 12))
    layer = HyperDense(3)
    with tf.GradientTape(persistent=True) as tape:
        outputs = layer(inputs)
        loss = tf.reduce_mean(tf.square(outputs - targets))
    variables = layer.trainable_variables
    gradients = tape.gradient(loss, variables)
    kernel_gradient = tape.gradient(loss, layer.kernel)
    del tape
    kernel_before = layer.kernel.numpy().copy()
    bias_before = layer.bias.numpy().copy()
    tf.keras.optimizers.SGD(learning_rate=0.01).apply_gradients(zip(gradients, variables))
    hashes = source_hashes(inspect.getfile(HyperDense))
    checks = {
        "output_shape": list(outputs.shape) == [16, 12],
        "all_expected_parameters_tracked": real_parameter_count(variables) == 36,
        "kernel_is_tracked": any(v is layer.kernel for v in variables),
        "kernel_has_finite_gradient": kernel_gradient is not None and bool(np.isfinite(kernel_gradient.numpy()).all()),
        "tracked_variables_have_finite_gradients": all(g is not None and bool(np.isfinite(g.numpy()).all()) for g in gradients),
        "kernel_changed_after_standard_optimizer_step": bool(np.any(layer.kernel.numpy() != kernel_before)),
        "bias_changed_after_standard_optimizer_step": bool(np.any(layer.bias.numpy() != bias_before)),
        "installed_source_matches_inspected_commit": hashes["git_blob_sha1"] == UPSTREAM["hyperdense_git_blob_sha1"],
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "environment": environment(seed),
        "upstream": UPSTREAM,
        "installed_hyperdense_hashes": hashes,
        "declared_runtime_dependencies": metadata.requires("HypercomplexKeras"),
        "input_shape": list(inputs.shape),
        "output_shape": list(outputs.shape),
        "expected_real_trainable_parameters": 36,
        "reported_real_trainable_parameters": real_parameter_count(variables),
        "trainable_variables": [{"name": v.name, "shape": list(v.shape)} for v in variables],
        "loss_before_step": float(loss.numpy()),
        "kernel_gradient_norm": None if kernel_gradient is None else float(tf.linalg.global_norm([kernel_gradient]).numpy()),
        "kernel_max_abs_update": float(np.max(np.abs(layer.kernel.numpy() - kernel_before))),
        "bias_max_abs_update": float(np.max(np.abs(layer.bias.numpy() - bias_before))),
        "default_kernel_components_identical": bool(np.all(kernel_before == kernel_before[0:1])),
        "checks": checks,
        "scope": "Unmodified HypercomplexKeras 1.0.2 on this environment only; no claim about all TensorFlow versions.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=str, default=str(ROOT / "results" / "phase0" / "upstream_smoke.json"))
    args = parser.parse_args()
    report = run_audit(args.seed)
    write_json(args.output, report)
    print(f"Upstream gate: {report['status']}; tracked {report['reported_real_trainable_parameters']}/36 real parameters")
    print(f"Kernel maximum update: {report['kernel_max_abs_update']}")
    print(f"Report: {args.output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
