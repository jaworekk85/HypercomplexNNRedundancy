"""Bounded BN-only controls and read-only recovery analysis for paper A.

Existing numerical sources are imported, never modified. The BN protocol freezes
sources, original checkpoints and split/phase-A records before measuring outcomes.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import time

import numpy as np
import torch
from torch import nn

from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from scripts.run_activation_queue import lock
from src.cnn_training import evaluate, prepare_batch, seed_everything
from src.compression.multilayer import compact_site, sites, units
from src.data.cifar_cached import file_sha256
from src.evaluation.routing_hardening import recalibrate_bn_layers

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/paper_a_bn.json"
CURVES = ["results/compression_curve", "results/compression_curve_refinement",
          "results/compression_curve_eurosat", "results/compression_curve_eurosat_refinement"]


def now():
    return datetime.now(timezone.utc).isoformat()


def rel(path):
    return str(Path(path).relative_to(ROOT)).replace("\\", "/")


def verify_registered_sources():
    """Check every extant source manifest used by the completed main queues."""
    paths = ["results/compression_curve/input_manifest.json",
             "results/compression_curve_queue_protocol.json",
             "results/activation_sensitivity_queue_protocol.json",
             "results/pseudolabel_repair_queue_protocol.json",
             "results/structure_control_queue_protocol.json",
             "results/ssl_compression_pilot_runner_protocol.json"]
    checked = {}
    for name in paths:
        manifest = curve.read(ROOT / name)
        hashes = manifest.get("source_hashes", manifest.get("hashes", {}))
        if not hashes:
            raise ValueError(f"Empty registered source manifest: {name}")
        for source, expected in hashes.items():
            if file_sha256(ROOT / source) != expected:
                raise ValueError(f"Frozen source changed: {source} ({name})")
            checked[source] = expected
    return checked


@torch.no_grad()
def cumulative_bn(model, images, mean, std, batch_size):
    """One unlabeled pass, only BN train mode, unbiased per-batch variances."""
    if len(images) == 0 or len(images) % batch_size:
        raise ValueError("Use a nonempty whole number of equal batches")
    model.eval()
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
    if not layers or any(not m.track_running_stats for _, m in layers):
        raise ValueError("Tracked BN layers required")
    momenta = [m.momentum for _, m in layers]
    try:
        for _, layer in layers:
            layer.reset_running_stats()
            layer.momentum = None
            layer.train()
        for start in range(0, len(images), batch_size):
            model(prepare_batch(images[start:start + batch_size], mean, std))
    finally:
        for (_, layer), momentum in zip(layers, momenta):
            layer.momentum = momentum
        model.eval()
    allowed = {f"{name}.{suffix}" for name, _ in layers
               for suffix in ("running_mean", "running_var", "num_batches_tracked")}
    checks = {
        "parameters_bitwise_unchanged": all(torch.equal(params[n], p) for n, p in model.named_parameters()),
        "other_buffers_bitwise_unchanged": all(n in allowed or torch.equal(buffers[n], b) for n, b in model.named_buffers()),
        "all_modules_in_evaluation_mode": all(not m.training for m in model.modules()),
        "no_parameter_gradients": all(p.grad is None for p in model.parameters()),
        "expected_batch_counters": all(int(m.num_batches_tracked) == len(images) // batch_size for _, m in layers),
        "finite_statistics": all(bool(torch.isfinite(m.running_mean).all() and torch.isfinite(m.running_var).all()) for _, m in layers),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"checks": checks, "images": len(images), "labels_used": False,
            "estimator": "Cumulative mean of equal-batch BN means and unbiased within-batch variances; simultaneous train-mode BNs",
            "full_forward_passes": 1, "bn_layers": len(layers)}


def initial_model(study, source, fraction, policy, device):
    parent = ROOT / study["output"]
    path = parent / "compact" / source["architecture"] / f"seed_{source['seed']}" / f"f_{round(1000*fraction):03d}" / f"{policy}.pt"
    model = curve.factory(study["dataset"], source["architecture"])(
        curve.reduced_config(source["config"], source["architecture"], fraction), curve.kind_of(source["architecture"])).to(device).eval()
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model"])
        return model, {"checkpoint": rel(path), "sha256": file_sha256(path)}
    if policy != "greedy_pruned":
        raise FileNotFoundError(path)
    model = curve.load_source(study, source, device)
    phase = parent / "phase_a" / f"{source['architecture']}_seed_{source['seed']}.json"
    records = [r for r in curve.read(phase)["records"] if r["fraction"] == fraction and r["case"] == policy]
    if len(records) != 1:
        raise ValueError("Missing/duplicate frozen victim record")
    if [tuple(row["site"]) for row in records[0]["sites"]] != sites(model):
        raise ValueError("Unexpected site order")
    for record in records[0]["sites"]:
        site, victims = tuple(record["site"]), set(record["victims"])
        model = compact_site(model, site, [j for j in range(units(model, site)) if j not in victims])
    return model.eval(), {"reconstructed_same_victims": True, "phase_a": rel(phase), "phase_a_sha256": file_sha256(phase)}


def freeze_bn_protocol(config):
    checked = verify_registered_sources()
    names = set(checked) | {CONFIG, "scripts/paper_a_controls.py", "src/evaluation/routing_hardening.py"}
    inputs = {}
    for cfg in config["curve_configs"]:
        study = curve.read(ROOT / cfg)
        names.add(cfg)
        parent = ROOT / study["output"]
        for arch in config["architectures"]:
            for seed in config["seeds"]:
                phase = parent / "phase_a" / f"{arch}_seed_{seed}.json"
                inputs[rel(phase)] = file_sha256(phase)
                for fraction in [0.] + config["fractions"]:
                    folder = parent / "compact" / arch / f"seed_{seed}" / f"f_{round(1000*fraction):03d}"
                    for policy in (["baseline"] if fraction == 0 else config["policies"]):
                        path = folder / f"{policy}.pt"
                        if path.exists():
                            inputs[rel(path)] = file_sha256(path)
        # Original checkpoints, split files and first-update audits underpin reconstructed models.
        if study["dataset"] == "cifar":
            sources = curve.sources_of(curve.read(ROOT / "configs/multisource_compensation.json"))
        else:
            sources = [{"path": r["path"], "seed": r["seed"], "architecture": "cnn6"}
                       for r in curve.read(ROOT / study["training_output"] / "summary.json")["runs"]]
        for source in sources:
            if source["seed"] not in config["seeds"] or source["architecture"] not in config["architectures"]:
                continue
            folder = ROOT / source["path"]
            kind = curve.kind_of(source["architecture"])
            checks = curve.read(folder / kind / "metrics.json")["first_update"]["checks"]
            if not all(checks.values()):
                raise AssertionError(f"Quaternion training audit failed: {source}")
            for path in [folder / kind / "model.pt", folder / kind / "metrics.json", folder / "summary.json", folder / "split_indices.npz"]:
                inputs[rel(path)] = file_sha256(path)
    definition = {"config": config, "source_hashes": {n: file_sha256(ROOT / n) for n in sorted(names)}, "input_hashes": inputs}
    path = ROOT / config["output"] / "protocol.json"
    if path.exists() and curve.read(path) != definition:
        raise ValueError("BN sources/config/inputs changed; use a new output")
    write_json(path, definition)
    return definition


def run_bn():
    config = curve.read(ROOT / CONFIG)
    output = ROOT / config["output"]
    with lock("compression_curve_runner.lock"):
        freeze_bn_protocol(config)
        torch.set_num_threads(config["cpu_threads"])
        device = torch.device(config["device"])
        rows = []
        write_json(output / "state.json", {"status": "running", "started_utc": now()})
        for cfg in config["curve_configs"]:
            study = curve.read(ROOT / cfg)
            study = {**study, "architectures": config["architectures"], "seeds": config["seeds"]}
            images, targets, mean, std, sources = curve.load_data(study, device)
            for source in sources:
                seed_everything(source["seed"])
                ids = curve.split_ids(source, None)
                selected = np.random.default_rng(config["order_seed_offset"] + source["seed"]).permutation(ids["train"])[:config["training_images"]]
                if len(selected) != config["training_images"] or set(selected) & set(np.concatenate([ids["validation"], ids["calibration"]])):
                    raise AssertionError("BN calibration split overlap")
                stem = output / study["dataset"] / source["architecture"] / f"seed_{source['seed']}"
                stem.mkdir(parents=True, exist_ok=True)
                np.save(stem / "bn_training_indices.npy", selected)
                calibration = torch.as_tensor(images[selected], device=device)
                for fraction in [0.] + config["fractions"]:
                    for policy in (["baseline"] if fraction == 0 else config["policies"]):
                        folder = stem / f"f_{round(1000*fraction):03d}" / policy
                        path = folder / "metrics.json"
                        if path.exists():
                            rows.append(curve.read(path))
                            continue
                        started = time.perf_counter()
                        model, provenance = initial_model(study, source, fraction, policy, device)
                        before = curve.metrics(model, images, targets, ids, mean, std, study, device)
                        phase = curve.read(ROOT / study["output"] / "phase_a" / f"{source['architecture']}_seed_{source['seed']}.json")
                        original = next(r for r in phase["rows"] if r["fraction"] == fraction and r["case"] == policy)
                        if any(abs(before[s]["accuracy"] - original["evaluation"][s]["accuracy"]) > 1e-9 or
                               abs(before[s]["cross_entropy"] - original["evaluation"][s]["cross_entropy"]) > 1e-4
                               for s in ("validation", "check_calibration")):
                            raise AssertionError("Initial model does not reproduce original phase-A metrics")
                        result = {"dataset": study["dataset"], "architecture": source["architecture"], "seed": source["seed"],
                                  "fraction": fraction, "policy": policy, "before": before,
                                  "real_parameters": sum(p.numel() for p in model.parameters()), "initial_provenance": provenance,
                                  "training_images": len(selected), "bn_training_indices_sha256": file_sha256(stem / "bn_training_indices.npy"),
                                  "official_test_evaluated": False, "methods": {}}
                        folder.mkdir(parents=True, exist_ok=True)
                        torch.save({"model": model.state_dict()}, folder / "initial.pt")
                        for method in config["methods"]:
                            tuned = deepcopy(model)
                            torch.cuda.synchronize() if device.type == "cuda" else None
                            start = time.perf_counter()
                            if method == "cumulative":
                                audit = cumulative_bn(tuned, calibration, mean, std, config["batch_size"])
                            else:
                                names = [n for n, m in tuned.named_modules() if isinstance(m, nn.BatchNorm2d)]
                                audit = recalibrate_bn_layers(tuned, calibration, mean, std, names, config["batch_size"])
                                audit["full_forward_passes"] = len(names)
                            torch.cuda.synchronize() if device.type == "cuda" else None
                            elapsed = time.perf_counter() - start
                            evaluation = curve.metrics(tuned, images, targets, ids, mean, std, study, device)
                            checkpoint = folder / f"{method}.pt"
                            torch.save({"model": {n: v.detach().cpu() for n, v in tuned.state_dict().items()}}, checkpoint)
                            result["methods"][method] = {"evaluation": evaluation, "audit": audit, "bn_seconds": elapsed,
                                                         "checkpoint": rel(checkpoint), "sha256": file_sha256(checkpoint)}
                        result["case_seconds"] = time.perf_counter() - started
                        write_json(path, result)
                        rows.append(result)
                        summarize_bn(rows, output, "running")
                        print(f"{study['dataset']} seed {source['seed']} f={fraction} {policy}: " + ", ".join(
                            f"{m} {100*r['evaluation']['validation']['accuracy']:.2f}%" for m, r in result["methods"].items()), flush=True)
            del images, calibration
        freeze_bn_protocol(config)  # detect any source/input drift during the run
        summarize_bn(rows, output, "completed")
        write_json(output / "state.json", {"status": "completed", "completed_utc": now(), "cases": len(rows), "official_test_evaluated": False})


def summarize_bn(rows, output, status):
    aggregates = []
    for dataset, fraction, policy in sorted({(r["dataset"], r["fraction"], r["policy"]) for r in rows}):
        group = [r for r in rows if (r["dataset"], r["fraction"], r["policy"]) == (dataset, fraction, policy)]
        for method in ["before", "cumulative", "layerwise_global"]:
            values = [{"seed": r["seed"], "accuracy": (r["before"] if method == "before" else r["methods"][method]["evaluation"])["validation"]["accuracy"]} for r in group]
            aggregates.append({"dataset": dataset, "fraction": fraction, "policy": policy, "method": method, "seed_values": values,
                               "mean_accuracy": float(np.mean([r["accuracy"] for r in values])),
                               "sample_sd_accuracy": float(np.std([r["accuracy"] for r in values], ddof=1)) if len(values) > 1 else None})
    write_json(output / "summary.json", {"status": status, "cases": rows, "aggregates": aggregates, "official_test_evaluated": False})
    lines = ["# Paper A: BN-only control", "", f"Status: {status}. Validation results; test not evaluated.", "",
             "2048 unaugmented training images, identical within seed across policies. No labels/gradients/weight updates.",
             "Cumulative: one forward pass, simultaneous train-mode BNs. Global: six passes, sequential eval-mode moments.",
             "These differ in estimator and cost; report both. This is not a reproduction of EagleEye or REPAIR.", "",
             "| Dataset | Removed/site | Policy | Repair | n seeds | Val accuracy mean (%) | SD (pp) |",
             "| --- | ---: | --- | --- | ---: | ---: | ---: |"]
    for row in aggregates:
        sd = "—" if row["sample_sd_accuracy"] is None else f"{100*row['sample_sd_accuracy']:.2f}"
        lines.append(f"| {row['dataset']} | {row['fraction']} | {row['policy']} | {row['method']} | {len(row['seed_values'])} | {100*row['mean_accuracy']:.2f} | {sd} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_recovery():
    """Extract all completed histories; no model inference or optimizer calls."""
    output = ROOT / "results/paper_a_evidence"
    source_hashes, rows, seen = {}, [], {}
    for parent_name in CURVES:
        parent = ROOT / parent_name
        summary = curve.read(parent / "summary.json")
        dataset = "eurosat" if "eurosat" in parent_name else "cifar"
        source_hashes[rel(parent / "summary.json")] = file_sha256(parent / "summary.json")
        for phase in sorted((parent / "phase_a").glob("*.json")):
            source_hashes[rel(phase)] = file_sha256(phase)
            for row in curve.read(phase)["rows"]:
                key = (dataset, row["architecture"], row["seed"], row["fraction"], row["case"], 0)
                metrics = row["evaluation"]
                if key in seen:
                    if seen[key] != metrics:
                        raise ValueError("Conflicting duplicate baseline")
                    continue
                seen[key] = metrics
                rows.append({"dataset": dataset, "architecture": row["architecture"], "seed": row["seed"],
                             "fraction": row["fraction"], "case": row["case"], "policy": row["policy"], "epoch": 0,
                             "evaluation": metrics, "real_parameters": row["real_parameters"], "source": rel(phase)})
        for path in sorted((parent / "finetuned").rglob("metrics.json")):
            record = curve.read(path)
            source_hashes[rel(path)] = file_sha256(path)
            for entry in record["history"]:
                key = (dataset, record["architecture"], record["seed"], record["fraction"], record["case"], entry["epoch"])
                metrics = {s: entry[s] for s in ("validation", "check_calibration")}
                if key in seen:
                    if seen[key] != metrics:
                        raise ValueError("Conflicting duplicate recovery history")
                    continue
                seen[key] = metrics
                rows.append({"dataset": dataset, "architecture": record["architecture"], "seed": record["seed"],
                             "fraction": record["fraction"], "case": record["case"], "policy": record["policy"],
                             "epoch": entry["epoch"], "evaluation": metrics, "real_parameters": record["real_parameters"],
                             "epoch_seconds": entry["epoch_seconds"], "source": rel(path)})
    # Same-victim QNN continuations were measured later in the circulant control.
    supplement = ROOT / "results/structure_control/quaternion_paired_repair/finetuned"
    for path in sorted(supplement.rglob("metrics.json")):
        record = curve.read(path)
        source_hashes[rel(path)] = file_sha256(path)
        for entry in record["history"]:
            rows.append({"dataset": "cifar", "architecture": "cnn6", "seed": record["seed"],
                         "fraction": record["fraction"], "case": record["case"], "policy": record["policy"],
                         "epoch": entry["epoch"], "evaluation": {s: entry[s] for s in ("validation", "check_calibration")},
                         "real_parameters": record["real_parameters"], "epoch_seconds": entry["epoch_seconds"], "source": rel(path)})
    aggregates = []
    keys = sorted({(r["dataset"], r["architecture"], r["fraction"], r["policy"], r["epoch"]) for r in rows})
    for key in keys:
        group = [r for r in rows if (r["dataset"], r["architecture"], r["fraction"], r["policy"], r["epoch"]) == key]
        values = [{"seed": seed, "accuracy": float(np.mean([r["evaluation"]["validation"]["accuracy"] for r in group if r["seed"] == seed]))}
                  for seed in sorted({r["seed"] for r in group})]
        aggregates.append(dict(zip(("dataset", "architecture", "fraction", "policy", "epoch"), key), seed_values=values,
                               mean_accuracy=float(np.mean([r["accuracy"] for r in values])),
                               sample_sd_accuracy=float(np.std([r["accuracy"] for r in values], ddof=1))))
    write_json(output / "recovery.json", {"status": "completed", "source_hashes": source_hashes, "rows": rows, "aggregates": aggregates,
                                         "test_evaluated": False, "new_training": False,
                                         "budget_caveat": "Equal epochs/attempted updates, not audited equal successful AMP steps. Epoch time excludes teacher training and compression."})
    lookup = {(r["dataset"], r["architecture"], r["fraction"], r["policy"], r["epoch"]): r for r in aggregates}
    lines = ["# Paper A: saved recovery histories", "", "No new training or test inference. Means of seed means; random replicates average within seed.",
             "Equal epochs/attempted updates do not establish equal successful AMP updates or runtime savings.", "",
             "| Dataset | Removed/site | Policy | Before | Epoch 1 | Epoch 2 | Epoch 3 |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for dataset in ("cifar", "eurosat"):
        for fraction in (.25, .5, .75):
            for policy in ("greedy_comp", "greedy_pruned", "bn_pruned"):
                values = [lookup.get((dataset, "cnn6", fraction, policy, epoch)) for epoch in range(4)]
                lines.append(f"| {dataset} | {fraction} | {policy} | " + " | ".join("not measured" if r is None else f"{100*r['mean_accuracy']:.2f}" for r in values) + " |")
    (output / "RECOVERY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.8), sharex=True)
    for index, dataset in enumerate(("cifar", "eurosat")):
        for column, fraction in enumerate((.25, .5, .75)):
            ax = axes[index, column]
            for policy, label in [("greedy_comp", "Compensated"), ("greedy_pruned", "Same victims, pruned"), ("bn_pruned", "BN magnitude")]:
                values = [(e, lookup[(dataset, "cnn6", fraction, policy, e)]) for e in range(4) if (dataset, "cnn6", fraction, policy, e) in lookup]
                ax.plot([e for e, r in values], [100*r["mean_accuracy"] for e, r in values], marker="o", label=label)
            ax.set_title(f"{dataset.upper()}, {fraction:.0%} units removed/site")
            ax.set_xticks(range(4))
            ax.grid(alpha=.25)
            if column == 0: ax.set_ylabel("Validation accuracy (%)")
            if index == 1: ax.set_xlabel("Repair epochs")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"recovery.{suffix}", dpi=180)
    plt.close(fig)
    print(f"Extracted {len(rows)} recovery observations from {len(source_hashes)} artifacts.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("bn", "analyze"):
        raise SystemExit("Usage: python -m scripts.paper_a_controls bn|analyze")
    if sys.argv[1] == "bn":
        run_bn()
    else:
        analyze_recovery()
