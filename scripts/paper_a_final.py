"""Frozen, resumable final held-out evaluation. No optimizer or model selection."""
from pathlib import Path
import sys

import numpy as np
import torch

from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from scripts.eurosat_cnn import load_eurosat
from scripts.paper_a_controls import ROOT, now, rel, verify_registered_sources
from scripts.run_activation_queue import lock
from src.cnn_training import evaluate, seed_everything
from src.data.cifar_cached import file_sha256, load_cached_cifar
from src.models.circulant_cnn import make_circulant

CONFIG = "configs/paper_a_final.json"


def sources(study):
    if study["dataset"] == "cifar":
        return curve.sources_of(curve.read(ROOT / "configs/multisource_compensation.json"))
    return [{"architecture": arch, "seed": r["seed"], "path": r["path"],
             "config": curve.read(ROOT / r["path"] / "summary.json")["config"]}
            for r in curve.read(ROOT / study["training_output"] / "summary.json")["runs"]
            for arch in ("cnn6", "real_cnn6")]


def cases(config):
    rows = []

    def add(dataset, source, fraction, policy, stage, checkpoint, expected, family=None):
        family = family or curve.kind_of(source["architecture"])
        identifier = f"{dataset}/{family}/seed_{source['seed']}/f_{round(1000*fraction):03d}/{policy}_{stage}"
        if any(r["id"] == identifier for r in rows):
            raise ValueError(f"Duplicate final case {identifier}")
        rows.append({"id": identifier, "dataset": dataset, "family": family, "architecture": source["architecture"],
                     "seed": source["seed"], "fraction": fraction, "policy": policy, "stage": stage,
                     "checkpoint": rel(checkpoint), "config": curve.reduced_config(source["config"], source["architecture"], fraction),
                     "expected_validation": expected})

    for cfg in config["curve_configs"]:
        study = curve.read(ROOT / cfg)
        parent = ROOT / study["output"]
        summary = curve.read(parent / "summary.json")
        if summary["status"] != "completed":
            raise ValueError("Incomplete parent curve")
        for source in sources(study):
            if source["architecture"] not in config["architectures"] or source["seed"] not in config["seeds"]:
                continue
            arch, seed = source["architecture"], source["seed"]
            relevant = [r for r in summary["rows"] if r["architecture"] == arch and r["seed"] == seed]
            for r in relevant:
                f, p, s = r["fraction"], r["policy"], r["stage"]
                if (f == 0 and p == "baseline" and s == "before"):
                    checkpoint = parent / "compact" / arch / f"seed_{seed}/f_000/baseline.pt"
                elif f in config["fractions"] and p in config["policies"] and s in ("before", "after"):
                    base = parent / ("compact" if s == "before" else "finetuned") / arch / f"seed_{seed}" / f"f_{round(1000*f):03d}"
                    checkpoint = base / (f"{p}.pt" if s == "before" else f"{p}/model.pt")
                elif f in config["fractions"] and p == "scratch" and s == "scratch":
                    checkpoint = parent / "scratch" / arch / f"seed_{seed}" / f"f_{round(1000*f):03d}" / curve.kind_of(arch) / "model.pt"
                else:
                    continue
                add(study["dataset"], source, f, p, s, checkpoint, r["evaluation"]["validation"])
    # BN-only: include every estimator and original state, without test-driven selection.
    bn = curve.read(ROOT / "results/paper_a_bn/summary.json")
    if bn["status"] != "completed" or len(bn["cases"]) != 42:
        raise ValueError("Wait for all 42 BN cases before freezing test evaluation")
    for r in bn["cases"]:
        study = curve.read(ROOT / ("configs/compression_curve.json" if r["dataset"] == "cifar" else "configs/compression_curve_eurosat.json"))
        source = next(s for s in sources(study) if s["architecture"] == r["architecture"] and s["seed"] == r["seed"])
        folder = ROOT / "results/paper_a_bn" / r["dataset"] / r["architecture"] / f"seed_{r['seed']}" / f"f_{round(1000*r['fraction']):03d}" / r["policy"]
        # Other initial states are already in the primary selection.
        if r["policy"] == "greedy_pruned":
            add(r["dataset"], source, r["fraction"], r["policy"], "before", folder / "initial.pt", r["before"]["validation"])
        for method, result in r["methods"].items():
            add(r["dataset"], source, r["fraction"], r["policy"], f"bn_{method}", ROOT / result["checkpoint"], result["evaluation"]["validation"])
    # Existing paired-weight repair and circulant control, all recorded policies.
    study = curve.read(ROOT / "configs/compression_curve.json")
    cifar_sources = sources(study)
    for path in sorted((ROOT / "results/structure_control/quaternion_paired_repair/finetuned").rglob("metrics.json")):
        r = curve.read(path)
        source = next(s for s in cifar_sources if s["architecture"] == "cnn6" and s["seed"] == r["seed"])
        add("cifar", source, r["fraction"], r["policy"], "after", path.parent / "model.pt", r["final"]["validation"])
    circ = ROOT / "results/structure_control"
    summary = curve.read(circ / "summary.json")
    for seed in config["seeds"]:
        source = {"architecture": "circulant_cnn6", "seed": seed,
                  "config": curve.read(circ / f"baselines/seed_{seed}/summary.json")["config"]}
        for r in summary["rows"]:
            if r["architecture"] != "circulant_cnn6" or r["seed"] != seed:
                continue
            f, p, s = r["fraction"], r["policy"], r["stage"]
            if f == 0 and p == "baseline" and s == "before":
                checkpoint = circ / f"baselines/seed_{seed}/circulant/model.pt"
            elif f in (.25, .5) and p in ("greedy_comp", "greedy_pruned", "bn_pruned") and s in ("before", "after"):
                base = circ / "circulant" / ("compact" if s == "before" else "finetuned") / "circulant_cnn6" / f"seed_{seed}" / f"f_{round(1000*f):03d}"
                checkpoint = base / (f"{p}.pt" if s == "before" else f"{p}/model.pt")
            else:
                continue
            add("cifar", source, f, p, s, checkpoint, r["evaluation"]["validation"], "circulant")
    return sorted(rows, key=lambda r: r["id"])


def freeze(config, rows):
    names = set(verify_registered_sources()) | {CONFIG, "scripts/paper_a_final.py", "scripts/paper_a_controls.py"}
    data = [".cache/keras/datasets/cifar-10-batches-py-target_archive", ".cache/eurosat/prepared/eurosat_nrgb.npz",
            ".cache/eurosat/prepared/split_indices.npz", ".cache/eurosat/prepared/data.json",
            "results/cifar_cnn/data.json", "results/paper_a_bn/protocol.json", "results/paper_a_bn/summary.json"]
    definition = {"config": config, "cases": rows, "source_hashes": {n: file_sha256(ROOT / n) for n in sorted(names)},
                  "checkpoint_hashes": {r["checkpoint"]: file_sha256(ROOT / r["checkpoint"]) for r in rows},
                  "data_hashes": {n: file_sha256(ROOT / n) for n in data},
                  "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda}}
    path = ROOT / config["output"] / "protocol.json"
    if path.exists() and curve.read(path) != definition:
        raise ValueError("Final frozen inputs or sources changed; do not silently reevaluate")
    write_json(path, definition)
    return definition


def load_test(dataset):
    if dataset == "cifar":
        _, (images, targets) = load_cached_cifar(ROOT / ".cache/keras/datasets/cifar-10-batches-py-target_archive")
        stats = curve.read(ROOT / "results/cifar_cnn/data.json")["statistics"]
        # Validation remains available only for numerical restoration verification.
        train, _ = load_cached_cifar(ROOT / ".cache/keras/datasets/cifar-10-batches-py-target_archive")
        return images, targets, stats, train
    images, targets, split, meta = load_eurosat(ROOT / ".cache/eurosat/prepared")
    if file_sha256(ROOT / ".cache/eurosat/prepared/eurosat_nrgb.npz") != meta["arrays_sha256"] or file_sha256(ROOT / ".cache/eurosat/prepared/split_indices.npz") != meta["split_sha256"]:
        raise ValueError("EuroSAT prepared-data checksum mismatch")
    if set(split["test"]) & set(np.concatenate([split[k] for k in ("train", "validation", "calibration")])):
        raise ValueError("Test split overlap")
    return images[split["test"]], targets[split["test"]], meta["statistics"], (images, targets)


def report(rows, output, status):
    groups = []
    for key in sorted({(r["dataset"], r["family"], r["fraction"], r["policy"], r["stage"]) for r in rows}):
        group = [r for r in rows if (r["dataset"], r["family"], r["fraction"], r["policy"], r["stage"]) == key]
        groups.append(dict(zip(("dataset", "family", "fraction", "policy", "stage"), key),
                           seed_values=[{"seed": r["seed"], **r["test"]} for r in group],
                           mean_accuracy=float(np.mean([r["test"]["accuracy"] for r in group])),
                           sample_sd_accuracy=float(np.std([r["test"]["accuracy"] for r in group], ddof=1)) if len(group) > 1 else None))
    effects = []
    index = {(r["dataset"], r["family"], r["fraction"], r["policy"], r["stage"]): r for r in groups}
    for key, a in index.items():
        dataset, family, fraction, policy, stage = key
        if policy != "greedy_comp": continue
        for reference in ("greedy_pruned", "bn_pruned"):
            b = index.get((dataset, family, fraction, reference, stage))
            if b is None: continue
            av, bv = ({v["seed"]: v["accuracy"] for v in r["seed_values"]} for r in (a, b))
            if set(av) != set(bv): continue
            differences = [{"seed": seed, "accuracy_difference_pp": 100*(av[seed]-bv[seed])} for seed in sorted(av)]
            effects.append({"dataset": dataset, "family": family, "fraction": fraction, "stage": stage,
                            "comparison": f"greedy_comp - {reference}", "paired_seeds": differences,
                            "mean_difference_pp": float(np.mean([v["accuracy_difference_pp"] for v in differences]))})
    write_json(output / "summary.json", {"status": status, "rows": rows, "aggregates": groups, "paired_effects": effects,
                                         "test_evaluated": True, "cifar_historically_seen": True, "eurosat_final_holdout": True})
    lines = ["# Paper A: frozen final test evaluation", "", f"Status: {status}.",
             "EuroSAT: final held-out test (5400 tiles); random tile split, not geographic transfer. CIFAR: official 10000-image test, historically seen in early pilots.",
             "No post-test tuning or selection. Prespecified subset .25/.5/.75; seven-point curves remain validation evidence.",
             "BN estimators use 2048 unlabeled training images; fine-tuning uses 3 labeled epochs; scratch uses 50 epochs plus validation selection.", "",
             "| Dataset | Family | Removed/site | Policy | Stage | n seeds | Test mean (%) | SD (pp) |", "| --- | --- | ---: | --- | --- | ---: | ---: | ---: |"]
    for r in groups:
        sd = "—" if r["sample_sd_accuracy"] is None else f"{100*r['sample_sd_accuracy']:.2f}"
        lines.append(f"| {r['dataset']} | {r['family']} | {r['fraction']} | {r['policy']} | {r['stage']} | {len(r['seed_values'])} | {100*r['mean_accuracy']:.2f} | {sd} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    config = curve.read(ROOT / CONFIG)
    output = ROOT / config["output"]
    with lock("compression_curve_runner.lock"):
        selected = cases(config)
        protocol = freeze(config, selected)
        torch.set_num_threads(config["cpu_threads"])
        device = torch.device(config["device"])
        seed_everything(20261105)
        write_json(output / "state.json", {"status": "running", "started_utc": now(), "planned_cases": len(selected)})
        rows = []
        for dataset in ("cifar", "eurosat"):
            images, targets, stats, (validation_images, validation_targets) = load_test(dataset)
            x, y = torch.as_tensor(images, device=device), torch.as_tensor(targets, device=device)
            channels = len(stats["mean"])
            mean = torch.tensor(stats["mean"], device=device).reshape(1, channels, 1, 1)
            std = torch.tensor(stats["std"], device=device).reshape(1, channels, 1, 1)
            study = curve.read(ROOT / ("configs/compression_curve.json" if dataset == "cifar" else "configs/compression_curve_eurosat.json"))
            dataset_sources = sources(study)
            for case in (r for r in selected if r["dataset"] == dataset):
                folder = output / "predictions" / case["id"]
                path = folder / "metrics.json"
                if path.exists():
                    saved = curve.read(path)
                    if saved["checkpoint_sha256"] != protocol["checkpoint_hashes"][case["checkpoint"]] or file_sha256(folder / "logits.npz") != saved["predictions_sha256"]:
                        raise ValueError("Cached final prediction provenance mismatch")
                    rows.append(saved)
                    continue
                make = make_circulant if case["family"] == "circulant" else curve.factory(dataset, case["architecture"])
                model = make(case["config"], case["family"]).to(device).eval()
                model.load_state_dict(torch.load(ROOT / case["checkpoint"], map_location=device, weights_only=True)["model"])
                # Verify frozen restoration on validation before exposing this model to test.
                source = next(s for s in dataset_sources if s["seed"] == case["seed"] and s["architecture"] == ("cnn6" if case["family"] == "circulant" else case["architecture"]))
                ids = curve.split_ids(source, None)["validation"]
                validation, _ = evaluate(model, torch.as_tensor(validation_images[ids], device=device), torch.as_tensor(validation_targets[ids], device=device), mean, std, config["batch_size"])
                expected = case["expected_validation"]
                if validation["accuracy"] != expected["accuracy"] or abs(validation["cross_entropy"]-expected["cross_entropy"]) > 1e-4:
                    raise AssertionError(f"Validation restoration check failed before test: {case['id']}")
                metrics, logits = evaluate(model, x, y, mean, std, config["batch_size"])
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / "logits.npz", logits=logits, targets=targets)
                result = {**case, "test": metrics, "test_images": len(images), "verified_validation": validation,
                          "real_parameters": sum(p.numel() for p in model.parameters()),
                          "checkpoint_sha256": protocol["checkpoint_hashes"][case["checkpoint"]],
                          "predictions_sha256": file_sha256(folder / "logits.npz")}
                write_json(path, result)
                rows.append(result)
                report(rows, output, "running")
                print(f"{len(rows)}/{len(selected)} {case['id']}: test {100*metrics['accuracy']:.2f}%", flush=True)
            del x, y, validation_images, validation_targets, images
        freeze(config, selected)
        report(rows, output, "completed")
        write_json(output / "state.json", {"status": "completed", "completed_utc": now(), "cases": len(rows), "test_evaluated": True})


if __name__ == "__main__":
    run()
