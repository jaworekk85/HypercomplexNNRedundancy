"""Conditional one-seed physical compression of fixed and learned routing models."""

import argparse
import gc
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from scripts.cifar_cnn import write_json
from scripts.learned_grouping import load_export, make_model
from src.cnn_training import evaluate, prepare_batch, seed_everything
from src.compression.quaternion_channels import compact_block, removal_order
from src.data.cifar_cached import file_sha256, load_cached_cifar
from src.evaluation.block_ablation import capture_blocks, incoming_norms, predict_from_block
from src.evaluation.cnn_report import spatial_diagnostics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_case(directory, model_config, device):
    manifest = read(directory / "manifest.json")
    model = make_model(model_config, "identity")
    model = compact_block(model, manifest["block"]-1, manifest["kept_units"]).to(device).eval()
    model.load_state_dict(torch.load(directory / "model.pt", weights_only=True, map_location=device)["model"], strict=True)
    return model


def prepare_mode(mode, model, config, training, folder, val_x, val_y, mean, std, calibration_values, old):
    index = config["block"]-1
    _, matrices = spatial_diagnostics(calibration_values, True)
    matrix = matrices["grouped"]
    if mode == "identity":
        # Reuse the exact frozen calibration matrix, avoiding batch-rounding
        # differences from reconstructing the original control's ranking.
        matrix_path = ROOT / old["config"]["source_block_analysis"] / "quaternion" / f"block_{config['block']}" / "calibration_similarity.npz"
        with np.load(matrix_path) as previous:
            matrix = previous["trained_grouped"]
    norms = incoming_norms(model, index)
    saved_matrix = folder / f"{mode}_calibration_similarity.npz"
    if not saved_matrix.exists():
        np.savez_compressed(saved_matrix, grouped=matrix)
    elif not np.array_equal(np.load(saved_matrix)["grouped"], matrix, equal_nan=True):
        raise ValueError("Changed routed calibration matrix")
    probe = prepare_batch(val_x[:32], mean, std)
    features = capture_blocks(model, probe, 32)[index]
    with torch.no_grad():
        unpruned_logits = model(probe).cpu().numpy()
    before = {name:value.cpu().clone() for name,value in model.state_dict().items()}
    specs = [("baseline", 0, 0.)]
    for policy in config["policies"]:
        for repetition in range(config["random_repetitions"] if policy == "random" else 1):
            specs.extend((policy,repetition,fraction) for fraction in config["removed_fractions"] if fraction > 0)
    results = []
    for policy, repetition, fraction in specs:
        order = removal_order(policy, norms, matrix, config["random_seed_offset"]+config["seed"]*100+repetition) if policy != "baseline" else list(range(32))
        removed = order[:round(32*fraction)]
        kept = sorted(set(range(32))-set(removed))
        case = f"{policy}_r{repetition:02d}_removed{len(removed):02d}"
        directory = folder / mode / case
        source_model = (ROOT / training["config"]["baseline_source"] / "quaternion" / "model.pt" if mode == "identity" else
                        ROOT / config["source_pilot"] / mode / "model.pt")
        manifest = {"mode": mode, "seed": config["seed"], "policy": policy, "repetition": repetition,
                    "removed_fraction": fraction, "block": config["block"], "removed_units": removed, "kept_units": kept,
                    "random_seed": config["random_seed_offset"]+config["seed"]*100+repetition if policy == "random" else None,
                    "fine_tuning_updates": 0, "model_config": training["config"],
                    "source_checkpoint_sha256": file_sha256(source_model), "calibration_matrix_sha256": file_sha256(saved_matrix)}
        if (directory / "validation.json").exists():
            row = read(directory / "validation.json")
            if read(directory / "manifest.json") != manifest or file_sha256(directory / "model.pt") != row["checkpoint_sha256"]:
                raise ValueError("Changed routed compact case")
            results.append(row)
            continue
        compact = compact_block(model, index, kept).eval()
        with torch.no_grad():
            logits = compact(probe).cpu().numpy()
        expected = predict_from_block(model, index, features, removed, 32) if removed else unpruned_logits
        error = float(np.max(np.abs(logits-expected)))
        if error > 1e-4 or (not removed and error != 0):
            raise AssertionError("Routed compact/masked equivalence failed")
        write_json(directory / "manifest.json", manifest)
        torch.save({"model": {name:value.cpu() for name,value in compact.state_dict().items()}}, directory / "model.pt")
        restored = load_case(directory, training["config"], val_x.device)
        with torch.no_grad():
            reload_error = float((restored(probe)-compact(probe)).abs().max())
        if reload_error != 0:
            raise AssertionError("Routed compact reload failed")
        if mode == "identity":
            old_directory = ROOT / config["baseline_compression"] / f"seed_{config['seed']}" / case
            old_manifest = read(old_directory / "manifest.json")
            if old_manifest["removed_units"] != removed:
                raise AssertionError("Reused identity selection differs")
            previous = next(r for r in old["rows"] if r["seed"] == config["seed"] and r["case"] == case)
            validation = previous["validation"]
        else:
            validation, _ = evaluate(compact, val_x, val_y, mean, std)
        row = {"mode": mode, "seed": config["seed"], "case": case, "policy": policy, "repetition": repetition,
               "removed_fraction": fraction, "real_trainable_parameters": sum(p.numel() for p in compact.parameters()),
               "buffer_elements": sum(b.numel() for b in compact.buffers()),
               "checkpoint_bytes": (directory / "model.pt").stat().st_size,
               "checkpoint_sha256": file_sha256(directory / "model.pt"), "manifest_sha256": file_sha256(directory / "manifest.json"),
               "compact_mask_max_abs_error": error, "reload_max_abs_error": reload_error,
               "validation": validation, "identity_metrics_reused": mode == "identity"}
        write_json(directory / "validation.json", row)
        results.append(row)
        del compact, restored
    if not all(torch.equal(value.cpu(), before[name]) for name,value in model.state_dict().items()):
        raise AssertionError("Source routed model mutated")
    print(f"Routed compression {mode}: 25 frozen cases", flush=True)
    return results


def run(config, folder):
    source = ROOT / config["source_pilot"]
    training = read(source / "summary.json")
    if read(source / "verification.json")["status"] != "passed":
        raise ValueError("Unverified grouping pilot")
    if not all(file_sha256(ROOT / name) == digest for name,digest in training["protocol"]["source_hashes"].items()):
        raise ValueError("Changed frozen routed-model sources")
    frozen_selection = read(source / "selection_frozen.json")
    if frozen_selection["compression_gate"] != training["compression_gate"]:
        raise ValueError("Post-test gate modification")
    protocol = {"config": config, "training_summary_sha256": file_sha256(source / "summary.json"),
                "selection_freeze_sha256": file_sha256(source / "selection_frozen.json"),
                "baseline_summary_sha256": file_sha256(ROOT / config["baseline_compression"] / "summary.json"),
                "source_hashes": {**training["protocol"]["source_hashes"], "scripts/learned_grouping_compression.py": file_sha256(Path(__file__))}}
    if (folder / "protocol.json").exists() and read(folder / "protocol.json") != protocol:
        raise ValueError("Changed conditional compression protocol")
    write_json(folder / "protocol.json", protocol)
    if not training["compression_gate"]["supports_compression_followup"]:
        write_json(folder / "summary.json", {"status": "not_run_gate_failed", "gate": training["compression_gate"],
                   "protocol": protocol, "new_compact_models": 0, "new_test_inference": False})
        (folder / "REPORT.md").write_text("# Conditional routing-compression stage\n\n## Measured gate\n\nThe prespecified validation-only pilot gate failed. No new compact models, test inference or fine-tuning were performed. This is a retained stop result, not a claim that learned grouping cannot work generally. See the training pilot and novelty audit for scope.\n", encoding="utf-8")
        print("Conditional compression skipped: prespecified gate failed", flush=True)
        return
    torch.set_num_threads(4)
    seed_everything(config["seed"])
    device = torch.device(training["config"]["device"])
    (images, labels), (test_images, test_labels) = load_cached_cifar(ROOT / training["config"]["archive"])
    with np.load(source / "split_indices.npz") as split:
        validation = split["validation"]
    statistics = read(source / "data.json")["statistics"]
    mean = torch.tensor(statistics["mean"], dtype=torch.float32, device=device).reshape(1,3,1,1)
    std = torch.tensor(statistics["std"], dtype=torch.float32, device=device).reshape(1,3,1,1)
    val_x, val_y = torch.as_tensor(images[validation], device=device), torch.as_tensor(labels[validation], device=device)
    old = read(ROOT / config["baseline_compression"] / "summary.json")
    rows = []
    for mode in config["modes"]:
        if mode == "identity":
            baseline = ROOT / training["config"]["baseline_source"] / "quaternion"
            model = make_model(training["config"], "identity").to(device).eval()
            model.load_state_dict(torch.load(baseline / "model.pt", weights_only=True, map_location=device)["model"], strict=False)
            calibration_path = baseline / "calibration_activations.npz"
        else:
            model = load_export(source / mode / "model.pt", device)
            calibration_path = source / mode / "calibration_activations.npz"
        with np.load(calibration_path) as saved:
            calibration_values = saved["trained"]
        rows.extend(prepare_mode(mode, model, config, training, folder, val_x, val_y, mean, std, calibration_values, old))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(folder / "validation_frozen.json", {"rows": rows, "all_new_cases_frozen_before_new_test": True})
    test_x, test_y = torch.as_tensor(test_images, device=device), torch.as_tensor(test_labels, device=device)
    results = []
    for row in rows:
        directory = folder / row["mode"] / row["case"]
        if (directory / "metrics.json").exists():
            result = read(directory / "metrics.json")
        elif row["mode"] == "identity":
            previous = next(r for r in old["rows"] if r["seed"] == config["seed"] and r["case"] == row["case"])
            prior_predictions = ROOT / config["baseline_compression"] / f"seed_{config['seed']}" / row["case"] / "test_predictions.npz"
            shutil.copyfile(prior_predictions, directory / "test_predictions.npz")
            result = {**row, "test": previous["test"], "new_test_inference": False}
            write_json(directory / "metrics.json", result)
        else:
            model = load_case(directory, training["config"], device)
            metrics, logits = evaluate(model, test_x, test_y, mean, std)
            np.savez_compressed(directory / "test_predictions.npz", logits=logits, labels=test_labels)
            result = {**row, "test": metrics, "new_test_inference": True}
            write_json(directory / "metrics.json", result)
            del model
        results.append(result)
    write_json(folder / "summary.json", {"status": "completed", "protocol": protocol, "rows": results,
               "validation_freeze_sha256": file_sha256(folder / "validation_frozen.json"), "fine_tuning_updates": 0})
    lines = ["# Conditional compression of learned routing", "", "## Hypothesis and protocol", "",
             "One training seed, 2026. The validation-only hard-quality gate passed. Fixed identity, fixed random and learned-then-hardened routing are compressed to matched actual real parameter counts using frozen random/raw-norm/activation policies, no fine-tuning. Random removal has ten repetitions, not ten independent trainings. Identity validation/test results are reused unchanged; its routed compact states are serialized in the same format. New validation manifests are frozen before new test inference.", "",
             "## Measured results", "", "| Mode | Removed last-block units | Policy | Retained real parameters | Mean test accuracy | Drop from own unpruned model (pp) |", "| --- | ---: | --- | ---: | ---: | ---: |"]
    for mode in config["modes"]:
        baseline_accuracy = next(r["test"]["accuracy"] for r in results if r["mode"] == mode and r["removed_fraction"] == 0)
        for fraction in config["removed_fractions"]:
            for policy in (["baseline"] if fraction == 0 else config["policies"]):
                subset = [r for r in results if r["mode"] == mode and r["removed_fraction"] == fraction and r["policy"] == policy]
                accuracy = float(np.mean([r["test"]["accuracy"] for r in subset]))
                lines.append(f"| {mode} | {100*fraction:.0f}% | {policy} | {subset[0]['real_trainable_parameters']} | {100*accuracy:.2f}% | {100*(baseline_accuracy-accuracy):.2f} |")
    lines.extend(["", "## Interpretation limits", "", "Channel removal gives actual smaller architectures. Assignment alone does not. This one-seed comparison cannot establish a repeatable gain or novelty; soft-training capacity and initial quality remain confounds. No latency/FLOP gain or special algebraic property is claimed. Context: Mena2018, Wang2019FLGC, Zhang2021StructuredShuffle, Vecchi2020, Singh2026 (references.bib)."])
    (folder / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Conditional routed compression completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "learned_grouping_compression.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "learned_grouping_compression")
    args = parser.parse_args()
    run(read(args.config), args.output_dir)
