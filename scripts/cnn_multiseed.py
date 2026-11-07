"""Confirm the frozen CNN protocol with two additional seeds, with exact resume."""

import argparse
import gc
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from scripts.cifar_cnn import make_model, source_hashes, train_one, write_json
from scripts.verify_cifar_cnn import verify
from src.cnn_training import evaluate
from src.data.cifar_cached import file_sha256, load_cached_cifar
from src.evaluation.cnn_report import save_cnn_report

ROOT = Path(__file__).resolve().parents[1]


def run(plan, folder):
    original = json.loads((ROOT / plan["source_run"] / "summary.json").read_text(encoding="utf-8"))
    if original["source_hashes"] != source_hashes():
        raise ValueError("Frozen training sources changed")
    gate = json.loads((ROOT / plan["source_block_analysis"] / "summary.json").read_text(encoding="utf-8"))["gate"]
    if not gate["controlled_removal_experiment_supported"]:
        raise ValueError("Diagnostic gate did not support the planned extension")
    own_hash = file_sha256(Path(__file__))
    provenance = {"plan": plan, "runner_sha256": own_hash, "source_hashes": source_hashes(),
                  "original_summary_sha256": file_sha256(ROOT / plan["source_run"] / "summary.json"), "selected_block": gate["selected_block"]}
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "protocol.json").exists():
        if json.loads((folder / "protocol.json").read_text(encoding="utf-8")) != provenance:
            raise ValueError("Multi-seed protocol changed; use another output directory")
    else:
        write_json(folder / "protocol.json", provenance)
    config = original["config"]
    torch.set_num_threads(config["cpu_threads"])
    device = torch.device(config["device"])
    (images, targets), (test_images, test_targets) = load_cached_cifar(ROOT / config["archive"])
    source = ROOT / plan["source_run"]
    with np.load(source / "split_indices.npz") as split:
        indices = {key: split[key] for key in ("train", "validation", "calibration")}
    statistics = json.loads((source / "data.json").read_text(encoding="utf-8"))["statistics"]
    tensors = {key: torch.as_tensor(images[index], device=device) for key, index in indices.items()}
    labels = {key: torch.as_tensor(targets[index], device=device) for key, index in indices.items()}
    mean = torch.tensor(statistics["mean"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(statistics["std"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    del images, targets
    for seed in plan["new_seeds"]:
        current = {**config, "seed": seed}
        seed_folder = folder / f"seed_{seed}"
        seed_folder.mkdir(exist_ok=True)
        if (seed_folder / "config.json").exists() and json.loads((seed_folder / "config.json").read_text(encoding="utf-8")) != current:
            raise ValueError("Seed config changed")
        write_json(seed_folder / "config.json", current)
        for name in ("split_indices.npz", "data.json", "environment.freeze.txt"):
            if not (seed_folder / name).exists():
                shutil.copyfile(source / name, seed_folder / name)
        for kind in plan["models"]:
            write_json(folder / "progress.json", {"stage": "training", "seed": seed, "kind": kind, "test_evaluation_started": False})
            train_one(current, kind, tensors, labels, mean, std, seed_folder / kind, resume=True)
    # All four checkpoint choices are now complete; no training changes follow test evaluation.
    del tensors, labels
    gc.collect()
    torch.cuda.empty_cache()
    test_x, test_y = torch.as_tensor(test_images, device=device), torch.as_tensor(test_targets, device=device)
    completed = [{"seed": config["seed"], "path": plan["source_run"], "models": original["models"]}]
    for seed in plan["new_seeds"]:
        seed_folder = folder / f"seed_{seed}"
        if (seed_folder / "summary.json").exists():
            result = json.loads((seed_folder / "summary.json").read_text(encoding="utf-8"))
        else:
            current = {**config, "seed": seed}
            records = []
            for kind in plan["models"]:
                write_json(folder / "progress.json", {"stage": "final_test", "seed": seed, "kind": kind, "all_trainings_finished": True})
                model_folder = seed_folder / kind
                report = json.loads((model_folder / "metrics.json").read_text(encoding="utf-8"))
                if not report["test_evaluated"]:
                    model = make_model(current, kind).to(device).eval()
                    model.load_state_dict(torch.load(model_folder / "model.pt", map_location=device, weights_only=True)["model"])
                    report["test"], logits = evaluate(model, test_x, test_y, mean, std)
                    report["test_evaluated"] = True
                    report["target_80_percent_achieved"] = report["test"]["accuracy"] >= config["target_accuracy"]
                    np.savez_compressed(model_folder / "test_predictions.npz", logits=logits, labels=test_targets)
                    write_json(model_folder / "metrics.json", report)
                    del model
                records.append(report)
                print(f"FINAL seed {seed} {kind}: test={report['test']['accuracy']:.4f}", flush=True)
            save_cnn_report(records, current, seed_folder)
            # The preserved original formatter contains a seed literal; correct only the generated text.
            report_path = seed_folder / "REPORT.md"
            report_path.write_text(report_path.read_text(encoding="utf-8").replace("One seed, 2026.", f"One seed, {seed}."), encoding="utf-8")
            environment = {"python": sys.version, "torch": torch.__version__, "cuda_build": torch.version.cuda,
                           "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(),
                           "device": str(device), "seed": seed, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                           "versions": {name: metadata.version(name) for name in ("numpy", "matplotlib", "pytest")}}
            result = {"config": current, "environment": environment, "models": records, "source_hashes": source_hashes(),
                      "summary_status": "completed", "primary_target_achieved": records[0]["target_80_percent_achieved"],
                      "runner_sha256": own_hash, "all_four_new_trainings_preceded_test": True}
            write_json(seed_folder / "summary.json", result)
        verify(seed_folder)
        completed.append({"seed": seed, "path": str(seed_folder.relative_to(ROOT)), "models": result["models"]})
    statistics_out = {}
    for kind in plan["models"]:
        values = [next(r for r in seed["models"] if r["kind"] == kind)["test"]["accuracy"] for seed in completed]
        statistics_out[kind] = {"test_accuracies": values, "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}
    differences = np.array(statistics_out["quaternion"]["test_accuracies"]) - statistics_out["real"]["test_accuracies"]
    statistics_out["paired_quaternion_minus_real"] = {"values": differences.tolist(), "mean": float(differences.mean()), "sample_sd": float(differences.std(ddof=1))}
    write_json(folder / "summary.json", {"status": "completed", "plan": plan, "runs": completed, "statistics": statistics_out, "provenance": provenance})
    lines = ["# Frozen CNN protocol: three training seeds", "", "## Hypotheses", "",
             "This confirms training variability without tuning to test results. It does not establish a general quaternion advantage or novelty.", "",
             "## Protocol", "", "Reuse seed 2026; change only the training seed to 2027 and 2028. Fixed architecture, optimizer, 50 epochs, split, normalization and validation-loss checkpoint selection. Complete all four new trainings before evaluating the new selected models on test. The official test set was previously used in the pilots; this is not a new blind benchmark.", "",
             "## Measured results", "", "| Seed | Quaternion test accuracy | Real test accuracy | Paired difference (pp) |", "| ---: | ---: | ---: | ---: |"]
    for index, seed in enumerate(completed):
        q, r = (statistics_out[kind]["test_accuracies"][index] for kind in ("quaternion", "real"))
        lines.append(f"| {seed['seed']} | {100*q:.2f}% | {100*r:.2f}% | {100*(q-r):+.2f} |")
    for kind in plan["models"]:
        value = statistics_out[kind]
        lines.extend(["", f"{kind}: mean {100*value['mean']:.2f}%, sample SD {100*value['sample_sd']:.2f} percentage points (n=3 training seeds)."])
    paired = statistics_out["paired_quaternion_minus_real"]
    lines.extend(["", f"Paired difference: {100*paired['mean']:+.2f} pp, sample SD {100*paired['sample_sd']:.2f} pp.", "",
                  "## Interpretation", "", "Three seeds characterize variability under this one architecture/training protocol. Parameter budgets are close (73,898 vs 72,954) but scalar widths, compute and initial quality differ. No significance claim or broad superiority claim follows from three seeds. Effective rank and ablation diagnostics do not by themselves prove useful physical compression."])
    (folder / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(folder / "progress.json", {"stage": "completed", "seeds": plan["seeds"]})
    print("Three-seed comparison completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "cnn_multiseed.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "cnn_multiseed")
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")), args.output_dir)
