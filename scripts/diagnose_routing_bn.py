"""Compare soft/hard routing with original/refitted BN, at fixed weights."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import torch

from scripts.cifar_cnn import write_json
from scripts.learned_grouping import make_model
from src.cnn_training import evaluate, seed_everything
from src.data.cifar_cached import file_sha256, load_cached_cifar
from src.evaluation.routing_hardening import recalibrate_bn_layers

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
SOURCES = ["scripts/diagnose_routing_bn.py", "src/evaluation/routing_hardening.py"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def contrasts(records, split):
    values = {name: row["evaluation"][split] for name, row in records.items()}
    s, h, sr, hr = (values[name] for name in ("soft_original", "hard_original", "soft_recalibrated", "hard_recalibrated"))
    original_gap = s["accuracy"] - h["accuracy"]
    refit_gap = sr["accuracy"] - hr["accuracy"]
    return {"original_soft_minus_hard_accuracy_pp": 100*original_gap,
            "refitted_soft_minus_hard_accuracy_pp": 100*refit_gap,
            "hard_refit_accuracy_gain_pp": 100*(hr["accuracy"]-h["accuracy"]),
            "soft_refit_accuracy_gain_pp": 100*(sr["accuracy"]-s["accuracy"]),
            "gap_reduction_pp": 100*(original_gap-refit_gap),
            "gap_reduction_fraction": (original_gap-refit_gap)/original_gap if original_gap else None,
            "original_hard_minus_soft_cross_entropy": h["cross_entropy"]-s["cross_entropy"],
            "refitted_hard_minus_soft_cross_entropy": hr["cross_entropy"]-sr["cross_entropy"]}


def write_report(output, summary):
    rows = ["# BatchNorm intervention at the soft-to-hard boundary", "",
            "## Hypotheses", "",
            "The original one-seed pilot's hardening loss may partly reflect downstream BN running-statistic mismatch. "
            "A separate possibility is that the soft mixing operator provides behavior a permutation cannot reproduce. "
            "These hypotheses are not mutually exclusive.", "",
            "## Protocol", "",
            "Replay the frozen seed-2026 pilot through epoch 40; retain full pre/post-hardening checkpoints. "
            "Compare the same weights in four prespecified conditions. Refit only the four downstream BNs, "
            "in forward order, using all 48,000 training images without augmentation. "
            "All modules, including dropout, stay in evaluation mode. Moments accumulate in float64; variance uses N-1 "
            "observations including spatial positions. Earlier BNs use their newly fitted evaluation statistics. "
            "BN affine parameters, every model parameter, upstream BN buffers, counters, and routing remain unchanged by refitting. "
            "No optimizer steps or official-test inference are performed after the boundary.", "",
            "## Measured results", "",
            "| Condition | Validation accuracy | Validation CE | Calibration accuracy | Calibration CE |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for name, result in summary["conditions"].items():
        v, c = result["evaluation"]["validation"], result["evaluation"]["calibration"]
        rows.append(f"| {name} | {100*v['accuracy']:.2f}% | {v['cross_entropy']:.6f} | {100*c['accuracy']:.2f}% | {c['cross_entropy']:.6f} |")
    v = summary["contrasts"]["validation"]
    rows += ["", f"Validation hard-routing gain from BN refitting: **{v['hard_refit_accuracy_gain_pp']:+.2f} pp**. "
             f"Soft-routing control change: **{v['soft_refit_accuracy_gain_pp']:+.2f} pp**. "
             f"The soft-minus-hard gap is {v['original_soft_minus_hard_accuracy_pp']:.2f} pp with original BN and "
             f"{v['refitted_soft_minus_hard_accuracy_pp']:.2f} pp with both BNs refitted.", "",
             "Replay verification: numerical epoch metrics and boundary metrics match the original within "
             f"{summary['replay_verification']['tolerance']:.1e}. The original boundary weights were not saved, "
             "so bitwise identity to those unavailable weights cannot be checked.", "",
             "## Interpretation and limits", "",
             "The paired soft refit is a control for the refitting procedure itself. The observed hard-model recovery "
             "measures what this training-only BN-buffer intervention repairs at fixed weights. Any remaining gap is "
             "not repaired by this procedure; it does not prove an unavoidable limitation of learned grouping. "
             "The result is one seed, one CNN, one boundary and one refitting estimator. No new method, general gain, "
             "compression saving or publication readiness is claimed. A hard model may still need weight adaptation; "
             "that is not tested here.", "",
             "The retained soft router has 1,024 extra real parameters. Hard condition checkpoints retain those frozen "
             "logits for exact diagnosis; they are not compact exports. The pilot's existing fixed-router export "
             "mechanism can remove them, but rearrangement alone removes no backbone channels.", "",
             "Prior context: Mena2018 (permutation relaxation); Wang2019FLGC and Zhang2021StructuredShuffle "
             "(learned grouping/shuffle), as recorded in references.bib. This diagnostic makes no priority claim.", "",
             "## Artifacts", "",
             "- replay/boundary_soft.pt and replay/boundary_hard.pt: full boundary states.",
             "- intervention_protocol.json: choices, source hashes, split and input hashes before the intervention.",
             "- */model.pt, predictions.npz, metrics.json: reproducible fixed-weight conditions.",
             "- evaluation_targets.npz: held-out labels and original split indices.",
             "- summary.json and comparison.png: numeric results and comparison.", ""]
    (output / "REPORT.md").write_text("\n".join(rows), encoding="utf-8")


def plot_results(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    labels = ["Soft\noriginal BN", "Hard\noriginal BN", "Soft\nrefitted BN", "Hard\nrefitted BN"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, split in zip(axes, ("validation", "calibration")):
        values = [100*record["evaluation"][split]["accuracy"] for record in summary["conditions"].values()]
        bars = ax.bar(labels, values, color=["#326ea8", "#bb5545", "#709cc1", "#55a077"])
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set_ylim(0, 100)
        ax.set_title(f"{split.capitalize()} (n=1,000)")
        ax.set_ylabel("Accuracy (%)")
    fig.suptitle("Epoch 40: fixed weights; BN refit on 48,000 training images only; one seed")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)


def run(diagnostic):
    output, reference = ROOT / diagnostic["output"], ROOT / diagnostic["pilot"]
    replay_verification = read(output / "replay_verification.json")
    if replay_verification["status"] != "passed":
        raise ValueError("A verified boundary replay is required")
    frozen = read(reference / "protocol.json")
    config = frozen["config"]
    if diagnostic["official_test_evaluation"] or diagnostic["bn_fit_split"] != "train":
        raise ValueError("This is a training-only BN intervention with no official test inference")
    hashes = {**frozen["source_hashes"], **{name: file_sha256(ROOT / name) for name in SOURCES}}
    for name, digest in frozen["source_hashes"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f"Frozen pilot source changed: {name}")
    input_paths = [output / "replay" / "boundary_soft.pt", output / "replay" / "boundary_hard.pt",
                   output / "replay_protocol.json", output / "replay_verification.json",
                   reference / "split_indices.npz", reference / "data.json"]
    input_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in input_paths}
    protocol = {"diagnostic": diagnostic, "source_hashes": hashes, "input_hashes": input_hashes,
                "official_test_evaluated": False, "weight_updates_after_boundary": 0,
                "seed": config["seed"], "device": config["device"],
                "environment": {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
                                "cuda": torch.version.cuda, "numpy": np.__version__,
                                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}}
    if (output / "intervention_protocol.json").exists() and read(output / "intervention_protocol.json") != protocol:
        raise ValueError("Intervention protocol changed; use a different output directory")
    write_json(output / "intervention_protocol.json", protocol)
    if (output / "summary.json").exists():
        summary = read(output / "summary.json")
        # A process may have stopped after numeric output but before plotting.
        write_report(output, summary)
        plot_results(output, summary)
        print("BN diagnostic already complete", flush=True)
        return summary
    torch.set_num_threads(config["cpu_threads"])
    seed_everything(config["seed"])
    device = torch.device(config["device"])
    (images, labels), unused_test = load_cached_cifar(ROOT / config["archive"])
    del unused_test
    with np.load(reference / "split_indices.npz") as split:
        indices = {key: split[key] for key in ("train", "validation", "calibration")}
    if len(indices["train"]) != diagnostic["bn_fit_samples"] or any(np.intersect1d(indices["train"], indices[key]).size for key in diagnostic["evaluation_splits"]):
        raise ValueError("Training refit set differs from the declared disjoint split")
    tensors = {key: torch.as_tensor(images[index], device=device) for key, index in indices.items()}
    targets = {key: torch.as_tensor(labels[index], device=device) for key, index in indices.items()}
    stats = read(reference / "data.json")["statistics"]
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    boundaries, base_models = {}, {}
    for phase in ("soft", "hard"):
        boundaries[phase] = torch.load(output / "replay" / f"boundary_{phase}.pt", map_location="cpu", weights_only=True)
        model = make_model(config, "learned").to(device).eval()
        model.load_state_dict(boundaries[phase]["model"])
        base_models[phase] = model
    if not all(torch.equal(value, boundaries["hard"]["model"][name]) for name, value in boundaries["soft"]["model"].items() if name not in ("router.indices", "router.hard_enabled")):
        raise AssertionError("Boundary comparison changed more than assignment buffers")
    np.savez_compressed(output / "evaluation_targets.npz", **{f"{key}_labels": labels[indices[key]] for key in diagnostic["evaluation_splits"]},
                        **{f"{key}_indices": indices[key] for key in diagnostic["evaluation_splits"]})
    records = {}
    for condition in diagnostic["conditions"]:
        folder = output / condition
        folder.mkdir(parents=True, exist_ok=True)
        phase = condition.split("_")[0]
        model = deepcopy(base_models[phase]).eval()
        refit = recalibrate_bn_layers(model, tensors["train"], mean, std, diagnostic["bn_layers"], diagnostic["bn_batch_size"]) if condition.endswith("recalibrated") else None
        metrics, predictions = {}, {}
        for key in diagnostic["evaluation_splits"]:
            metrics[key], predictions[f"{key}_logits"] = evaluate(model, tensors[key], targets[key], mean, std)
        if condition.endswith("original"):
            expected = boundaries["hard"]["hardening"]["soft_validation" if phase == "soft" else "immediate_hard_validation"]
            if any(abs(metrics["validation"][key]-expected[key]) > diagnostic["replay_metric_tolerance"] for key in expected):
                raise AssertionError("Saved boundary does not reproduce the verified validation metrics")
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        torch.save({"model": state, "config": config, "condition": condition, "epoch": diagnostic["boundary_epoch"]}, folder / "model.pt")
        np.savez_compressed(folder / "predictions.npz", **predictions)
        result = {"condition": condition, "evaluation": metrics, "refit": refit,
                  "retained_real_parameter_elements": sum(p.numel() for p in model.parameters()),
                  "routing_is_hard": bool(model.router.hard_enabled), "weight_updates_after_boundary": 0,
                  "official_test_evaluated": False, "checkpoint_sha256": file_sha256(folder / "model.pt"),
                  "predictions_sha256": file_sha256(folder / "predictions.npz")}
        write_json(folder / "metrics.json", result)
        records[condition] = result
        print(f"{condition}: validation={100*metrics['validation']['accuracy']:.2f}%, calibration={100*metrics['calibration']['accuracy']:.2f}%", flush=True)
        del model
    if any(file_sha256(ROOT / name) != digest for name, digest in input_hashes.items()):
        raise AssertionError("A protected input changed")
    summary = {"status": "complete", "seed": config["seed"], "boundary_epoch": diagnostic["boundary_epoch"],
               "conditions": records, "contrasts": {key: contrasts(records, key) for key in diagnostic["evaluation_splits"]},
               "replay_verification": replay_verification, "weight_updates_after_boundary": 0,
               "official_test_evaluated": False, "compact_parameter_savings": 0,
               "scope": diagnostic["scope"]}
    write_json(output / "summary.json", summary)
    write_report(output, summary)
    plot_results(output, summary)
    print(json.dumps(summary["contrasts"], indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/routing_bn_diagnostic.json")
    arguments = parser.parse_args()
    run(read(ROOT / arguments.config))
