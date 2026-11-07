"""One frozen BN-aware weight reference on the existing three-seed checkpoints."""

import argparse
import csv
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_weights

from scripts.cifar_cnn import make_model, write_json
from scripts.cnn_compression import load_compact
from src.cnn_training import evaluate, prepare_batch, seed_everything
from src.compression.bn_magnitude import bn_group_scores, bn_removal_order
from src.compression.quaternion_channels import compact_block
from src.data.cifar_cached import file_sha256, load_cached_cifar
from src.evaluation.block_ablation import capture_blocks, predict_from_block, spearman
from src.models.quaternion_conv_torch import hamilton_kernel

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["scripts/bn_compression.py", "src/compression/bn_magnitude.py"]


@torch.no_grad()
def folding_checks(model, block_index, scores):
    conv, bn = model.blocks[block_index][3:5]
    # Independent reference uses official installed PyTorch folding on expanded weights.
    weight, bias = fuse_conv_bn_weights(hamilton_kernel(conv.kernel.detach().cpu().double()),
                    conv.bias.detach().cpu().double() if conv.bias is not None else None,
                    bn.running_mean.cpu().double(), bn.running_var.cpu().double(), bn.eps,
                    bn.weight.detach().cpu().double(), bn.bias.detach().cpu().double())
    explicit = weight.reshape(4, conv.out_channels, 4*conv.in_channels, *conv.kernel.shape[-2:]).square().sum(dim=(0, 2, 3, 4)).sqrt()
    compact_formula = torch.tensor(scores["bn_folded_group_l2"], dtype=torch.float64)
    error = float((explicit - compact_formula).abs().max())
    torch.testing.assert_close(explicit, compact_formula, rtol=1e-12, atol=1e-12)
    device = conv.kernel.device
    fw, fb = fuse_conv_bn_weights(hamilton_kernel(conv.kernel), conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    probe = torch.randn(2, 4*conv.in_channels, 8, 8, device=device)
    expected, actual = bn(conv(probe)), F.conv2d(probe, fw, fb, stride=conv.stride, padding=conv.padding)
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-5)
    return {"score_vs_expanded_float64_max_abs_error": error,
            "conv_bn_vs_folded_float32_max_abs_error": float((expected-actual).abs().max())}


def prepare_seed(run, config, folder, old_folder, images, labels, indices, statistics):
    seed, device = run["seed"], torch.device(config["device"])
    seed_everything(seed)
    source = ROOT / run["path"]
    training = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    model_config = training["config"]
    model = make_model(model_config, "quaternion").to(device).eval()
    checkpoint_path = source / "quaternion" / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    index = config["block"] - 1
    conv, bn = model.blocks[index][3:5]
    scores = bn_group_scores(conv, bn)
    order = bn_removal_order(scores["bn_folded_group_l2"])
    checks = folding_checks(model, index, scores)
    q_report = next(r for r in training["models"] if r["kind"] == "quaternion")
    checks["original_gradient_update_audit_passed"] = all(q_report["first_update"]["checks"].values())
    val_x = torch.as_tensor(images[indices["validation"]], device=device)
    val_y = torch.as_tensor(labels[indices["validation"]], device=device)
    mean = torch.tensor(statistics["mean"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(statistics["std"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    probe = prepare_batch(val_x[:32], mean, std)
    features = capture_blocks(model, probe, 32)[index]
    with torch.no_grad():
        original_logits = model(probe).cpu().numpy()
        zero = compact_block(model, index, list(range(conv.out_channels)))
        zero_error = float(np.max(np.abs(zero(probe).cpu().numpy() - original_logits)))
    if zero_error != 0:
        raise AssertionError("Zero-removal equivalence failed")
    checks["zero_removal_max_abs_logit_error"] = zero_error
    del zero
    ablations = json.loads((old_folder / "block_analysis" / f"seed_{seed}" / "quaternion" / "summary.json").read_text(encoding="utf-8"))["blocks"][index]["ablations"]
    harms = [row["validation_loss_increase"] for row in ablations]
    diagnostics = {"raw_norm_vs_individual_loss_harm_spearman": spearman(scores["raw_kernel_l2"], harms),
                   "bn_norm_vs_individual_loss_harm_spearman": spearman(scores["bn_folded_group_l2"], harms)}
    rows = []
    for fraction in config["removed_fractions"]:
        removed = order[:round(conv.out_channels*fraction)]
        kept = sorted(set(range(conv.out_channels)) - set(removed))
        case_id = f"bn_folded_group_l2_removed{len(removed):02d}"
        case_dir = folder / f"seed_{seed}" / case_id
        manifest = {"seed": seed, "policy": config["policy"], "block": config["block"], "removed_fraction": fraction,
                    "original_units": conv.out_channels, "removed_units": removed, "kept_units": kept,
                    "base_config": model_config, "source_checkpoint_sha256": file_sha256(checkpoint_path),
                    "score_formula": config["score"], "ranking_dtype": "CPU float64", "fine_tuning_updates": 0}
        # Verify that the raw magnitude reference would select the exact original units.
        old_manifest = json.loads((old_folder / f"seed_{seed}" / f"incoming_norm_r00_removed{len(removed):02d}" / "manifest.json").read_text(encoding="utf-8"))
        checks[f"raw_reference_{len(removed)}_selection_reproduced"] = old_manifest["removed_units"] == bn_removal_order(scores["raw_kernel_l2"])[:len(removed)]
        if (case_dir / "validation.json").exists():
            if json.loads((case_dir / "manifest.json").read_text(encoding="utf-8")) != manifest:
                raise ValueError("BN manifest changed")
            row = json.loads((case_dir / "validation.json").read_text(encoding="utf-8"))
            if file_sha256(case_dir / "model.pt") != row["checkpoint_sha256"]:
                raise ValueError("BN compact checkpoint changed")
            rows.append(row)
            continue
        compact = compact_block(model, index, kept).eval()
        expected = predict_from_block(model, index, features, removed, 32)
        with torch.no_grad():
            compact_logits = compact(probe).cpu().numpy()
        error = float(np.max(np.abs(compact_logits - expected)))
        if error > 1e-4:
            raise AssertionError("BN-policy compact vs mask equivalence failed")
        write_json(case_dir / "manifest.json", manifest)
        torch.save({"model": {name: value.detach().cpu() for name, value in compact.state_dict().items()}}, case_dir / "model.pt")
        restored = load_compact(case_dir, model_config, device)
        with torch.no_grad():
            reload_error = float((restored(probe) - compact(probe)).abs().max())
        if reload_error != 0:
            raise AssertionError("BN-policy reload equivalence failed")
        metrics, _ = evaluate(compact, val_x, val_y, mean, std, config["batch_size"])
        row = {"seed": seed, "case": case_id, "policy": config["policy"], "removed_fraction": fraction,
               "removed_quaternion_units": len(removed), "retained_quaternion_units": len(kept),
               "real_trainable_parameters": sum(p.numel() for p in compact.parameters()),
               "nontrainable_buffer_elements": sum(b.numel() for b in compact.buffers()),
               "checkpoint_bytes": (case_dir / "model.pt").stat().st_size,
               "checkpoint_sha256": file_sha256(case_dir / "model.pt"), "manifest_sha256": file_sha256(case_dir / "manifest.json"),
               "compact_vs_masked_max_abs_logit_error": error, "checkpoint_reload_max_abs_error": reload_error,
               "validation": metrics, "validation_accuracy_drop": q_report["selected_validation"]["accuracy"] - metrics["accuracy"],
               "validation_cross_entropy_increase": metrics["cross_entropy"] - q_report["selected_validation"]["cross_entropy"]}
        write_json(case_dir / "validation.json", row)
        rows.append(row)
        del compact, restored
    checks["source_state_unchanged"] = all(torch.equal(value.cpu(), before[name]) for name, value in model.state_dict().items())
    if not all(value for key, value in checks.items() if isinstance(value, bool)):
        raise AssertionError("Source/update/reference check failed")
    write_json(folder / f"seed_{seed}" / "scores.json", {"seed": seed, "block": config["block"], **scores,
               "removal_order": order, "checks": checks, "diagnostics": diagnostics,
               "source_checkpoint_sha256": file_sha256(checkpoint_path)})
    print(f"BN seed {seed}: rankings and both compact validation cases frozen", flush=True)
    del model, val_x, val_y, features
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def run(config, folder):
    torch.set_num_threads(4)
    old_folder = ROOT / config["source_compression"]
    old = json.loads((old_folder / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((old_folder / "verification.json").read_text(encoding="utf-8"))
    if old["status"] != "completed" or verification["status"] != "passed":
        raise ValueError("Original comparison must be completed and verified")
    old_sources = old["protocol"]["source_hashes"]
    if not all(file_sha256(ROOT / name) == digest for name, digest in old_sources.items()):
        raise ValueError("Frozen original sources changed")
    if config["fine_tuning_updates"] or config["batchnorm_recalibration"] or config["block"] != 3:
        raise ValueError("Only the fixed block-3 no-training follow-up is supported")
    protocol = {"config": config, "source_hashes": {**old_sources, **{name: file_sha256(ROOT / name) for name in SOURCES}},
                "original_summary_sha256": file_sha256(old_folder / "summary.json"),
                "original_verification_sha256": file_sha256(old_folder / "verification.json"),
                "multiseed_summary_sha256": file_sha256(ROOT / config["source_multiseed"] / "summary.json")}
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "protocol.json").exists() and json.loads((folder / "protocol.json").read_text(encoding="utf-8")) != protocol:
        raise ValueError("BN follow-up protocol changed; use another output directory")
    write_json(folder / "protocol.json", protocol)
    multiseed = json.loads((ROOT / config["source_multiseed"] / "summary.json").read_text(encoding="utf-8"))
    if [r["seed"] for r in multiseed["runs"]] != config["seeds"]:
        raise ValueError("Training seeds changed")
    source = ROOT / multiseed["runs"][0]["path"]
    original = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    (images, labels), (test_images, test_labels) = load_cached_cifar(ROOT / original["config"]["archive"])
    with np.load(source / "split_indices.npz") as split:
        indices = {key: split[key] for key in ("validation", "calibration")}
    statistics = json.loads((source / "data.json").read_text(encoding="utf-8"))["statistics"]
    rows = []
    for run_record in multiseed["runs"]:
        rows.extend(prepare_seed(run_record, config, folder, old_folder, images, labels, indices, statistics))
    write_json(folder / "validation_frozen.json", {"rows": rows, "all_six_cases_frozen_before_test": True,
               "no_policy_or_size_selected_from_validation_or_test": True})
    device = torch.device(config["device"])
    test_x, test_y = torch.as_tensor(test_images, device=device), torch.as_tensor(test_labels, device=device)
    mean = torch.tensor(statistics["mean"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(statistics["std"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    results = []
    for row in rows:
        case_dir = folder / f"seed_{row['seed']}" / row["case"]
        if (case_dir / "metrics.json").exists():
            result = json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))
        else:
            model = load_compact(case_dir, {**original["config"], "seed": row["seed"]}, device)
            metrics, logits = evaluate(model, test_x, test_y, mean, std, config["batch_size"])
            np.savez_compressed(case_dir / "test_predictions.npz", logits=logits, labels=test_labels)
            result = {**row, "test": metrics, "test_evaluated_after_all_new_validation_cases_frozen": True}
            write_json(case_dir / "metrics.json", result)
            del model
        results.append(result)
    environment = {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
                   "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                   "gpu": torch.cuda.get_device_name(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "score_dtype_device": "CPU float64", "inference_dtype_device": "CUDA float32"}
    write_json(folder / "summary.json", {"status": "completed", "config": config, "rows": results, "protocol": protocol,
               "environment": environment, "fine_tuning_updates": 0,
               "validation_freeze_sha256": file_sha256(folder / "validation_frozen.json")})
    render_report(results, old["rows"], config, folder)
    print("BatchNorm-aware magnitude follow-up completed", flush=True)


def render_report(rows, reference_rows, config, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = []
    for fraction in config["removed_fractions"]:
        for policy in ("random", "incoming_norm", "activation_similarity", config["policy"]):
            subset = [r for r in rows + reference_rows if r["removed_fraction"] == fraction and r["policy"] == policy]
            per_seed = [float(np.mean([r["test"]["accuracy"] for r in subset if r["seed"] == seed])) for seed in config["seeds"]]
            groups.append({"removed_fraction": fraction, "policy": policy, "real_trainable_parameters": subset[0]["real_trainable_parameters"],
                           "checkpoint_bytes": subset[0]["checkpoint_bytes"], "test_accuracy_mean": float(np.mean(per_seed)),
                           "test_accuracy_sample_sd_across_training_seed_means": float(np.std(per_seed, ddof=1)), "test_accuracy_per_seed": per_seed,
                           "validation_accuracy_mean": float(np.mean([np.mean([r["validation"]["accuracy"] for r in subset if r["seed"] == seed]) for seed in config["seeds"]]))})
    paired = []
    for fraction in config["removed_fractions"]:
        bn = next(g for g in groups if g["removed_fraction"] == fraction and g["policy"] == config["policy"])
        for policy in ("incoming_norm", "random", "activation_similarity"):
            other = next(g for g in groups if g["removed_fraction"] == fraction and g["policy"] == policy)
            effect = np.array(bn["test_accuracy_per_seed"]) - other["test_accuracy_per_seed"]
            paired.append({"removed_fraction": fraction, "reference": policy, "values_pp": (100*effect).tolist(),
                           "mean_pp": float(100*effect.mean()), "sample_sd_pp": float(100*effect.std(ddof=1))})
    write_json(folder / "comparison.json", {"groups": groups, "bn_minus_reference_paired_effects": paired})
    with (folder / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
        simple = [{**{k:v for k,v in row.items() if k not in ("validation", "test")}, **{f"{stage}_{key}": value for stage in ("validation", "test") for key,value in row[stage].items()}} for row in rows]
        writer = csv.DictWriter(handle, fieldnames=list(simple[0]))
        writer.writeheader()
        writer.writerows(simple)
    fig, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    for policy in ("random", "incoming_norm", "activation_similarity", config["policy"]):
        selected = [g for g in groups if g["policy"] == policy]
        axis.errorbar([g["real_trainable_parameters"] for g in selected], [100*g["test_accuracy_mean"] for g in selected],
                      yerr=[100*g["test_accuracy_sample_sd_across_training_seed_means"] for g in selected], marker="o", capsize=3, label=policy)
    axis.set(xlabel="Actual retained real trainable parameters", ylabel="Test accuracy (%)", title="Frozen BN-aware reference, three training seeds")
    axis.legend(fontsize=8)
    fig.savefig(folder / "comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    lines = ["# BatchNorm-aware magnitude follow-up", "", "## Hypothesis", "",
             "Conventional BN-folded weight magnitude may be a more meaningful reference than raw quaternion kernel norm. This is one fixed follow-up, not a novel method or a guaranteed stronger result.", "", "## Fixed protocol", "",
             "Same three selected models, block 3 and physical 25/50% removal as before. No training, BN re-calibration, compensation or new policy/size selection. All six new manifests and validation evaluations were frozen before test. Existing reference results are reused unchanged. The test set was already used in preceding pilots; this is exploratory.",
             "Score: joint Frobenius norm of all four output rows in the expanded Hamilton convolution after per-row gamma/sqrt(running_var+eps) scaling. CPU float64 ranking, lowest score removed first, original-index tie break. Folded bias and downstream effects are excluded. Folding is used to score only: component-wise BN generally does not preserve the Hamilton kernel structure. Source: PyTorchFusion260; quaternion pruning context: Vecchi2020, Singh2026 (references.bib).", "",
             "## Measured results", "", "| Removed block units | Policy | Retained real parameters | State bytes | Validation mean | Test mean +/- training-seed SD |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for group in groups:
        lines.append(f"| {100*group['removed_fraction']:.0f}% | {group['policy']} | {group['real_trainable_parameters']} | {group['checkpoint_bytes']} | {100*group['validation_accuracy_mean']:.2f}% | {100*group['test_accuracy_mean']:.2f} +/- {100*group['test_accuracy_sample_sd_across_training_seed_means']:.2f}% |")
    lines.extend(["", "Random is averaged across ten removals within each training seed before calculating training-seed SD. Three seeds are not enough for a broad superiority/significance claim.", "", "Paired BN-aware minus reference test accuracy (pp; seeds 2026, 2027, 2028):", ""])
    for effect in paired:
        lines.append(f"- {100*effect['removed_fraction']:.0f}% vs {effect['reference']}: {', '.join(f'{v:+.2f}' for v in effect['values_pp'])}; mean {effect['mean_pp']:+.2f}, sample SD {effect['sample_sd_pp']:.2f}.")
    lines.extend(["", "## Interpretation", "",
                  "Changing BN scaling can change magnitude rankings; usefulness must be judged by held-out performance rather than the score's motivation. Better, unchanged and worse results are all retained. The baseline ignores folded offsets, ReLU and outgoing weights, so even an exact folded-kernel norm is not full functional importance. No score variant was selected to fix a negative result.",
                  "Actual compact parameter counts match the frozen controls: 64,298 and 54,698 (12.99% and 25.98% total-model savings). A 25/50% block-unit removal fraction is not the whole-model saving. No new MAC or latency measurement is claimed."])
    (folder / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "bn_compression.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "bn_compression")
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")), args.output_dir)
