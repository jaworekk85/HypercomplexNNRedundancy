"""Independently verify BN follow-up states, selections and saved test metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.fusion import fuse_conv_bn_weights

from scripts.cifar_cnn import make_model, write_json
from scripts.cnn_compression import load_compact
from src.compression.bn_magnitude import bn_group_scores, bn_removal_order
from src.compression.quaternion_channels import compact_block
from src.data.cifar_cached import file_sha256
from src.evaluation.block_ablation import classification_metrics
from src.models.quaternion_conv_torch import hamilton_kernel

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(folder):
    torch.set_num_threads(4)
    summary = read(folder / "summary.json")
    config, protocol = summary["config"], summary["protocol"]
    old_folder = ROOT / config["source_compression"]
    frozen = read(folder / "validation_frozen.json")
    checks = {"completed": summary["status"] == "completed", "exactly_six_new_cases": len(summary["rows"]) == 6,
              "same_three_seeds": sorted({r["seed"] for r in summary["rows"]}) == config["seeds"],
              "no_optimizer_updates": summary["fine_tuning_updates"] == 0,
              "no_bn_recalibration": not config["batchnorm_recalibration"],
              "source_hashes_unchanged": all(file_sha256(ROOT / name) == digest for name, digest in protocol["source_hashes"].items()),
              "original_comparison_unchanged": file_sha256(old_folder / "summary.json") == protocol["original_summary_sha256"],
              "original_verification_unchanged": file_sha256(old_folder / "verification.json") == protocol["original_verification_sha256"],
              "new_validation_frozen_before_test": frozen["all_six_cases_frozen_before_test"],
              "validation_freeze_unchanged": file_sha256(folder / "validation_frozen.json") == summary["validation_freeze_sha256"]}
    training = read(ROOT / config["source_multiseed"] / "summary.json")
    originals, orders = {}, {}
    for run in training["runs"]:
        seed = run["seed"]
        path = ROOT / run["path"] / "quaternion" / "model.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = make_model(saved["config"], "quaternion").eval()
        model.load_state_dict(saved["model"], strict=True)
        originals[seed] = (model, file_sha256(path))
        score_report = read(folder / f"seed_{seed}" / "scores.json")
        scores = bn_group_scores(model.blocks[2][3], model.blocks[2][4])
        checks[f"{seed}/saved_scores_reproduce"] = all(scores[key] == score_report[key] for key in scores)
        orders[seed] = bn_removal_order(scores["bn_folded_group_l2"])
        checks[f"{seed}/saved_order_reproduces"] = orders[seed] == score_report["removal_order"]
        conv, bn = model.blocks[2][3:5]
        weight, _ = fuse_conv_bn_weights(hamilton_kernel(conv.kernel.double()), None,
                         bn.running_mean.double(), bn.running_var.double(), bn.eps, bn.weight.double(), bn.bias.double())
        explicit = weight.reshape(4, 32, 128, 3, 3).square().sum(dim=(0, 2, 3, 4)).sqrt().detach().numpy()
        checks[f"{seed}/independent_expanded_folded_norm"] = bool(np.allclose(explicit, scores["bn_folded_group_l2"], rtol=1e-12, atol=1e-12))
        for name, value in score_report["checks"].items():
            if isinstance(value, bool):
                checks[f"{seed}/measured_{name}"] = value
        checks[f"{seed}/measured_zero_removal_equivalence"] = score_report["checks"]["zero_removal_max_abs_logit_error"] == 0
    frozen_rows = {(r["seed"], r["case"]): r for r in frozen["rows"]}
    old = read(old_folder / "summary.json")
    for row in summary["rows"]:
        seed, case = row["seed"], row["case"]
        prefix = f"{seed}/{case}"
        directory = folder / f"seed_{seed}" / case
        manifest = read(directory / "manifest.json")
        model, source_hash = originals[seed]
        removed_count = round(32 * row["removed_fraction"])
        checks[prefix + "/frozen_selection_reproduces_bn_policy"] = manifest["removed_units"] == orders[seed][:removed_count]
        checks[prefix + "/kept_partition"] = manifest["kept_units"] == sorted(set(range(32)) - set(manifest["removed_units"]))
        checks[prefix + "/original_checkpoint_unchanged"] = manifest["source_checkpoint_sha256"] == source_hash
        checks[prefix + "/checkpoint_unchanged"] = file_sha256(directory / "model.pt") == row["checkpoint_sha256"]
        checks[prefix + "/manifest_unchanged"] = file_sha256(directory / "manifest.json") == row["manifest_sha256"]
        actual = load_compact(directory, manifest["base_config"], "cpu")
        expected = compact_block(model, 2, manifest["kept_units"])
        checks[prefix + "/actual_weights_are_exact_slices_of_original"] = all(torch.equal(value, expected.state_dict()[name]) for name, value in actual.state_dict().items())
        parameters = sum(p.numel() for p in actual.parameters())
        checks[prefix + "/actual_parameter_count"] = parameters == row["real_trainable_parameters"] == 73898 - 1200*removed_count
        checks[prefix + "/actual_buffer_count"] = sum(b.numel() for b in actual.buffers()) == row["nontrainable_buffer_elements"] == 902 - 8*removed_count
        checks[prefix + "/actual_checkpoint_bytes"] = (directory / "model.pt").stat().st_size == row["checkpoint_bytes"]
        controls = [r for r in old["rows"] if r["seed"] == seed and r["removed_fraction"] == row["removed_fraction"]]
        checks[prefix + "/matched_real_parameter_count_to_controls"] = all(r["real_trainable_parameters"] == parameters for r in controls)
        checks[prefix + "/measured_compact_mask_equivalence"] = row["compact_vs_masked_max_abs_logit_error"] <= 1e-4
        checks[prefix + "/measured_reload_equivalence"] = row["checkpoint_reload_max_abs_error"] == 0
        checks[prefix + "/validation_frozen_values_preserved"] = all(row[key] == value for key, value in frozen_rows[seed, case].items())
        with np.load(directory / "test_predictions.npz") as arrays:
            logits, labels = arrays["logits"], arrays["labels"]
        metrics = classification_metrics(logits, labels)
        checks[prefix + "/official_test_dimensions_and_finite"] = logits.shape == (10000, 10) and labels.shape == (10000,) and bool(np.isfinite(logits).all())
        checks[prefix + "/recomputed_test_accuracy"] = abs(metrics["accuracy"] - row["test"]["accuracy"]) < 1e-12
        checks[prefix + "/recomputed_test_cross_entropy"] = abs(metrics["cross_entropy"] - row["test"]["cross_entropy"]) < 1e-5
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
              "verifier_sha256": file_sha256(Path(__file__)), "scope": "Inspect exact compact states, independently folded norms and saved predictions; no training or new model inference."}
    write_json(folder / "verification.json", result)
    print(f"BN magnitude artifact verification: {result['status']}; {len(checks)} checks", flush=True)
    if not all(checks.values()):
        raise AssertionError([name for name, passed in checks.items() if not passed])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?", default=ROOT / "results" / "bn_compression")
    verify(parser.parse_args().folder)
