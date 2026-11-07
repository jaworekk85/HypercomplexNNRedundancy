"""Replay the frozen pilot to its boundary without modifying its trainer.

The trainer's hardening and checkpoint hooks only capture extra artifacts.
Both boundary files contain optimizer/scaler/scheduler/RNG states. Their
scheduler state is the usual post-epoch checkpoint state (ready for epoch 41).
The soft model is captured before hardening; the hard model immediately after.
Validation and hardening consume no training RNG or optimizer steps.
"""

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

from scripts import learned_grouping as pilot
from scripts.cifar_cnn import write_json
from src.cnn_training import seed_everything
from src.data.cifar_cached import file_sha256, load_cached_cifar
from src.models.channel_routing import LearnableChannelPermutation

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class BoundaryReached(Exception):
    pass


def capture_boundary(config, tensors, labels, mean, std, folder, hashes):
    """Run the original trainer, stopping after the boundary checkpoint."""
    folder = Path(folder)
    soft_path, hard_path = folder / "boundary_soft.pt", folder / "boundary_hard.pt"
    if (folder / "last.pt").exists():
        last = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
        if last["epoch"] == config["soft_epochs"]:
            if not soft_path.exists() or not hard_path.exists():
                raise RuntimeError("Boundary checkpoint exists without both captures")
            if last["config"] != config or last["source_hashes"] != hashes:
                raise ValueError("Boundary resume provenance mismatch")
            return last["hardening"]
        if last["epoch"] > config["soft_epochs"]:
            raise ValueError("Replay has advanced beyond its intended boundary")
    original_make, original_harden, original_save = pilot.make_model, LearnableChannelPermutation.harden, pilot.save_last
    live, captured = {}, {}

    def make_model(configuration, mode):
        model = original_make(configuration, mode)
        live["model"] = model
        return model

    def harden(router):
        captured["soft"] = {name: value.detach().cpu().clone() for name, value in live["model"].state_dict().items()}
        original_harden(router)

    def save(path, state):
        if state["epoch"] != config["soft_epochs"]:
            return original_save(path, state)
        if "soft" not in captured:
            raise RuntimeError("Soft boundary was not captured")
        soft = {**state, "model": captured["soft"], "best_state": None,
                "best_validation": None, "best_epoch": None, "hardening": None,
                "history": deepcopy(state["history"]),
                "boundary_note": "Pre-hardening model; optimizer/RNG unchanged by hardening/evaluation; scheduler advanced as at an ordinary post-epoch checkpoint."}
        soft["history"][-1].update(phase_after_epoch="soft",
                                   validation_loss=state["hardening"]["soft_validation"]["cross_entropy"],
                                   validation_accuracy=state["hardening"]["soft_validation"]["accuracy"])
        original_save(soft_path, soft)
        original_save(hard_path, state)
        original_save(path, state)
        with (folder / "training.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(state["history"][0]))
            writer.writeheader()
            writer.writerows(state["history"])
        raise BoundaryReached()

    with patch.object(pilot, "make_model", make_model), \
            patch.object(LearnableChannelPermutation, "harden", harden), \
            patch.object(pilot, "save_last", save):
        try:
            pilot.train_mode(config, "learned", tensors, labels, mean, std, folder, hashes)
        except BoundaryReached:
            pass
    if not soft_path.exists() or not hard_path.exists():
        raise RuntimeError("Replay failed to capture the boundary")
    return read(folder / "hardening.json")


def verify_replay(reference, replay, tolerance):
    expected = read(reference / "learned" / "hardening.json")
    actual = read(replay / "hardening.json")
    errors = {}
    for phase in ("soft_validation", "immediate_hard_validation"):
        for metric in ("accuracy", "cross_entropy"):
            errors[f"{phase}.{metric}"] = abs(expected[phase][metric] - actual[phase][metric])
    errors["max_abs_logit_shift"] = abs(expected["max_abs_logit_shift"] - actual["max_abs_logit_shift"])
    with (reference / "learned" / "training.csv").open(encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))
    with (replay / "training.csv").open(encoding="utf-8") as handle:
        new_rows = list(csv.DictReader(handle))
    keys = ("train_augmented_loss", "train_augmented_accuracy", "validation_loss", "validation_accuracy", "learning_rate")
    trace_error = max(abs(float(a[key])-float(b[key])) for a, b in zip(old_rows, new_rows) for key in keys)
    checks = {"all_boundary_metrics_match": max(errors.values()) <= tolerance,
              "all_epoch_metrics_match": trace_error <= tolerance,
              "expected_epoch_count": len(new_rows) == expected["epoch"],
              "matching_is_identical": actual["routing"]["matching"] == expected["routing"]["matching"],
              "temperature_is_identical": actual["routing"]["temperature"] == expected["routing"]["temperature"]}
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "metric_absolute_errors": errors, "maximum_epoch_metric_absolute_error": trace_error,
            "tolerance": tolerance,
            "limitation": "Original epoch-40 weights were not retained; this checks the numerical training trace and boundary metrics, not bitwise identity against an unavailable checkpoint."}


def run(diagnostic):
    reference, output = ROOT / diagnostic["pilot"], ROOT / diagnostic["output"]
    frozen = read(reference / "protocol.json")
    config, hashes = frozen["config"], frozen["source_hashes"]
    if config["soft_epochs"] != diagnostic["boundary_epoch"]:
        raise ValueError("Diagnostic boundary differs from the frozen pilot")
    for name, digest in hashes.items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    protected = {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
                 for path in [reference / "protocol.json", reference / "learned" / "last.pt",
                              reference / "learned" / "model.pt", reference / "learned" / "hardening.json",
                              reference / "learned" / "training.csv", reference / "split_indices.npz", reference / "data.json"]}
    protocol = {"diagnostic": diagnostic, "pilot_protocol": frozen,
                "replay_script_sha256": file_sha256(Path(__file__)), "protected_inputs": protected,
                "test_evaluated": False}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "replay_protocol.json").exists() and read(output / "replay_protocol.json") != protocol:
        raise ValueError("Replay protocol changed; preserve this run and use another output")
    write_json(output / "replay_protocol.json", protocol)
    torch.set_num_threads(config["cpu_threads"])
    seed_everything(config["seed"])
    device = torch.device(config["device"])
    (images, targets), unused_test = load_cached_cifar(ROOT / config["archive"])
    del unused_test
    with np.load(reference / "split_indices.npz") as split:
        indices = {key: split[key] for key in ("train", "validation", "calibration")}
    stats = read(reference / "data.json")["statistics"]
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    tensors = {key: torch.as_tensor(images[index], device=device) for key, index in indices.items()}
    labels = {key: torch.as_tensor(targets[index], device=device) for key, index in indices.items()}
    capture_boundary(config, tensors, labels, mean, std, output / "replay", hashes)
    verification = verify_replay(reference, output / "replay", diagnostic["replay_metric_tolerance"])
    verification["checks"]["protected_inputs_unchanged"] = all(file_sha256(ROOT / name) == digest for name, digest in protected.items())
    verification["status"] = "passed" if all(verification["checks"].values()) else "failed"
    write_json(output / "replay_verification.json", verification)
    print(json.dumps(verification, indent=2), flush=True)
    if verification["status"] != "passed":
        raise AssertionError("Original boundary was not reproduced; do not attribute interventions to that event")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/routing_bn_diagnostic.json")
    arguments = parser.parse_args()
    run(read(ROOT / arguments.config))
