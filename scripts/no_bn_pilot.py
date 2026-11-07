"""Train one no-BN feasibility baseline through the unchanged frozen trainer."""
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import torch

from scripts import cifar_cnn as trainer
from scripts.compression_curve import read
from scripts.paper_a_controls import now, verify_registered_sources
from scripts.run_activation_queue import lock
from src.data.cifar_cached import file_sha256, load_cached_cifar
from src.models.cifar_no_bn import make_no_bn

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/no_bn_pilot.json"


def run():
    study = read(ROOT / CONFIG)
    source, output = ROOT / study["baseline_source"], ROOT / study["output"]
    baseline = read(source / "summary.json")["config"]
    config = {**baseline, "seed": study["seed"], "models": [study["kind"]], "epochs": study["epochs"],
              "device": study["device"], "cpu_threads": study["cpu_threads"], "normalization": "none",
              "checkpoint_selection": "Minimum validation cross-entropy; no test evaluation in this study."}
    with lock("compression_curve_runner.lock"):
        frozen = verify_registered_sources()
        hashes = {**frozen, **{name: file_sha256(ROOT / name) for name in
                  [CONFIG, "scripts/no_bn_pilot.py", "src/models/cifar_no_bn.py"]}}
        inputs = {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in
                  [source / "summary.json", source / "data.json", source / "split_indices.npz", source / study["kind"] / "metrics.json"]}
        protocol = {"config": study, "training_config": config, "source_hashes": hashes, "input_hashes": inputs,
                    "environment": {"torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda},
                    "test_evaluated": False}
        path = output / "protocol.json"
        if path.exists() and read(path) != protocol:
            raise ValueError("No-BN sources/config/input changed; use a new output")
        trainer.write_json(path, protocol)
        trainer.write_json(output / "state.json", {"status": "running", "started_utc": now(), "test_evaluated": False})
        torch.set_num_threads(study["cpu_threads"])
        device = torch.device(study["device"])
        (images, targets), unused_test = load_cached_cifar(ROOT / config["archive"])
        del unused_test
        with np.load(source / "split_indices.npz") as split:
            ids = {name: split[name] for name in ("train", "validation", "calibration")}
        tensors = {name: torch.as_tensor(images[value], device=device) for name, value in ids.items()}
        labels = {name: torch.as_tensor(targets[value], device=device) for name, value in ids.items()}
        stats = read(source / "data.json")["statistics"]
        mean = torch.tensor(stats["mean"], device=device).reshape(1, 3, 1, 1)
        std = torch.tensor(stats["std"], device=device).reshape(1, 3, 1, 1)
        del images
        for name in ("data.json", "split_indices.npz"):
            shutil.copyfile(source / name, output / name)
        with patch.object(trainer, "make_model", make_no_bn), patch.object(trainer, "source_hashes", lambda: hashes):
            result = trainer.train_one(config, study["kind"], tensors, labels, mean, std, output / study["kind"], True)
        if any(file_sha256(ROOT / name) != digest for name, digest in hashes.items()):
            raise ValueError("Source drift during no-BN training")
        reference = read(source / study["kind"] / "metrics.json")
        summary = {"status": "completed", "normalization": "none", "seed": study["seed"], "config": config,
                   "no_bn": result, "existing_bn_reference": reference,
                   "validation_accuracy_difference_pp": 100*(result["selected_validation"]["accuracy"]-reference["selected_validation"]["accuracy"]),
                   "test_evaluated": False, "compression_performed": False}
        trainer.write_json(output / "summary.json", summary)
        lines = ["# One-seed no-BN feasibility pilot", "", "## Hypothesis", "",
                 "The existing six-convolution QNN may train without BN under the original settings. This is a controlled removal ablation, not an optimized normalization-free method.", "",
                 "## Measured result", "", f"Seed {study['seed']}; same train/validation/calibration split, 50 epochs, validation-CE checkpoint selection; no test inference.", "",
                 "| Model | Real parameters | Selected val accuracy (%) | Best epoch |", "| --- | ---: | ---: | ---: |",
                 f"| Original BN | {reference['real_trainable_parameters']} | {100*reference['selected_validation']['accuracy']:.2f} | {reference['best_epoch']} |",
                 f"| No BN | {result['real_trainable_parameters']} | {100*result['selected_validation']['accuracy']:.2f} | {result['best_epoch']} |", "",
                 "## Interpretation limits", "",
                 "One seed, no learning-rate/initialization search, 896 fewer affine parameters, bias-free convolutions retained. Remaining initialization/batch/augmentation settings unchanged. Poor results would not prove normalization is necessary. Good baseline quality would justify separately planning compression without BN; no such compression has been performed here. Prior context: Brock2021NFNets (not reproduced)."]
        (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        trainer.write_json(output / "state.json", {"status": "completed", "completed_utc": now(), "test_evaluated": False})
        print(f"No-BN selected validation {100*result['selected_validation']['accuracy']:.2f}% vs original BN {100*reference['selected_validation']['accuracy']:.2f}%", flush=True)


if __name__ == "__main__":
    run()
