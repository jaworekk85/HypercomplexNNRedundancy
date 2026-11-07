"""Bounded matched-victim ResNet20 compression, resumed by site and epoch."""
from copy import deepcopy
import argparse
import os
from pathlib import Path
import platform

import numpy as np
import torch

from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from scripts.paper_a_controls import now, rel
from scripts.run_activation_queue import lock
from scripts.resnet20_baselines import make_model
from src.cnn_training import evaluate, first_update_audit, prepare_batch, seed_everything
from src.compression.multilayer import site_maps
from src.compression.resnet_core import choose, compact, fold_error, moments, refit
from src.compression.structure_catalogue import collect
from src.data.cifar_cached import file_sha256
from src.models.cifar_resnet import CifarResNet20
from src.trainable_map_repair import atomic_save, repair

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/resnet20_compression.json"
NEW = [CONFIG, "scripts/resnet20_compression.py", "src/compression/resnet_core.py", "tests_torch/test_resnet_core.py", "scripts/smoke_resnet20_compression.py", "src/compression/structure_catalogue.py", "src/models/phm_conv.py"]
SPLITS = ("validation", "check_calibration")


def register(study):
    parent = curve.read(ROOT/study["parent_protocol"])
    for name, digest in {**parent["source_hashes"], **parent["input_hashes"]}.items():
        if file_sha256(ROOT/name) != digest: raise ValueError(f"Frozen parent changed: {name}")
    inputs = [ROOT/study["parent_protocol"], ROOT/study["baseline_config"], ROOT/"results/resnet20_baselines/summary.json"]
    for family in study["families"]:
        folder = ROOT/"results/resnet20_baselines"/family
        row = curve.read(folder/"metrics.json")
        if file_sha256(folder/"model.pt") != row["checkpoint_sha256"]: raise ValueError("Baseline checkpoint changed")
        inputs.extend([folder/"model.pt", folder/"metrics.json"])
    definition = {"config": study, "source_hashes": {**parent["source_hashes"], **{p: file_sha256(ROOT/p) for p in NEW}},
                  "input_hashes": {**parent["input_hashes"], **{rel(p): file_sha256(p) for p in inputs}},
                  "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda},
                  "prior_art": study["prior_art"], "test_evaluated": False}
    output = ROOT/study["output"]
    output.mkdir(parents=True, exist_ok=True)
    path = output/"protocol.json"
    if path.exists() and curve.read(path) != definition: raise ValueError("Registered ResNet compression changed")
    write_json(path, definition)
    return definition


def widths(model):
    return [block.conv1.out_channels for block in model.blocks]


def save_model(model, path):
    atomic_save({"model": {n: v.detach().cpu() for n, v in model.state_dict().items()}, "hidden_widths": widths(model)}, path)
    return file_sha256(path)


def load_model(path, baseline, family, device):
    data = torch.load(path, map_location="cpu", weights_only=True)
    model = CifarResNet20(family, baseline["real_widths"], baseline["quaternion_widths"], baseline["classes"], data["hidden_widths"]).to(device)
    model.load_state_dict(data["model"])
    return model.eval()


def prepare(study):
    smoke = curve.read(ROOT/".cache/resnet20_compression_smoke/last_readiness.json")
    if smoke["status"] != "passed" or smoke["scientific_results"] or smoke["source_hashes"] != {p: file_sha256(ROOT/p) for p in NEW}:
        raise ValueError("Run current synthetic end-to-end smoke first")
    definition = register(study)
    write_json(ROOT/study["output"]/"readiness.json", {"status": "passed", "source_hashes": definition["source_hashes"],
               "checks": "2 CPU tests: real/Q zero-removal identity, same-mask physical compaction, substitution/fold, direct output-fit and unchanged other state",
               "scientific_results": False, "end_to_end_smoke": smoke, "updated_utc": now()})
    print("READY: tests passed; no scientific training yet", flush=True)


def report(study, rows, status):
    output = ROOT/study["output"]
    write_json(output/"summary.json", {"status": status, "config": study, "results": rows, "test_evaluated": False})
    lines = ["# ResNet20 internal-channel compression", "", f"Status: {status}; one seed2026.", "", "## Hypothesis", "",
             "At similar compact size, direct consumer-output fitting may retain more quality than shared-map folding after removing whole Q/scalar internal channels.", "",
             "## Measured results", "", "| Family | Fraction | Policy | Real trainable parameters | Validation before (%) | Validation after 3ep (%) |", "|---|---:|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['family']} | {row['fraction']:.2f} | {row['policy']} | {row['parameters']} | {100*row['before']['validation']['accuracy']:.2f} | {100*row['after']['validation']['accuracy']:.2f} |")
    lines += ["", "## Interpretation and limits", "", study["scope"], "", study["resume"], "",
              "One discovery seed only: no significance, generalization, quaternion superiority or methodological novelty claim. Identical fractions and same masks within a family, not identical masks across families. Q units group four components while real units are scalar; report real parameters and baseline quality. The final real classifier and residual outputs/shortcuts are unchanged by pruning. 3ep supervised repair uses all surviving parameters; no test. Previous He2017ChannelPruning, Kim2020, VieiraValle2022ELM and Singh2026 remain prior art. No automatic replication or PHM run."]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(study):
    output = ROOT/study["output"]
    definition = register(study)
    ready = curve.read(output/"readiness.json")
    if ready["status"] != "passed" or ready["source_hashes"] != definition["source_hashes"]: raise ValueError("Readiness mismatch")
    with lock("compression_curve_runner.lock"):
        rows = []
        try:
            seed_everything(study["seed"])
            device = torch.device(study["device"])
            baseline = curve.read(ROOT/study["baseline_config"])
            images, labels, mean, std, _ = curve.load_data(curve.read(ROOT/"configs/compression_curve.json"), device)
            with np.load(ROOT/baseline["split_source"]) as split:
                ids = {"train": split["train"], "validation": split["validation"], "fit": split["calibration"][::2], "check_calibration": split["calibration"][1::2]}
            sets = [set(v) for v in ids.values()]
            if any(a & b for i, a in enumerate(sets) for b in sets[i+1:]): raise AssertionError("Data splits overlap")
            fit_images = prepare_batch(torch.as_tensor(images[ids["fit"]], device=device), mean, std)
            check_images = prepare_batch(torch.as_tensor(images[ids["check_calibration"]], device=device), mean, std)
            tensors = {s: torch.as_tensor(images[ids[s]], device=device) for s in ("train", *SPLITS)}
            targets = {s: torch.as_tensor(labels[ids[s]], device=device) for s in tensors}
            for family in study["families"]:
                baseline_path = ROOT/"results/resnet20_baselines"/family/"model.pt"
                original = make_model(baseline, family).to(device).eval()
                original.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=True)["model"])
                for fraction in study["fractions"]:
                    group = output/family/f"f_{round(1000*fraction):03d}"
                    for policy in study["policies"]:
                        folder = group/policy
                        folder.mkdir(parents=True, exist_ok=True)
                        if (folder/"result.json").exists():
                            row = curve.read(folder/"result.json")
                            if file_sha256(folder/"after.pt") != row["after_sha256"]: raise ValueError("Completed model changed")
                            rows.append(row); continue
                        current = deepcopy(original)
                        mask_rows = []
                        for index in range(9):
                            if (output/"STOP").exists():
                                report(study, rows, "paused")
                                write_json(output/"state.json", {"status": "paused", "family": family, "fraction": fraction, "policy": policy, "site": index, "updated_utc": now()}); return
                            site_dir = folder/"sites"/f"site_{index:02d}"
                            site_dir.mkdir(parents=True, exist_ok=True)
                            checkpoint, record = site_dir/"model.pt", site_dir/"metrics.json"
                            if record.exists():
                                site_row = curve.read(record)
                                if file_sha256(checkpoint) != site_row["checkpoint_sha256"]: raise ValueError("Saved site changed")
                                current = load_model(checkpoint, baseline, family, device)
                                mask_rows.append(site_row); continue
                            write_json(output/"state.json", {"status": "running", "pid": os.getpid(), "family": family, "fraction": fraction, "policy": policy, "site": index, "completed_cases": len(rows), "updated_utc": now()})
                            print(f"{family} f={fraction} {policy}: site {index+1}/9", flush=True)
                            fit = collect(current, index, fit_images, study["calibration_batch"])
                            count = round(current.blocks[index].conv1.out_channels*fraction)
                            if policy == "mapfold": victims, maps = choose(current.blocks[index], fit[0], count, study["alpha"])
                            else:
                                frozen = curve.read(group/"mapfold"/"sites"/f"site_{index:02d}"/"metrics.json")
                                victims = frozen["victims"]
                                q = family == "quaternion"
                                maps = site_maps(moments(fit[0], q), victims, study["alpha"], "quaternion_right" if q else "real")
                            folded, kept = compact(current, index, victims, maps)
                            replay = fold_error(current, folded, index, maps, check_images[:16])
                            if replay > .001: raise AssertionError("Residual fold equivalence failed")
                            audit = None
                            if policy == "pruned": new, _ = compact(current, index, victims)
                            elif policy == "direct":
                                check = collect(current, index, check_images, study["calibration_batch"])
                                new, audit = refit(current, folded, index, kept, fit, check, study["alpha"])
                            else: new = folded
                            site_row = {"index": index, "victims": victims, "kept": kept, "fold_max_abs_error": replay, "direct_audit": audit}
                            site_row["checkpoint_sha256"] = save_model(new, checkpoint)
                            write_json(record, site_row)
                            mask_rows.append(site_row)
                            current = new
                        before = {s: evaluate(current, tensors[s], targets[s], mean, std, 128)[0] for s in SPLITS}
                        before_hash = save_model(current, folder/"before.pt")
                        probe = deepcopy(current)
                        audit = first_update_audit(probe, prepare_batch(tensors["train"][:32], mean, std), targets["train"][:32], torch.optim.AdamW(probe.parameters(), lr=.0001))
                        del probe
                        def epoch_done(row, counts):
                            write_json(output/"state.json", {"status": "running", "pid": os.getpid(), "family": family, "fraction": fraction, "policy": policy, "epoch": row["epoch"], "completed_cases": len(rows), "updated_utc": now()})
                            print(f"{family} {fraction} {policy} repair {row['epoch']}/3: val={100*row['validation']['accuracy']:.2f}%", flush=True)
                        current, history, counts, done = repair(current, tensors, targets, mean, std, study["finetune"], study["seed"], folder/"resume.pt", output/"STOP", epoch_done)
                        if not done:
                            report(study, rows, "paused")
                            write_json(output/"state.json", {"status": "paused", "family": family, "fraction": fraction, "policy": policy, "updated_utc": now()}); return
                        after_hash = save_model(current, folder/"after.pt")
                        replay = load_model(folder/"after.pt", baseline, family, device)
                        measured = evaluate(replay, tensors["validation"], targets["validation"], mean, std, 128)[0]
                        if measured["accuracy"] != history[-1]["validation"]["accuracy"] or abs(measured["cross_entropy"]-history[-1]["validation"]["cross_entropy"]) > 1e-6: raise AssertionError("Checkpoint replay mismatch")
                        row = {"family": family, "fraction": fraction, "policy": policy, "parameters": sum(p.numel() for p in current.parameters()),
                               "hidden_widths": widths(current), "before": before, "after": {s: history[-1][s] for s in SPLITS}, "history": history,
                               "counts": counts, "site_records": mask_rows, "first_update": audit, "before_sha256": before_hash, "after_sha256": after_hash, "test_evaluated": False}
                        write_json(folder/"result.json", row)
                        rows.append(row)
                        report(study, rows, "running")
            register(study)
            report(study, rows, "completed")
            write_json(output/"state.json", {"status": "completed", "completed_cases": len(rows), "updated_utc": now(), "test_evaluated": False})
        except BaseException as error:
            write_json(output/"state.json", {"status": "failed", "error": f"{type(error).__name__}: {error}", "completed_cases": len(rows), "updated_utc": now()})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true"); modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    study = curve.read(ROOT/CONFIG)
    torch.set_num_threads(study["cpu_threads"])
    if args.run: run(study)
    else: prepare(study)
