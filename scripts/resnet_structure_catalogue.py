"""Bounded hard Q/dense/variable-n PHM selection after local pruning.

--prepare is synthetic CPU readiness only. --run checks completed predecessor
studies before any scientific GPU work. This does not implement those studies.
"""
import argparse
from copy import deepcopy
import math
from pathlib import Path

import torch

from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from scripts.paper_a_controls import now, rel
from scripts.run_activation_queue import lock
from scripts.resnet20_baselines import make_model
from src.cnn_training import evaluate, first_update_audit, prepare_batch, seed_everything
from src.compression.quaternion_channels import component_indices
from src.compression.structure_catalogue import (architecture, candidates, collect, mse, mutable_count,
    narrow_producer, parameter_count, raw_moments, restore, select)
from src.data.cifar_cached import file_sha256
from src.models.cifar_resnet import CifarResNet20
from src.models.quaternion_conv_torch import QuaternionConv2d, hamilton_kernel
from src.trainable_map_repair import atomic_save, repair

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/resnet_structure_catalogue.json"
NEW = [CONFIG, "scripts/resnet_structure_catalogue.py", "src/models/phm_conv.py", "src/compression/structure_catalogue.py", "tests_torch/test_structure_catalogue.py", "scripts/smoke_structure_catalogue.py"]
SPLITS = ("validation", "check_calibration")


def register(study):
    parent = curve.read(ROOT/study["parent_protocol"])
    for name, digest in {**parent["source_hashes"], **parent["input_hashes"]}.items():
        if file_sha256(ROOT/name) != digest: raise ValueError(f"Frozen parent changed: {name}")
    definition = {"config": study, "source_hashes": {**parent["source_hashes"], **{p: file_sha256(ROOT/p) for p in NEW}},
                  "input_hashes": {**parent["input_hashes"], study["parent_protocol"]: file_sha256(ROOT/study["parent_protocol"])},
                  "torch": torch.__version__, "prior_art": study["prior_art"]}
    output = ROOT/study["output"]
    output.mkdir(parents=True, exist_ok=True)
    path = output/"protocol.json"
    if path.exists() and curve.read(path) != definition: raise ValueError("Structure catalogue protocol changed")
    write_json(path, definition)
    return definition


def check_dependencies(study):
    paths = [ROOT/path for path in study["required_completed_studies"]]
    baseline = ROOT/study["baseline_checkpoint"]
    paths += [ROOT/study["baseline_metrics"], baseline.parent.parent/"summary.json"]
    for path in paths:
        if not path.exists(): raise RuntimeError(f"Stage awaits prerequisite: {rel(path)}")
        row = curve.read(path)
        if "status" in row and row["status"] != "completed": raise RuntimeError(f"Incomplete prerequisite: {rel(path)}")
    metrics = curve.read(ROOT/study["baseline_metrics"])
    if file_sha256(baseline) != metrics["checkpoint_sha256"]: raise ValueError("Q baseline checkpoint changed")
    paths.append(baseline)
    hashes = {rel(p): file_sha256(p) for p in paths}
    path = ROOT/study["output"]/"scientific_inputs.json"
    if path.exists() and curve.read(path) != hashes: raise ValueError("Scientific inputs changed")
    write_json(path, hashes)


def reference_budget(source, blocks, fraction):
    fixed = parameter_count(source)-sum(mutable_count(source.blocks[i]) for i in blocks)
    local = 0
    for i in blocks:
        block = source.blocks[i]
        width = max(1, math.ceil(block.conv1.out_channels*(1-fraction)))
        reduced = narrow_producer(block, list(range(width)))
        reduced.conv2 = QuaternionConv2d(width, block.outputs)
        local += mutable_count(reduced)
    return fixed, local


def policies(study, catalogues, budget):
    result = {}
    for policy in study["policies"]:
        subset = [[row for row in catalogue if
                   policy in ("variable_n", "weight_only_selector") or
                   policy == "all_q" and row["family"] == "q" or
                   policy == "q_real" and row["family"] in ("q", "real") or
                   policy == "fixed_phm4" and row["family"] == "phm" and row["n"] == 4] for catalogue in catalogues]
        try: result[policy] = {"status": "feasible", **select(subset, budget, "weight_error" if policy == "weight_only_selector" else "error")}
        except ValueError as error: result[policy] = {"status": "infeasible", "reason": str(error)}
    return result


def assemble(source, blocks, catalogues, choices):
    result = deepcopy(source).cpu().eval()
    for index, catalogue, identity in zip(blocks, catalogues, choices):
        result.blocks[index] = deepcopy(next(c for c in catalogue if c["id"] == identity)["block"])
    return result


def prepare(study):
    smoke = curve.read(ROOT/".cache/structure_catalogue_end_to_end/last_readiness.json")
    if smoke["status"] != "passed" or smoke["scientific_results"] or smoke["source_hashes"] != {p: file_sha256(ROOT/p) for p in NEW}:
        raise ValueError("Run current synthetic end-to-end CPU smoke first")
    seed_everything(study["seed"])
    config = {"real_widths": [4, 8, 16], "quaternion_widths": [2, 4, 8], "classes": 10}
    source = CifarResNet20("quaternion", **config).eval()
    images = torch.randn(8, 3, 8, 8)
    catalogues = []
    for index in study["blocks"]:
        pack = collect(source, index, images, 4)
        width = max(1, source.blocks[index].conv1.out_channels//2)
        rows, _ = candidates(source.blocks[index], pack, [max(1, width//2), width], study["dimensions"], study["ridge_alpha"])
        catalogues.append(rows)
    fixed, local = reference_budget(source, study["blocks"], .5)
    selected = policies(study, catalogues, local)
    for row in selected.values():
        if row["status"] != "feasible": continue
        model = assemble(source, study["blocks"], catalogues, row["choices"])
        if parameter_count(model) != fixed+row["cost"] or parameter_count(model) > fixed+local: raise AssertionError("Actual storage exceeds selected budget")
        replica = restore(architecture(model, config)).eval()
        replica.load_state_dict(model.state_dict())
        with torch.no_grad(): torch.testing.assert_close(model(images), replica(images), atol=0, rtol=0)
    # Force a mixed PHM/dense graph, even if a fit-based policy selects Q only.
    choices = [next(c["id"] for c in catalogues[0] if c["family"] == "phm" and c["n"] == 4),
               next(c["id"] for c in catalogues[1] if c["family"] == "real")]
    mixed = assemble(source, study["blocks"], catalogues, choices)
    audit = first_update_audit(mixed, images, torch.arange(8) % 10, torch.optim.AdamW(mixed.parameters(), lr=.0001))
    definition = register(study)
    write_json(ROOT/study["output"]/"readiness.json", {"status": "passed", "scientific_results": False,
              "source_hashes": definition["source_hashes"], "checks": "6 CPU tests, physical mixed-model reload/budget and forced mixed-graph update audit",
              "synthetic_policy_results": selected, "synthetic_update_audit": audit, "end_to_end_smoke": smoke, "updated_utc": now()})
    print("READY: synthetic CPU checks only; scientific predecessors still required", flush=True)


def report(study, results, selections, budget, status):
    output = ROOT/study["output"]
    write_json(output/"summary.json", {"status": status, "config": study, "results": results, "selections": selections,
               "global_budget": budget, "test_evaluated": False})
    lines = ["# Two-block Q/dense/PHM structure selection", "", f"Status: {status}; seed2026, two local internal-channel pruning sites.", "",
             "## Hypothesis", "", "Hard variable-n PHM selection may improve quality per stored parameter beyond a Q/real-only hybrid; not established novelty or gain.", "",
             "## Measured results", "", "| Policy | Actual trainable/stored reals | Budget slack | Validation before (%) | Validation after (%) |", "|---|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['policy']} | {row['real_parameters']} | {budget-row['real_parameters']} | {100*row['before']['validation']['accuracy']:.2f} | {100*row['after']['validation']['accuracy']:.2f} |")
    lines += ["", "## Interpretation and limits", "", study["scope"], "", study["method"], "",
              "Local independent candidate errors do not guarantee global classification ordering; earlier replacements alter later inputs. No validation/check-based type tuning. Infeasible policies remain explicit. <=common budget is not exact parameter equality; report slack and actual counts. Generic SVD/DP/ridge are established; no firstness claim. Layerwise PHM factors remain deployed and counted. Same3ep CE repairs; selected final epoch, no test. Epoch checkpoints and STOP support existing checked resume semantics."]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(study):
    definition = register(study)
    output = ROOT/study["output"]
    ready = curve.read(output/"readiness.json")
    if ready["status"] != "passed" or ready["source_hashes"] != definition["source_hashes"]: raise ValueError("Readiness changed")
    check_dependencies(study)
    with lock("compression_curve_runner.lock"):
        results, selections, budget = [], {}, None
        try:
            seed_everything(study["seed"])
            device = torch.device(study["device"])
            baseline_config = curve.read(ROOT/study["baseline_config"])
            source = make_model(baseline_config, "quaternion").eval()
            source.load_state_dict(torch.load(ROOT/study["baseline_checkpoint"], map_location="cpu", weights_only=True)["model"])
            images, labels, mean, std, _ = curve.load_data(curve.read(ROOT/"configs/compression_curve.json"), device)
            import numpy as np
            with np.load(ROOT/baseline_config["split_source"]) as split:
                ids = {"train": split["train"], "validation": split["validation"], "fit": split["calibration"][::2], "check_calibration": split["calibration"][1::2]}
            sets = [set(v) for v in ids.values()]
            if any(a & b for i, a in enumerate(sets) for b in sets[i+1:]): raise AssertionError("Overlapping splits")
            fit_images = prepare_batch(torch.as_tensor(images[ids["fit"]], device=device), mean, std)
            check_images = prepare_batch(torch.as_tensor(images[ids["check_calibration"]], device=device), mean, std)
            catalogues = []
            source_gpu = deepcopy(source).to(device)
            for index in study["blocks"]:
                if (output/"STOP").exists():
                    write_json(output/"state.json", {"status": "paused", "stage": "catalogue", "updated_utc": now()}); return
                path = output/f"catalogue_block_{index}.pt"
                meta_path = output/f"catalogue_block_{index}.json"
                if path.exists() and meta_path.exists():
                    if file_sha256(path) != curve.read(meta_path)["sha256"]: raise ValueError("Catalogue changed")
                    rows = torch.load(path, map_location="cpu", weights_only=False)
                else:
                    fit = collect(source_gpu, index, fit_images)
                    check = collect(source_gpu, index, check_images)
                    reference_width = max(1, math.ceil(source.blocks[index].conv1.out_channels*(1-study["reference_removed_fraction"])))
                    widths = sorted({max(1, round(reference_width*s)) for s in study["width_scales"]})
                    rows, skipped = candidates(source.blocks[index], fit, widths, study["dimensions"], study["ridge_alpha"])
                    cached_check = {}
                    for row in rows:
                        width = len(row["kept"])
                        scalar = component_indices(row["kept"], source.blocks[index].conv1.out_channels)
                        if width not in cached_check: cached_check[width] = raw_moments(check[0][:, scalar], check[1])
                        layer = row["block"].conv2
                        weight = hamilton_kernel(layer.kernel) if row["family"] == "q" else layer.effective_kernel() if row["family"] == "phm" else layer.weight
                        row["check_mse"] = mse(weight.detach(), cached_check[width])
                    atomic_save(rows, path)
                    write_json(meta_path, {"sha256": file_sha256(path), "candidates": [{k: v for k, v in row.items() if k != "block"} for row in rows], "skipped": skipped})
                catalogues.append(rows)
            del source_gpu
            fixed, local_budget = reference_budget(source, study["blocks"], study["reference_removed_fraction"])
            budget = fixed+local_budget
            selections = policies(study, catalogues, local_budget)
            tensors = {s: torch.as_tensor(images[ids[s]], device=device) for s in ("train", *SPLITS)}
            targets = {s: torch.as_tensor(labels[ids[s]], device=device) for s in tensors}
            for policy, selected in selections.items():
                if selected["status"] != "feasible": continue
                if (output/"STOP").exists():
                    report(study, results, selections, budget, "paused")
                    write_json(output/"state.json", {"status": "paused", "policy": policy, "updated_utc": now()}); return
                folder = output/policy
                folder.mkdir(parents=True, exist_ok=True)
                if (folder/"result.json").exists():
                    row = curve.read(folder/"result.json")
                    if file_sha256(folder/"compact_after.pt") != row["checkpoint_sha256"]: raise ValueError("Completed model changed")
                    results.append(row); continue
                model = assemble(source, study["blocks"], catalogues, selected["choices"]).to(device)
                count = parameter_count(model)
                if count != fixed+selected["cost"] or count > budget: raise AssertionError("Physical budget mismatch")
                spec = architecture(model, baseline_config)
                before = {s: evaluate(model, tensors[s], targets[s], mean, std, 128)[0] for s in SPLITS}
                probe = deepcopy(model)
                audit = first_update_audit(probe, prepare_batch(tensors["train"][:32], mean, std), targets["train"][:32], torch.optim.AdamW(probe.parameters(), lr=.0001))
                del probe
                atomic_save({"model": {n: v.detach().cpu() for n, v in model.state_dict().items()}, "architecture": spec}, folder/"compact_before.pt")
                def epoch_done(row, counts):
                    write_json(output/"state.json", {"status": "running", "policy": policy, "epoch": row["epoch"], "counts": counts, "updated_utc": now()})
                    print(f"{policy} epoch{row['epoch']}: val={100*row['validation']['accuracy']:.2f}%", flush=True)
                model, history, counts, done = repair(model, tensors, targets, mean, std, study["finetune"], study["seed"], folder/"resume.pt", output/"STOP", epoch_done)
                if not done:
                    report(study, results, selections, budget, "paused")
                    write_json(output/"state.json", {"status": "paused", "policy": policy, "updated_utc": now()}); return
                atomic_save({"model": {n: v.detach().cpu() for n, v in model.state_dict().items()}, "architecture": spec}, folder/"compact_after.pt")
                replica = restore(spec).to(device).eval()
                replica.load_state_dict(torch.load(folder/"compact_after.pt", map_location=device, weights_only=True)["model"])
                replay = evaluate(replica, tensors["validation"], targets["validation"], mean, std, 128)[0]
                if replay["accuracy"] != history[-1]["validation"]["accuracy"] or abs(replay["cross_entropy"]-history[-1]["validation"]["cross_entropy"]) > 1e-6: raise AssertionError("Mixed checkpoint replay failed")
                row = {"policy": policy, "choices": selected["choices"], "real_parameters": count, "before": before, "after": {s: history[-1][s] for s in SPLITS},
                       "history": history, "counts": counts, "architecture": spec, "checkpoint_sha256": file_sha256(folder/"compact_after.pt"),
                       "discarded_update_audit": audit, "test_evaluated": False}
                write_json(folder/"result.json", row)
                results.append(row)
                report(study, results, selections, budget, "running")
                del model, replica
            register(study); check_dependencies(study)
            report(study, results, selections, budget, "completed")
            write_json(output/"state.json", {"status": "completed", "completed_policies": len(results), "updated_utc": now()})
        except BaseException as error:
            write_json(output/"state.json", {"status": "failed", "error": f"{type(error).__name__}: {error}", "updated_utc": now()}); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true"); mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    study = curve.read(ROOT/CONFIG)
    torch.set_num_threads(study["cpu_threads"])
    if args.run: run(study)
    else: prepare(study)
