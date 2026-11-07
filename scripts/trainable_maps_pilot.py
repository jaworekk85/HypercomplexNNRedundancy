"""One fixed two-arm CE pilot. Default prepares; --run starts/resumes epochs."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import platform

import numpy as np
import torch

from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from scripts.paper_a_controls import now, rel
from scripts.run_activation_queue import lock
from scripts.tail_compensation_pilot import intermediate
from src.cnn_training import evaluate, first_update_audit, prepare_batch, seed_everything
from src.compression.structure_cost import describe, make_model
from src.compression.trainable_maps import attach, deploy
from src.data.cifar_cached import file_sha256
from src.trainable_map_repair import atomic_save, repair

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/trainable_maps_pilot.json"
NEW = [CONFIG, "scripts/trainable_maps_pilot.py", "src/compression/trainable_maps.py",
       "src/trainable_map_repair.py", "tests_torch/test_trainable_maps.py"]
SPLITS = ("validation", "check_calibration")


def register(study):
    parent = curve.read(ROOT/study["parent_protocol"])
    for name, digest in {**parent["source_hashes"], **parent["input_hashes"]}.items():
        if file_sha256(ROOT/name) != digest: raise ValueError(f"Frozen parent changed: {name}")
    extra = [ROOT/study["parent_protocol"], ROOT/study["pilot"]/"direct_before.pt",
             ROOT/study["reference_group"]/"phase_a.json"]
    protocol = {"config": study,
                "source_hashes": {**parent["source_hashes"], **{p: file_sha256(ROOT/p) for p in NEW}},
                "input_hashes": {**parent["input_hashes"], **{rel(p): file_sha256(p) for p in extra}},
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "numpy": np.__version__, "cuda": torch.version.cuda},
                "prior_art": study["prior_art"], "test_evaluated": False}
    output = ROOT/study["output"]
    output.mkdir(parents=True, exist_ok=True)
    path = output/"protocol.json"
    if path.exists() and curve.read(path) != protocol: raise ValueError("Registered study changed")
    write_json(path, protocol)
    return protocol


def starting_model(study, device):
    phase = curve.read(ROOT/study["reference_group"]/"phase_a.json")
    source, victims = phase["source"], phase["frozen_victims"]
    compact = make_model(curve.reduced_config(source["config"], "cnn6", study["fraction"]), "cifar", "bn", "quaternion").to(device).eval()
    compact.load_state_dict(torch.load(ROOT/study["pilot"]/"direct_before.pt", map_location=device, weights_only=True)["model"])
    reference = intermediate(study, source, victims, 3, device)
    kernel = reference.blocks[2][3].kernel.detach()
    outputs = sorted(set(range(kernel.shape[1])) - set(victims[5]))
    c = kernel.index_select(1, torch.tensor(outputs, device=device)).index_select(2, torch.tensor(victims[4], device=device)).clone()
    return compact, c, source, victims


def audit(model, images, targets, lr):
    probe = deepcopy(model)
    layer = probe.blocks[2][3]
    before = layer.map_delta.detach().clone()
    c = layer.removed_kernel.detach().clone()
    result = first_update_audit(probe, images, targets, torch.optim.AdamW(probe.parameters(), lr=lr))
    if isinstance(layer.map_delta, torch.nn.Parameter):
        gradients = layer.map_delta.grad.flatten(1).norm(dim=1)
        updates = (layer.map_delta.detach()-before).flatten(1).norm(dim=1)
        if not bool((gradients > 0).all() and (updates > 0).all()): raise AssertionError("Map component audit failed")
        result["map_component_gradient_norms"] = gradients.tolist()
        result["map_component_update_norms"] = updates.tolist()
    if not torch.equal(c, layer.removed_kernel): raise AssertionError("Frozen C changed")
    return result


@torch.no_grad()
def export_check(model, images):
    previous = model.training
    model.eval()
    compact = deploy(model).eval()
    architecture = describe(compact)
    if architecture["real_parameters"] != 19098 or architecture["quaternion_convolutions"] != 6 or architecture["dense_real_convolutions"]:
        raise AssertionError("Deployed family/size changed")
    error = float((model(images)-compact(images)).abs().max())
    if error > .0001: raise AssertionError(f"Exact map fold failed: {error}")
    model.train(previous)
    return compact, {"max_abs_logit_error": error, "architecture": architecture}


def prepare(study):
    # Technical discarded-copy updates only; no dataset, held-out outcome or scientific training.
    seed_everything(study["seed"])
    model, c, _, victims = starting_model(study, torch.device("cpu"))
    x, y = torch.randn(8, 3, 32, 32), torch.arange(8) % 10
    checks = {}
    for case in study["cases"]:
        candidate = attach(model, c, case == "learned_maps")
        count = sum(p.numel() for p in candidate.parameters())
        expected = 20122 if case == "learned_maps" else 19098
        if count != expected: raise AssertionError("Unexpected training budget")
        update = audit(candidate, x, y, study["finetune"]["learning_rate"])
        _, folding = export_check(candidate, x)
        with torch.no_grad(): initial_error = float((model(x)-candidate.eval()(x)).abs().max())
        if initial_error != 0: raise AssertionError("Starting function changed")
        checks[case] = {"trainable_reals": count, "frozen_C_reals": c.numel(),
                        "map_storage_reals": candidate.blocks[2][3].map_delta.numel(),
                        "initial_max_abs_logit_error": initial_error, "update": update, "export": folding}
    protocol = register(study)
    output = ROOT/study["output"]
    write_json(output/"readiness.json", {"status": "passed", "scientific_results": False,
              "checks": checks, "frozen_victims": victims, "source_hashes": protocol["source_hashes"], "updated_utc": now()})
    if not (output/"state.json").exists():
        write_json(output/"state.json", {"status": "ready_not_started", "scientific_results": False, "updated_utc": now()})
    print("READY: two cases, no scientific training performed", flush=True)


def report(study, rows, status):
    output = ROOT/study["output"]
    write_json(output/"summary.json", {"status": status, "config": study, "cases": rows, "test_evaluated": False})
    lines = ["# Trainable Q-map correction pilot", "", f"Status: {status}. CIFAR BN seed2026, f=.5, ONE late consumer.", "",
             "## Hypothesis", "", "Temporary jointly learned quaternion corrections may change optimization at the same deployed compact size. Exact predecessor not located; no novelty or priority claim.", "",
             "## Measured results", "", "| Case | Training reals | Deployment reals | Validation before (%) | Validation epoch3 (%) |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['case']} | {row['training_reals']} | 19098 | {100*row['before']['validation']['accuracy']:.2f} | {100*row['after']['validation']['accuracy']:.2f} |")
    if not rows: lines.append("No scientific case completed yet.")
    lines += ["", "## Interpretation and limits", "",
              "Two new matched three-epoch CE repairs, fixed masks and identical direct-Q initialization. A corrections start at zero; B and all ordinary parameters train, C remains frozen. DeltaA has 1024 temporary real trainables; C stores 9216 frozen reals and the product basis 64. Control retains the same buffers with zero fixed DeltaA. Both export to standard six-Q-convolution models with 19098 trainable reals. BN/dropout follow the existing recipe. No teacher distillation, test evaluation, LR tuning or automatic replication.", "",
              "Single previously studied discovery source: no independent confirmation, significance or superiority claim. Joint A+B changes parameterization within the same compact Q functional family. AdamW and correction-centered decay are specified choices; plain-SGD preconditioning analysis does not describe AdamW exactly. Broad learned-compensator/folding prior art: Ding2021ResRep, Zhang2026RCP; hypercomplex adapters: Mahabadi2021Compacter; optimizer reparameterization: Ding2023RepOptimizers.", "",
              "Atomic completed-epoch model/optimizer/scheduler/scaler/RNG checkpoints. STOP checked after each batch; an interrupted partial epoch is replayed from its beginning. Remove STOP only when resuming. Schedule follows attempted AMP steps; successful steps recorded separately. Final epoch is fixed; validation never selects a checkpoint."]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(study):
    output = ROOT/study["output"]
    register(study)
    readiness = curve.read(output/"readiness.json")
    if readiness["status"] != "passed" or readiness["source_hashes"] != curve.read(output/"protocol.json")["source_hashes"]:
        raise ValueError("Readiness does not cover current sources")
    with lock("compression_curve_runner.lock"):
        rows = []
        device = torch.device(study["device"])
        seed_everything(study["seed"])
        data = curve.read(ROOT/"configs/compression_curve.json")
        images, labels, mean, std, _ = curve.load_data(data, device)
        initial, c, source, _ = starting_model(study, device)
        ids = curve.split_ids(source, None)
        sets = [set(ids[s]) for s in ("train", "validation", "fit", "check_calibration")]
        if any(a & b for i, a in enumerate(sets) for b in sets[i+1:]): raise AssertionError("Data splits overlap")
        tensors = {s: torch.as_tensor(images[ids[s]], device=device) for s in ("train", *SPLITS)}
        targets = {s: torch.as_tensor(labels[ids[s]], device=device) for s in tensors}
        x = prepare_batch(tensors["check_calibration"][:32], mean, std)
        try:
            for case in study["cases"]:
                folder = output/case
                folder.mkdir(parents=True, exist_ok=True)
                if (folder/"result.json").exists():
                    row = curve.read(folder/"result.json")
                    if file_sha256(ROOT/row["checkpoint"]) != row["checkpoint_sha256"]: raise ValueError("Completed checkpoint changed")
                    rows.append(row)
                    continue
                if (output/"STOP").exists():
                    write_json(output/"state.json", {"status": "paused", "case": case, "updated_utc": now()})
                    report(study, rows, "paused")
                    return
                model = attach(initial, c, case == "learned_maps")
                before = {s: evaluate(model, tensors[s], targets[s], mean, std, 128)[0] for s in SPLITS}
                first = audit(model, prepare_batch(tensors["train"][:32], mean, std), targets["train"][:32], study["finetune"]["learning_rate"])
                training_reals = sum(p.numel() for p in model.parameters())
                frozen_c = model.blocks[2][3].removed_kernel.clone()
                def epoch_done(row, count):
                    write_json(output/"state.json", {"status": "running", "case": case, "completed_epoch": row["epoch"], "pid": os.getpid(), "counts": count, "updated_utc": now()})
                    print(f"{case} epoch {row['epoch']}/3 val={100*row['validation']['accuracy']:.2f}% time={row['epoch_seconds']:.1f}s steps={count}", flush=True)
                write_json(output/"state.json", {"status": "running", "case": case, "pid": os.getpid(), "updated_utc": now()})
                print(f"START {case}: {training_reals} training reals, 19098 deployed", flush=True)
                model, history, counts, done = repair(model, tensors, targets, mean, std, study["finetune"], study["seed"], folder/"resume.pt", output/"STOP", epoch_done)
                if not done:
                    write_json(output/"state.json", {"status": "paused", "case": case, "completed_epoch": len(history), "updated_utc": now()})
                    report(study, rows, "paused")
                    return
                if not torch.equal(frozen_c, model.blocks[2][3].removed_kernel): raise AssertionError("C changed during repair")
                compact, folded = export_check(model, x)
                after = {s: evaluate(compact, tensors[s], targets[s], mean, std, 128)[0] for s in SPLITS}
                for s in SPLITS:
                    if abs(after[s]["accuracy"]-history[-1][s]["accuracy"]) > 1e-12: raise AssertionError("Export changed predictions")
                atomic_save({"model": {k: v.cpu() for k, v in compact.state_dict().items()}}, folder/"compact_after.pt")
                row = {"case": case, "before": before, "after": after, "history": history,
                       "training_reals": training_reals, "deployed_reals": 19098, "counts": counts,
                       "discarded_copy_update_audit": first, "fold": folded,
                       "checkpoint": rel(folder/"compact_after.pt"), "checkpoint_sha256": file_sha256(folder/"compact_after.pt"), "test_evaluated": False}
                write_json(folder/"result.json", row)
                rows.append(row)
                report(study, rows, "running")
                del model, compact
            register(study)
            report(study, rows, "completed")
            write_json(output/"state.json", {"status": "completed", "completed_cases": len(rows), "completed_utc": now(), "test_evaluated": False})
        except BaseException as error:
            write_json(output/"state.json", {"status": "failed", "error": f"{type(error).__name__}: {error}", "updated_utc": now()})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    study = curve.read(ROOT/CONFIG)
    torch.set_num_threads(study["cpu_threads"])
    if args.run: run(study)
    else: prepare(study)
