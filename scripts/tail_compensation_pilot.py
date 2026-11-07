"""One fixed late-consumer logit-refinement pilot, with an equal-step local control."""
from copy import deepcopy
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from scripts.paper_a_controls import now, rel
from scripts.run_activation_queue import lock
from src.cnn_training import first_update_audit, prepare_batch, seed_everything
from src.compression import multilayer as old
from src.compression.structure_cost import describe, make_model
from src.compression.tail_compensation import capture, refine
from src.data.cifar_cached import file_sha256
from src.finetuning import finetune

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/tail_compensation_pilot.json"
NEW = [CONFIG, "scripts/tail_compensation_pilot.py", "src/compression/tail_compensation.py", "tests_torch/test_tail_compensation.py"]


def freeze(study):
    parent = curve.read(ROOT / study["parent_protocol"])
    for name, expected in {**parent["source_hashes"], **parent["input_hashes"]}.items():
        if file_sha256(ROOT / name) != expected: raise ValueError(f"Frozen parent changed: {name}")
    inputs = [ROOT / study["parent_protocol"], ROOT / study["pilot"] / "summary.json"]
    for i in range(6):
        inputs += [ROOT / study["pilot"] / "sites" / f"site_{i:02d}" / name for name in ("model.pt", "metrics.json")]
    definition = {"config": study, "source_hashes": {**parent["source_hashes"], **{n: file_sha256(ROOT / n) for n in NEW}},
                  "input_hashes": {**parent["input_hashes"], **{rel(p): file_sha256(p) for p in inputs}},
                  "environment": {"torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda},
                  "prior_art": ["Singh2026", "Park2020Lookahead", "VieiraValle2022ELM"]}
    path = ROOT / study["output"] / "protocol.json"
    if path.exists() and curve.read(path) != definition: raise ValueError("Registered pilot changed")
    write_json(path, definition)


def intermediate(study, source, victims, last, device):
    model = make_model(source["config"], "cifar", "bn", "quaternion").to(device).eval()
    for i, site in enumerate(old.sites(model)):
        if i > last: break
        model, _, _ = old.compress_site(model, site, victims[i])
    path = ROOT / study["pilot"] / "sites" / f"site_{last:02d}" / "model.pt"
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model"])
    return model.eval()


def final_compaction(model, victims, fit, check, study):
    site = (2, 1)
    moments = old.SiteMoments(*old.site_gram(model, site, fit, 128), True)
    maps = old.site_maps(moments, victims[5], study["ridge_alpha"], "quaternion_right")
    compact, _, lost = old.compress_site(model, site, victims[5], maps)
    with torch.no_grad():
        error = float((compact(check[:32])-old.substituted_logits(model, site, maps, check[:32], 128)).abs().max())
    if error > .001: raise AssertionError("Real-head fold failed")
    return compact, {"site": [2, 1], "victims": victims[5], "maps": maps, "fold_max_abs_error": error, "projection_loss": lost}


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)
    return file_sha256(path)


def repair(model, tensors, targets, mean, std, study):
    probe = deepcopy(model)
    update = first_update_audit(probe, prepare_batch(tensors["train"][:32], mean, std), targets["train"][:32],
                                torch.optim.AdamW(probe.parameters(), lr=study["finetune"]["learning_rate"]))
    del probe
    count = {"successful_optimizer_steps": 0}
    original = torch.optim.AdamW
    class CountedAdamW(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.register_step_post_hook(lambda optimizer, args, kwargs: count.__setitem__("successful_optimizer_steps", count["successful_optimizer_steps"]+1))
    with patch.object(torch.optim, "AdamW", CountedAdamW):
        result, history = finetune(model, tensors, targets, mean, std, study["finetune"], study["seed"], ("validation", "check_calibration"))
    batch_size = study["finetune"]["batch_size"]
    return result, {"history": history, "final": {s: history[-1][s] for s in ("validation", "check_calibration")},
                    "discarded_copy_update_audit": update, **count,
                    "attempted_optimizer_steps": study["finetune"]["epochs"]*((len(tensors["train"])+batch_size-1)//batch_size)}


def report(study, results, status, mask_audit):
    pilot = curve.read(ROOT / study["pilot"] / "summary.json")
    references = {"map_fold": pilot["references"]["q_comp"], "direct_q": {"before": pilot["before"], "after": pilot["after"]}}
    summary = {"status": status, "config": study, "results": results, "references": references,
               "mask_and_reused_finalization_audit": mask_audit, "scientific_results": study["scientific_results"], "test_evaluated": False}
    output = ROOT / study["output"]
    write_json(output / "summary.json", summary)
    lines = ["# Tail-sensitive quaternion compensation: one late-consumer pilot", "", "## Hypothesis", "", study["hypothesis"], "",
             "## Measured results", "", f"Status: {status}. Scientific results: {study['scientific_results']}. CIFAR BN seed2026, .5 unit removal/site, validation only.",
             "Reuse sites0-3 and the site4 direct-Q fit; ONE last Q consumer is refined. This is not a five-site implementation or a new baseline. Same victims and compact budget. Two new 200-step unlabeled calibrations, followed by two matched supervised three-epoch repairs. Labels are excluded from calibration; check/validation never select steps or hyperparameters.", "",
             "| Case | Real parameters | Val before supervised repair (%) | Val after repair (%) |",
             "| --- | ---: | ---: | ---: |"]
    if study["scientific_results"]:
        q = references["map_fold"]
        lines += [f"| map_fold (reused; no extra calibration) | 19098 | {100*q['before']['evaluation']['validation']['accuracy']:.2f} | {100*q['after']['evaluation']['validation']['accuracy']:.2f} |",
                  f"| direct_q (reused; no extra calibration) | 19098 | {100*pilot['before']['validation']['accuracy']:.2f} | {100*pilot['after']['validation']['accuracy']:.2f} |"]
    for row in results:
        lines.append(f"| {row['case']} | {row['architecture']['real_parameters']} | {100*row['before']['validation']['accuracy']:.2f} | {100*row['after']['validation']['accuracy']:.2f} |")
    lines += ["", "### Calibration diagnostics BEFORE final head compaction", "",
              "| Case | Fit logit MSE initial/final | Check logit MSE initial/final | Fit local MSE initial/final | Check local MSE initial/final | Seconds |",
              "| --- | --- | --- | --- | --- | ---: |"]
    for row in results:
        a = row["calibration"]
        pairs = [f"{a['initial'][split][name]:.6g} / {a['final'][split][name]:.6g}" for name in ("logit_mse", "local_mse") for split in ("fit", "check")]
        lines.append(f"| {row['case']} | " + " | ".join(pairs) + f" | {a['seconds']:.2f} |")
    lines += ["", "## Interpretation limits", "",
              "One exploratory discovery seed/point/consumer; preserve unfavorable effects. A mean over minibatches or tests is not replication. Equal steps do not imply equal FLOPs/runtime: tail_logits differentiates the frozen BN/ReLU/pool/readout tail, local_extra only the consumer. The stem is cached in both. Ridge is centered at the same direct-Q kernel and is scaled by its squared norm; this differs from the earlier moment-centered direct solver. Final head uses the old map-fold treatment with the same victims, with maps recomputed for each path. Its weights may differ across paths, but it is never optimized during calibration.", "",
              "Distillation of pruned quaternion CNNs already exists (Singh2026 sec.IX-B); broader multi-layer distortion is also known (Park2020Lookahead). No first-Q-distillation, novel-principle, superiority-to-published-recipe or acceptance claim. This is a bounded implementation of one proposed late-layer refinement, not a full new method established by evidence. All six convolutional layers remain Q; scalar BN and the real ten-output head are unchanged architecture conventions. No new official-test inference or automatic extension.", "",
              "Sources/config/inputs frozen before scientific outcomes; per-case checkpoints resume completed calibration/repair. Two CPU checks and smoke outcomes are technical readiness, not scientific results. New repair records successful/attempted AMP steps; inherited scheduler advances on attempts, reused references lack successful-step instrumentation."]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(study):
    output = ROOT / study["output"]
    output.mkdir(parents=True, exist_ok=True)
    with lock("compression_curve_runner.lock"):
        if study["scientific_results"]: freeze(study)
        seed_everything(study["seed"])
        torch.set_num_threads(study["cpu_threads"])
        device = torch.device(study["device"])
        write_json(output / "state.json", {"status": "running", "pid": os.getpid(), "started_utc": now(), "test_evaluated": False})
        results = []
        try:
            data = curve.read(ROOT / "configs/compression_curve.json")
            images, labels, mean, std, sources = curve.load_data(data, device)
            source = next(s for s in sources if s["seed"] == study["seed"] and s["architecture"] == "cnn6")
            ids = curve.split_ids(source, study["limits"])
            if set(ids["fit"]) & set(ids["check_calibration"]): raise AssertionError("Fit/check overlap")
            phase = curve.read(ROOT / study["reference_group"] / "phase_a.json")
            victims = phase["frozen_victims"]
            teacher = intermediate(study, source, victims, 3, device)
            student = intermediate(study, source, victims, 4, device)
            fit_images = prepare_batch(torch.as_tensor(images[ids["fit"]], device=device), mean, std)
            check_images = prepare_batch(torch.as_tensor(images[ids["check_calibration"]], device=device), mean, std)
            fit, check = capture(teacher, student, fit_images, study["batch_size"]), capture(teacher, student, check_images, study["batch_size"])
            mask_audit = {"same_frozen_victims": victims, "fit_check_disjoint": True, "fit_images": len(fit_images), "check_images": len(check_images),
                          "cached_tail_fit_replay_error": fit["tail_replay_max_error"], "cached_tail_check_replay_error": check["tail_replay_max_error"]}
            if study["scientific_results"]:
                unchanged, _ = final_compaction(student, victims, fit_images, check_images, study)
                saved = make_model(curve.reduced_config(source["config"], "cnn6", .5), "cifar", "bn", "quaternion").to(device).eval()
                saved.load_state_dict(torch.load(ROOT / study["pilot"] / "direct_before.pt", map_location=device, weights_only=True)["model"])
                with torch.no_grad(): error = float((unchanged(check_images[:32])-saved(check_images[:32])).abs().max())
                if error > .001: raise AssertionError("Reused direct path could not be replayed")
                mask_audit["reused_direct_finalization_max_error"] = error
                del unchanged, saved
            del teacher
            tensors = {name: torch.as_tensor(images[ids[name]], device=device) for name in ("train", "validation", "check_calibration")}
            targets = {name: torch.as_tensor(labels[ids[name]], device=device) for name in tensors}
            for case in study["cases"]:
                folder = output / case
                folder.mkdir(parents=True, exist_ok=True)
                write_json(output / "state.json", {"status": "running", "case": case, "pid": os.getpid(), "completed_cases": len(results), "updated_utc": now(), "test_evaluated": False})
                if (folder / "result.json").exists():
                    row = curve.read(folder / "result.json")
                    for key in ("before_checkpoint", "after_checkpoint", "intermediate_checkpoint"):
                        if file_sha256(ROOT / row[key]) != row[key+"_sha256"]: raise ValueError("Completed output changed")
                    results.append(row); continue
                if (folder / "calibration.json").exists():
                    calibration = curve.read(folder / "calibration.json")
                    current = deepcopy(student)
                    if file_sha256(folder / "calibrated_site4.pt") != calibration["checkpoint_sha256"]: raise ValueError("Calibration checkpoint changed")
                    current.load_state_dict(torch.load(folder / "calibrated_site4.pt", map_location=device, weights_only=True)["model"])
                else:
                    current, calibration = refine(student, fit, check, case, study)
                    calibration["checkpoint_sha256"] = save_model(current, folder / "calibrated_site4.pt")
                    write_json(folder / "calibration.json", calibration)
                print(f"{case}: fit logit {calibration['initial']['fit']['logit_mse']:.6g}->{calibration['final']['fit']['logit_mse']:.6g}; check {calibration['initial']['check']['logit_mse']:.6g}->{calibration['final']['check']['logit_mse']:.6g}", flush=True)
                current, head = final_compaction(current, victims, fit_images, check_images, study)
                architecture = describe(current)
                if architecture["real_parameters"] != 19098 or architecture["quaternion_convolutions"] != 6 or architecture["dense_real_convolutions"]:
                    raise AssertionError("Compact family/size changed")
                before = curve.metrics(current, images, labels, ids, mean, std, data, device)
                before_hash = save_model(current, folder / "compact_before.pt")
                print(f"{case}: compact before repair val {100*before['validation']['accuracy']:.2f}%; starting matched repair", flush=True)
                current, repaired = repair(current, tensors, targets, mean, std, study)
                if describe(current) != architecture: raise AssertionError("Repair changed architecture")
                after_hash = save_model(current, folder / "compact_after.pt")
                row = {"case": case, "calibration": calibration, "final_head": head, "architecture": architecture, "before": before, "after": repaired["final"], "repair": repaired,
                       "before_checkpoint": rel(folder / "compact_before.pt"), "before_checkpoint_sha256": before_hash,
                       "after_checkpoint": rel(folder / "compact_after.pt"), "after_checkpoint_sha256": after_hash,
                       "intermediate_checkpoint": rel(folder / "calibrated_site4.pt"), "intermediate_checkpoint_sha256": calibration["checkpoint_sha256"], "test_evaluated": False}
                write_json(folder / "result.json", row)
                results.append(row)
                report(study, results, "running", mask_audit)
                print(f"DONE {case}: val {100*before['validation']['accuracy']:.2f}->{100*row['after']['validation']['accuracy']:.2f}%", flush=True)
                del current
            if study["scientific_results"]: freeze(study)
            report(study, results, "completed", mask_audit)
            write_json(output / "state.json", {"status": "completed", "completed_cases": len(results), "completed_utc": now(), "scientific_results": study["scientific_results"], "test_evaluated": False})
        except BaseException as error:
            write_json(output / "state.json", {"status": "failed", "error": f"{type(error).__name__}: {error}", "completed_cases": len(results), "updated_utc": now()})
            raise


if __name__ == "__main__":
    study = curve.read(ROOT / CONFIG)
    if sys.argv[1:]:
        if sys.argv[1:] != ["--smoke"]: raise SystemExit("Usage: python -m scripts.tail_compensation_pilot [--smoke]")
        study = {**study, "output": ".cache/tail_compensation_pilot_smoke", "device": "cpu", "steps": 4, "batch_size": 4,
                 "scientific_results": False, "limits": {"train": 32, "validation": 16, "fit": 12, "check_calibration": 12},
                 "finetune": {**study["finetune"], "epochs": 1, "batch_size": 16, "mixed_precision": False}}
    run(study)
