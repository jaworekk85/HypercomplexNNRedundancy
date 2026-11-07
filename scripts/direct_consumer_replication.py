"""Fixed replication of the direct-Q consumer pilot; all prior numeric files immutable."""
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
from scripts.structure_cost_study import sources, source_model
from src.cnn_training import first_update_audit, prepare_batch, seed_everything
from src.compression import multilayer as old
from src.compression.direct_consumer import refit_consumer
from src.compression.structure_cost import compact, describe, make_model
from src.data.cifar_cached import file_sha256
from src.finetuning import finetune

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/direct_consumer_replication.json"
NEW = [CONFIG, "scripts/direct_consumer_replication.py"]


def make_smoke(config):
    return {**config, "output": ".cache/direct_consumer_replication_smoke", "device": "cpu", "seeds": [2026],
            "fractions": [.5], "limits": {"train": 32, "validation": 16, "fit": 12, "check_calibration": 12},
            "scientific_results": False, "finetune": {**config["finetune"], "epochs": 1, "batch_size": 16, "mixed_precision": False},
            "smoke": True}


def planned(study):
    parent = curve.read(ROOT / "configs/structure_cost_study.json")
    jobs = []
    for dataset in study["datasets"]:
        data, all_sources = sources(parent, dataset)
        selected = [s for s in all_sources if s["normalization"] == "bn" and s["seed"] in study["seeds"]]
        selected.sort(key=lambda s: study["seeds"].index(s["seed"]))
        if study["include_existing_no_bn"] and dataset == "cifar":
            selected += [s for s in all_sources if s["normalization"] == "none"]
        if study.get("smoke"):
            selected = [next(s for s in selected if s["normalization"] == ("none" if dataset == "cifar" else "bn"))]
        for source in selected:
            for fraction in study["fractions"]:
                identifier = f"{dataset}/{source['normalization']}/seed_{source['seed']}/f_{round(1000*fraction):03d}"
                discovery = dataset == "cifar" and source["normalization"] == "bn" and source["seed"] == 2026 and fraction == .5
                jobs.append({"id": identifier, "source": source, "fraction": fraction, "discovery": discovery,
                             "reuse_discovery": discovery and not study.get("smoke"),
                             "reference_group": f"{study['reference_study']}/{identifier}"})
    return jobs


def freeze(study, jobs):
    parent_path = ROOT / study["discovery_pilot"] / "protocol.json"
    parent = curve.read(parent_path)
    hashes = {**parent["source_hashes"], **{n: file_sha256(ROOT / n) for n in NEW}}
    for name, h in {**parent["source_hashes"], **parent["input_hashes"]}.items():
        if file_sha256(ROOT / name) != h: raise ValueError(f"Frozen pilot source/input changed: {name}")
    paths = [parent_path, ROOT / study["discovery_pilot"] / "summary.json", ROOT / study["discovery_pilot"] / "direct_before.pt",
             ROOT / study["discovery_pilot"] / "finetuned/model.pt", ROOT / study["discovery_pilot"] / "independent_audit.json"]
    for job in jobs:
        folder = ROOT / job["reference_group"]
        paths.append(folder / "phase_a.json")
        for case in ("q_pruned", "q_comp", "free_q_init", "free_full_fit"):
            paths += [folder / case / "before.json", folder / case / "metrics.json", folder / case / "model.pt"]
    if "eurosat" in study["datasets"]:
        paths += [ROOT / ".cache/eurosat/prepared/data.json", ROOT / ".cache/eurosat/prepared/split_indices.npz"]
    definition = {"config": study, "jobs": jobs, "source_hashes": hashes,
                  "input_hashes": {**parent["input_hashes"], **{rel(p): file_sha256(p) for p in paths}},
                  "environment": {"torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda}}
    path = ROOT / study["output"] / "protocol.json"
    if path.exists() and curve.read(path) != definition: raise ValueError("Replication sources/config/inputs changed")
    write_json(path, definition)
    return definition


def surgery(model, site, victims, maps=None):
    if isinstance(old.producer(model, site)[1], torch.nn.BatchNorm2d):
        result, kept, lost = old.compress_site(model, site, victims, maps)
    else:
        result = compact(model, site, victims, maps)
        kept = sorted(set(range(old.units(model, site)))-set(victims)); lost = 0.
    return result, kept, lost


def measure(model, images, labels, ids, mean, std, data, device):
    return curve.metrics(model, images, labels, ids, mean, std, data, device)


def references(job):
    result = {}
    for case in ("q_pruned", "q_comp", "free_q_init", "free_full_fit"):
        folder = ROOT / job["reference_group"] / case
        result[case] = {"before": curve.read(folder / "before.json"), "after": curve.read(folder / "metrics.json")}
    return result


def audit_cached(row):
    for key in ("before_checkpoint", "after_checkpoint"):
        if file_sha256(ROOT / row[key]) != row[f"{key}_sha256"]:
            raise ValueError(f"Completed checkpoint changed: {row[key]}")


def run_job(study, job, images, labels, mean, std, data, device):
    output = ROOT / study["output"] / job["id"]
    final_path = output / "result.json"
    if final_path.exists():
        row = curve.read(final_path); audit_cached(row); return row
    output.mkdir(parents=True, exist_ok=True)
    source = job["source"]
    if job["reuse_discovery"]:
        pilot = curve.read(ROOT / study["discovery_pilot"] / "summary.json")
        before_checkpoint = f"{study['discovery_pilot']}/direct_before.pt"
        after_checkpoint = f"{study['discovery_pilot']}/finetuned/model.pt"
        result = {**job, "architecture": pilot["architecture"], "sites": pilot["sites"], "before": pilot["before"], "after": pilot["after"],
                  "repair": pilot["repair"], "reused_training": True, "references": references(job),
                  "before_checkpoint": before_checkpoint, "before_checkpoint_sha256": file_sha256(ROOT / before_checkpoint),
                  "after_checkpoint": after_checkpoint, "after_checkpoint_sha256": file_sha256(ROOT / after_checkpoint), "test_evaluated": False}
        write_json(final_path, result); return result
    seed_everything(source["seed"])
    ids = curve.split_ids(source, study["limits"])
    if set(ids["fit"]) & set(ids["check_calibration"]): raise AssertionError("Fit/check overlap")
    fit = prepare_batch(torch.as_tensor(images[ids["fit"]], device=device), mean, std)
    check = prepare_batch(torch.as_tensor(images[ids["check_calibration"]], device=device), mean, std)
    frozen = curve.read(ROOT / job["reference_group"] / "phase_a.json")["frozen_victims"]
    current = source_model(source, device)
    site_rows = []
    for index, site in enumerate(old.sites(current)):
        folder = output / "sites" / f"site_{index:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint, record_path = folder / "model.pt", folder / "metrics.json"
        if record_path.exists():
            row = curve.read(record_path)
            if file_sha256(checkpoint) != row["checkpoint_sha256"]: raise ValueError("Site checkpoint drift")
            if "moments_sha256" in row and file_sha256(folder / "moments.npz") != row["moments_sha256"]: raise ValueError("Site moments drift")
            current, _, _ = surgery(current, site, frozen[index])
            current.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
            site_rows.append(row); continue
        moments = old.SiteMoments(*old.site_gram(current, site, fit, study["batch_size"]), True)
        maps = old.site_maps(moments, frozen[index], study["ridge_alpha"], "quaternion_right")
        folded, kept, lost = surgery(current, site, frozen[index], maps)
        with torch.no_grad():
            error = float((folded(check[:min(32, len(check))])-old.substituted_logits(current, site, maps, check[:min(32, len(check))], study["batch_size"])).abs().max())
        if error > .001: raise AssertionError("Physical map-fold mismatch")
        row = {"site": list(site), "victims": frozen[index], "fold_max_abs_error": error, "projection_discarded_energy": lost}
        if index < 5:
            current, audit, arrays = refit_consumer(current, folded, site, kept, fit, check, study["ridge_alpha"], study["batch_size"])
            row["direct_audit"] = audit
            np.savez_compressed(folder / "moments.npz", **arrays)
            row["moments_sha256"] = file_sha256(folder / "moments.npz")
            print(f"{job['id']} site {site}: check {audit['check_prior_mse']:.5g}->{audit['check_direct_mse']:.5g}", flush=True)
        else: current = folded
        torch.save({"model": {n: v.detach().cpu() for n, v in current.state_dict().items()}}, checkpoint)
        row["checkpoint_sha256"] = file_sha256(checkpoint)
        write_json(record_path, row); site_rows.append(row)
    narrow = curve.reduced_config(source["config"], "cnn6", job["fraction"])
    reloaded = make_model(narrow, source["dataset"], source["normalization"], "quaternion").to(device).eval()
    reloaded.load_state_dict(current.state_dict())
    with torch.no_grad():
        if float((reloaded(check[:8])-current(check[:8])).abs().max()) > 1e-6: raise AssertionError("Compact reload failed")
    structure = describe(current)
    refs = references(job)
    if structure["real_parameters"] != refs["q_comp"]["before"]["real_parameters"] or structure["quaternion_convolutions"] != 6 or structure["dense_real_convolutions"]:
        raise AssertionError("Not the same compact Q architecture")
    before = measure(current, images, labels, ids, mean, std, data, device)
    before_path = output / "direct_before.pt"
    torch.save({"model": {n: v.detach().cpu() for n, v in current.state_dict().items()}}, before_path)
    repair_path = output / "finetuned/metrics.json"
    if repair_path.exists():
        repair = curve.read(repair_path)
        if file_sha256(repair_path.parent / "model.pt") != repair["checkpoint_sha256"]: raise ValueError("Repair checkpoint drift")
    else:
        tensors = {name: torch.as_tensor(images[ids[name]], device=device) for name in ("train", "validation", "check_calibration")}
        targets = {name: torch.as_tensor(labels[ids[name]], device=device) for name in tensors}
        copied = deepcopy(current)
        audit = first_update_audit(copied, prepare_batch(tensors["train"][:32], mean, std), targets["train"][:32],
                                   torch.optim.AdamW(copied.parameters(), lr=study["finetune"]["learning_rate"]))
        del copied
        counter = {"successful_optimizer_steps": 0}
        original_adamw = torch.optim.AdamW
        class CountedAdamW(original_adamw):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.register_step_post_hook(lambda optimizer, args, kwargs: counter.__setitem__("successful_optimizer_steps", counter["successful_optimizer_steps"]+1))
        with patch.object(torch.optim, "AdamW", CountedAdamW):
            tuned, history = finetune(current, tensors, targets, mean, std, study["finetune"], source["seed"], ("validation", "check_calibration"))
        if describe(tuned) != structure: raise AssertionError("Repair changed architecture")
        repair_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {n: v.detach().cpu() for n, v in tuned.state_dict().items()}}, repair_path.parent / "model.pt")
        repair = {"history": history, "final": {s: history[-1][s] for s in ("validation", "check_calibration")}, **counter,
                  "attempted_optimizer_steps": study["finetune"]["epochs"]*((len(ids["train"])+study["finetune"]["batch_size"]-1)//study["finetune"]["batch_size"]),
                  "discarded_copy_update_audit": audit, "checkpoint_sha256": file_sha256(repair_path.parent / "model.pt"), "test_evaluated": False}
        write_json(repair_path, repair)
    result = {**job, "architecture": structure, "sites": site_rows, "before": before, "after": repair["final"], "repair": repair,
              "reused_training": False, "references": refs, "before_checkpoint": rel(before_path), "before_checkpoint_sha256": file_sha256(before_path),
              "after_checkpoint": rel(repair_path.parent / "model.pt"), "after_checkpoint_sha256": file_sha256(repair_path.parent / "model.pt"),
              "test_evaluated": False, "scientific_results": study["scientific_results"]}
    write_json(final_path, result)
    print(f"DONE {job['id']}: direct val {100*before['validation']['accuracy']:.2f}->{100*repair['final']['validation']['accuracy']:.2f}%, {structure['real_parameters']} params", flush=True)
    return result


def summarize(study, jobs, results, status):
    aggregates, effects = [], []
    keys = sorted({(r["source"]["dataset"], r["source"]["normalization"], r["fraction"]) for r in results})
    for dataset, norm, fraction in keys:
        group = [r for r in results if (r["source"]["dataset"], r["source"]["normalization"], r["fraction"]) == (dataset, norm, fraction)]
        for stage in ("before", "after"):
            for case in ("q_pruned", "q_comp", "direct_q", "free_q_init", "free_full_fit"):
                seed_values = [{"seed": r["source"]["seed"], "discovery": r["discovery"],
                                **(r[stage]["validation"] if case == "direct_q" else r["references"][case][stage]["evaluation"]["validation"])} for r in group]
                parameters = group[0]["architecture"]["real_parameters"] if case == "direct_q" else group[0]["references"][case][stage]["real_parameters"]
                aggregates.append({"dataset": dataset, "normalization": norm, "fraction": fraction, "stage": stage, "case": case,
                                   "real_parameters": parameters, "seed_values": seed_values,
                                   "mean_accuracy": float(np.mean([v["accuracy"] for v in seed_values])),
                                   "sample_sd_accuracy": float(np.std([v["accuracy"] for v in seed_values], ddof=1)) if len(seed_values)>1 else None})
            for independent in (False, True):
                subset = [r for r in group if not independent or not r["discovery"]]
                if not subset: continue
                values = [{"seed": r["source"]["seed"], "discovery": r["discovery"],
                           "accuracy_difference_pp": 100*(r[stage]["validation"]["accuracy"]-r["references"]["q_comp"][stage]["evaluation"]["validation"]["accuracy"]),
                           "cross_entropy_difference": r[stage]["validation"]["cross_entropy"]-r["references"]["q_comp"][stage]["evaluation"]["validation"]["cross_entropy"]} for r in subset]
                effects.append({"dataset": dataset, "normalization": norm, "fraction": fraction, "stage": stage,
                                "exclude_discovery": independent, "seed_values": values,
                                "mean_accuracy_difference_pp": float(np.mean([v["accuracy_difference_pp"] for v in values]))})
    output = ROOT / study["output"]
    summary = {"status": status, "planned_jobs": len(jobs), "completed_jobs": len(results), "jobs": results,
               "aggregates": aggregates, "effects": effects, "scientific_results": study["scientific_results"], "test_evaluated": False,
               "new_repairs_completed": sum(not r["reused_training"] for r in results), "discovery_reused": sum(r["reused_training"] for r in results)}
    write_json(output / "summary.json", summary)
    lines = ["# Direct-Q consumer reconstruction: fixed replication", "", "## Hypothesis", "", study["hypothesis"], "",
             "## Measured results", "", f"Status: {status}; {len(results)}/{len(jobs)} jobs. Scientific results: {study['scientific_results']}. Validation, no new test inference.",
             "Discovery CIFAR BN seed2026 f=.5 is reused and marked; aggregates containing it are not independent confirmation. Separate new-seed effects below. No-BN remains one seed. Same compact Q size/masks; dense references have different budgets.", "",
             "| Dataset | Norm | Removed/site | Stage | Case | Parameters | n | Val mean (%) | SD (pp) |", "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |"]
    for r in aggregates:
        sd = "—" if r["sample_sd_accuracy"] is None else f"{100*r['sample_sd_accuracy']:.2f}"
        lines.append(f"| {r['dataset']} | {r['normalization']} | {r['fraction']} | {r['stage']} | {r['case']} | {r['real_parameters']} | {len(r['seed_values'])} | {100*r['mean_accuracy']:.2f} | {sd} |")
    lines += ["", "### Paired direct minus map-fold effects", "", "| Dataset | Norm | Removed/site | Stage | Exclude discovery | Mean (pp) | Seeds (pp) |", "| --- | --- | ---: | --- | --- | ---: | --- |"]
    for r in effects:
        lines.append(f"| {r['dataset']} | {r['normalization']} | {r['fraction']} | {r['stage']} | {r['exclude_discovery']} | {r['mean_accuracy_difference_pp']:+.2f} | " + ", ".join(f"{v['seed']}:{v['accuracy_difference_pp']:+.2f}" for v in r["seed_values"]) + " |")
    lines += ["", "## Interpretation limits", "", *[f"- {r}" for r in study["interpretation_rules"]], "",
              "Five quaternion consumers use direct pre-BN output ridge centered on their folded candidates; final real head remains map-fold-only. Fit and check halves are separate. Local fit improvements are partly guaranteed by the centered objective; local check/classification improvements are empirical and may be negative. Preserve all cases. No new LS algorithm or quaternion uniqueness claim (He2017ChannelPruning)."]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(study):
    jobs = planned(study)
    output = ROOT / study["output"]
    with lock("compression_curve_runner.lock"):
        freeze(study, jobs)
        torch.set_num_threads(study["cpu_threads"])
        device = torch.device(study["device"])
        results = []
        write_json(output / "state.json", {"status": "running", "pid": os.getpid(), "started_utc": now(), "planned_jobs": len(jobs), "test_evaluated": False})
        try:
            for dataset in study["datasets"]:
                data = curve.read(ROOT / ("configs/compression_curve.json" if dataset == "cifar" else "configs/compression_curve_eurosat.json"))
                data = {**data, "architectures": ["cnn6"], "seeds": study["seeds"]}
                if dataset == "eurosat":
                    meta = curve.read(ROOT / data["data"] / "data.json")
                    if file_sha256(ROOT / data["data"] / "eurosat_nrgb.npz") != meta["arrays_sha256"] or file_sha256(ROOT / data["data"] / "split_indices.npz") != meta["split_sha256"]:
                        raise ValueError("Prepared EuroSAT data drift")
                images, labels, mean, std, unused_sources = curve.load_data(data, device)
                for job in [j for j in jobs if j["source"]["dataset"] == dataset]:
                    write_json(output / "state.json", {"status": "running", "pid": os.getpid(), "current": job["id"], "completed_jobs": len(results), "planned_jobs": len(jobs), "updated_utc": now(), "test_evaluated": False})
                    results.append(run_job(study, job, images, labels, mean, std, data, device))
                    summarize(study, jobs, results, "running")
                del images
            freeze(study, jobs)
            if len(results) != len(jobs): raise AssertionError("Incomplete study")
            summarize(study, jobs, results, "completed")
            write_json(output / "state.json", {"status": "completed", "completed_utc": now(), "completed_jobs": len(results), "test_evaluated": False})
        except BaseException as error:
            write_json(output / "state.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "updated_utc": now(),
                       "error": f"{type(error).__name__}: {error}", "completed_jobs": len(results), "test_evaluated": False})
            raise


if __name__ == "__main__":
    config = curve.read(ROOT / CONFIG)
    if len(sys.argv) > 1:
        if sys.argv[1:] != ["--smoke"]: raise SystemExit("Usage: python -m scripts.direct_consumer_replication [--smoke]")
        config = make_smoke(config)
    run(config)
