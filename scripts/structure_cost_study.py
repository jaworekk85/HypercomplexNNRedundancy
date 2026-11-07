"""Bounded structural-cost study; frozen old studies are read-only inputs."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from scripts.paper_a_controls import now, rel, verify_registered_sources
from scripts.run_activation_queue import lock
from src.cnn_training import evaluate, first_update_audit, prepare_batch, seed_everything
from src.compression import multilayer as old
from src.compression.structure_cost import compact, describe, group_width, make_model, release_consumers
from src.data.cifar_cached import file_sha256
from src.finetuning import finetune

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/structure_cost_study.json"
NEW_SOURCES = [CONFIG, "scripts/structure_cost_study.py", "src/compression/structure_cost.py", "tests_torch/test_structure_cost.py"]


def sources(study, dataset):
    config = curve.read(ROOT / ("configs/compression_curve.json" if dataset == "cifar" else "configs/compression_curve_eurosat.json"))
    config = {**config, "architectures": ["cnn6"], "seeds": study["seeds"]}
    if dataset == "cifar":
        data = curve.read(ROOT / "configs/multisource_compensation.json")
        rows = [r for r in curve.sources_of(data) if r["architecture"] == "cnn6" and r["seed"] in study["seeds"]]
    else:
        rows = [{"architecture": "cnn6", "seed": r["seed"], "path": r["path"],
                 "config": curve.read(ROOT / r["path"] / "summary.json")["config"]}
                for r in curve.read(ROOT / config["training_output"] / "summary.json")["runs"] if r["seed"] in study["seeds"]]
    rows = [{**r, "normalization": "bn", "dataset": dataset, "curve_output": config["output"]} for r in rows]
    if dataset == "cifar":
        parent = ROOT / study["no_bn_source"]
        if curve.read(parent / "state.json")["status"] != "completed":
            raise ValueError("Finish the registered no-BN baseline first")
        summary = curve.read(parent / "summary.json")
        if summary["seed"] not in study["no_bn_seeds"] or len(study["no_bn_seeds"]) != 1:
            raise ValueError("Only the completed one-seed no-BN pilot is registered")
        rows.insert(0, {"architecture": "cnn6", "seed": summary["seed"], "path": study["no_bn_source"],
                        "config": summary["config"], "normalization": "none", "dataset": dataset})
    return config, rows


def source_model(source, device):
    model = make_model(source["config"], source["dataset"], source["normalization"], "quaternion").to(device).eval()
    model.load_state_dict(torch.load(ROOT / source["path"] / "quaternion/model.pt", map_location=device, weights_only=True)["model"])
    return model


def freeze(study):
    hashes = verify_registered_sources()
    # No-BN sources were separately registered after the older chains.
    for name, expected in curve.read(ROOT / study["no_bn_source"] / "protocol.json")["source_hashes"].items():
        if file_sha256(ROOT / name) != expected: raise ValueError(f"No-BN source changed: {name}")
        hashes[name] = expected
    for name in [*NEW_SOURCES, "src/finetuning.py", "scripts/paper_a_controls.py"]:
        hashes[name] = file_sha256(ROOT / name)
    inputs = {}
    for dataset in ("cifar", "eurosat"):
        config, rows = sources(study, dataset)
        for source in rows:
            folder = ROOT / source["path"]
            paths = [folder / name for name in ("quaternion/model.pt", "quaternion/metrics.json", "summary.json", "split_indices.npz", "data.json")]
            checks = curve.read(folder / "quaternion/metrics.json")["first_update"]["checks"]
            if not all(checks.values()): raise AssertionError("Failed source training audit")
            if source["normalization"] == "bn":
                parent = ROOT / source["curve_output"]
                paths += [parent / "phase_a" / f"cnn6_seed_{source['seed']}.json"]
                for fraction in study["fractions"]:
                    f = f"f_{round(1000*fraction):03d}"
                    paths += [parent / f"compact/cnn6/seed_{source['seed']}/{f}/greedy_comp.pt",
                              parent / f"finetuned/cnn6/seed_{source['seed']}/{f}/greedy_comp/metrics.json",
                              parent / f"finetuned/cnn6/seed_{source['seed']}/{f}/greedy_comp/model.pt"]
                    extra = ROOT / f"results/structure_control/quaternion_paired_repair/finetuned/cnn6/seed_{source['seed']}/{f}/greedy_pruned"
                    if dataset == "cifar": paths += [extra / "model.pt", extra / "metrics.json"]
            for path in paths: inputs[rel(path)] = file_sha256(path)
    definition = {"config": study, "source_hashes": hashes, "input_hashes": inputs,
                  "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda}}
    path = ROOT / study["output"] / "protocol.json"
    if path.exists() and curve.read(path) != definition: raise ValueError("Registered structural-cost inputs changed")
    write_json(path, definition)
    return definition


def map_mse(moments, row):
    """Evaluate already fitted maps on moments; never fit to the check half."""
    m, k, v = moments.moments, row["kept"], row["victim"]
    a = np.asarray(row["matrices"])
    cross = m[k][:, :, v]
    source = m[k][:, :, k]
    sse = np.trace(m[v, :, v])-2*np.einsum("koi,kio", a, cross)+np.einsum("koi,loj,kilj", a, a, source)
    if sse < -1e-6: raise AssertionError("Negative reconstruction SSE")
    return max(0., float(sse))/(4*moments.rows)


def chain(model, fraction, study, fit, check, frozen=None, family="quaternion"):
    current, records = model, []
    for index, site in enumerate(old.sites(model)):
        width = group_width(old.producer(current, site)[0])
        moments = old.SiteMoments(*old.site_gram(current, site, fit, study["batch_size"]), True)
        victims = frozen[index] if frozen is not None else old.greedy_victims(
            moments, round(fraction*width), old.consumer_energy(current, site), study["ridge_alpha"], "quaternion_right")
        maps = old.site_maps(moments, victims, study["ridge_alpha"], "full_real" if family == "free" else "quaternion_right")
        result = compact(current, site, victims, maps, family)
        probe = check[:study["equivalence_images"]]
        with torch.no_grad():
            error = float((result(probe)-old.substituted_logits(current, site, maps, probe, study["batch_size"])).abs().max())
        if error > study["equivalence_tolerance"]: raise AssertionError(f"Exact physical fold failed at {site}: {error}")
        record = {"site": list(site), "victims": victims, "fold_max_abs_error": error}
        if family == "free":
            # Both local fits are evaluated on identical current-path activations.
            # This is a diagnostic, distinct from comparing whole Q/free chains.
            heldout = old.SiteMoments(*old.site_gram(current, site, check, study["batch_size"]), True)
            qmaps = old.site_maps(moments, victims, study["ridge_alpha"], "quaternion_right")
            record["local_reconstruction"] = [
                {"victim": a["victim"], "fit_full_mse": map_mse(moments, a), "check_full_mse": map_mse(heldout, a),
                 "fit_q_mse": map_mse(moments, b), "check_q_mse": map_mse(heldout, b)} for a, b in zip(maps, qmaps)]
        records.append(record)
        current = result
    return current, records


def measure(model, tensors, targets, mean, std, batch_size):
    return {name: evaluate(model, tensors[name], targets[name], mean, std, batch_size)[0] for name in ("validation", "check_calibration")}


def construct(study, source, baseline, fraction, fit, check, folder):
    path = folder / "phase_a.json"
    narrow = curve.reduced_config(source["config"], "cnn6", fraction)
    if path.exists():
        saved, models = curve.read(path), {}
        for case in study["cases"]:
            model = make_model(narrow, source["dataset"], source["normalization"], "free" if case.startswith("free") else "quaternion").to(fit.device).eval()
            checkpoint = folder / f"{case}.pt"
            if file_sha256(checkpoint) != saved["checkpoint_hashes"][case]: raise ValueError("Initial checkpoint changed")
            model.load_state_dict(torch.load(checkpoint, map_location=fit.device, weights_only=True)["model"])
            models[case] = model
        return models, saved
    if source["normalization"] == "bn":
        phase = curve.read(ROOT / source["curve_output"] / "phase_a" / f"cnn6_seed_{source['seed']}.json")
        record = next(r for r in phase["records"] if r["fraction"] == fraction and r["case"] == "greedy_comp")
        frozen = [r["victims"] for r in record["sites"]]
        q = make_model(narrow, source["dataset"], "bn", "quaternion").to(fit.device).eval()
        original = ROOT / source["curve_output"] / f"compact/cnn6/seed_{source['seed']}/f_{round(1000*fraction):03d}/greedy_comp.pt"
        q.load_state_dict(torch.load(original, map_location=fit.device, weights_only=True)["model"])
        q_records = record["sites"]
    else:
        q, q_records = chain(deepcopy(baseline), fraction, study, fit, check)
        frozen = [r["victims"] for r in q_records]
    pruned = deepcopy(baseline)
    for site, victims in zip(old.sites(pruned), frozen): pruned = compact(pruned, site, victims)
    full, records = chain(deepcopy(baseline), fraction, study, fit, check, frozen, "free")
    embedded = release_consumers(q).eval()
    with torch.no_grad(): embed_error = float((embedded(check[:study["equivalence_images"]])-q(check[:study["equivalence_images"]])).abs().max())
    if embed_error > 1e-6: raise AssertionError("Free-Q-initialization does not reproduce Q model")
    models = {"q_pruned": pruned, "q_comp": q, "free_q_init": embedded, "free_full_fit": full}
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint_hashes = {}
    for case, model in models.items():
        # Standard architecture reload checks layout/physical widths, not only logits.
        reloaded = make_model(narrow, source["dataset"], source["normalization"], "free" if case.startswith("free") else "quaternion").to(fit.device).eval()
        reloaded.load_state_dict(model.state_dict())
        with torch.no_grad():
            if float((reloaded(check[:8])-model(check[:8])).abs().max()) > 1e-6: raise AssertionError("Compact reload mismatch")
        checkpoint = folder / f"{case}.pt"
        torch.save({"model": {n: v.detach().cpu() for n, v in model.state_dict().items()}}, checkpoint)
        checkpoint_hashes[case] = file_sha256(checkpoint)
    result = {"source": source, "fraction": fraction, "frozen_victims": frozen, "q_site_records": q_records,
              "full_site_records": records, "free_q_embedding_max_error": embed_error,
              "architectures": {case: describe(model) for case, model in models.items()}, "checkpoint_hashes": checkpoint_hashes}
    write_json(path, result)
    return models, result


def reused_ft(source, fraction, case):
    if source["normalization"] != "bn": return None
    f = f"f_{round(1000*fraction):03d}"
    if case == "q_comp":
        return ROOT / source["curve_output"] / f"finetuned/cnn6/seed_{source['seed']}/{f}/greedy_comp"
    if case == "q_pruned" and source["dataset"] == "cifar":
        return ROOT / f"results/structure_control/quaternion_paired_repair/finetuned/cnn6/seed_{source['seed']}/{f}/greedy_pruned"
    return None


def aggregate(rows):
    keys = sorted({(r["dataset"], r["normalization"], r["fraction"], r["case"], r["stage"]) for r in rows})
    result = []
    for key in keys:
        group = [r for r in rows if (r["dataset"], r["normalization"], r["fraction"], r["case"], r["stage"]) == key]
        if len({r["seed"] for r in group}) != len(group): raise ValueError("Duplicate seed in aggregate")
        result.append(dict(zip(("dataset", "normalization", "fraction", "case", "stage"), key),
            real_parameters=group[0]["real_parameters"], original_q_saving=group[0]["original_q_saving"],
            seed_values=[{"seed": r["seed"], **r["evaluation"]["validation"]} for r in group],
            mean_accuracy=float(np.mean([r["evaluation"]["validation"]["accuracy"] for r in group])),
            sample_sd_accuracy=float(np.std([r["evaluation"]["validation"]["accuracy"] for r in group], ddof=1)) if len(group) > 1 else None))
    return result


def report(study, rows, status, groups_done):
    output = ROOT / study["output"]
    aggregates = aggregate(rows)
    index = {(r["dataset"], r["normalization"], r["fraction"], r["case"], r["stage"]): r for r in aggregates}
    effects = []
    for dataset, norm, fraction, _, stage in index:
        for left, right in (("q_comp", "q_pruned"), ("free_full_fit", "q_comp"), ("free_q_init", "q_comp"), ("free_full_fit", "free_q_init")):
            key = dataset, norm, fraction, left, right, stage
            if any(tuple(e[k] for k in ("dataset", "normalization", "fraction", "left", "right", "stage")) == key for e in effects): continue
            a, b = index.get((dataset, norm, fraction, left, stage)), index.get((dataset, norm, fraction, right, stage))
            if a is None or b is None: continue
            aa, bb = ({v["seed"]: v["accuracy"] for v in r["seed_values"]} for r in (a, b))
            if set(aa) != set(bb): continue
            values = [{"seed": s, "accuracy_difference_pp": 100*(aa[s]-bb[s])} for s in sorted(aa)]
            effects.append(dict(zip(("dataset", "normalization", "fraction", "left", "right", "stage"), key),
                                seed_values=values, mean_difference_pp=float(np.mean([v["accuracy_difference_pp"] for v in values]))))
    write_json(output / "summary.json", {"status": status, "groups_done": groups_done, "planned_groups": 14,
               "rows": rows, "aggregates": aggregates, "effects": effects, "test_evaluated": False,
               "new_ft_cases": len([r for r in rows if r["stage"] == "after" and not r["reused_training"]]),
               "reused_ft_cases": len([r for r in rows if r["stage"] == "after" and r["reused_training"]])})
    lines = ["# Structural cost: all-site Q versus physically compiled unrestricted compensation", "",
             "## Hypotheses", "", *[f"- {h}" for h in study["hypotheses"]], "",
             "## Measured results", "", f"Status: {status}; {groups_done}/14 dataset/normalization/seed/fraction groups. Validation means; all seed values in summary.json. New study chosen after earlier outcomes; no test inference.", "",
             "Q-pruned and Q-comp remove identical groups. Free-Q-init starts with exactly Q-comp's predictions and releases five consumer convolutions to dense real weights. Free-full-fit folds unrestricted maps without projection. First conv remains Q; readout was already real. Same width is not matched parameters/compute. Negative savings mean MORE parameters than the full source QNN.", "",
             "| Dataset | Norm | Removed/site | Case | Stage | Real params | Saving vs full Q | n | Val mean (%) | SD (pp) |",
             "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in aggregates:
        sd = "—" if r["sample_sd_accuracy"] is None else f"{100*r['sample_sd_accuracy']:.2f}"
        lines.append(f"| {r['dataset']} | {r['normalization']} | {r['fraction']} | {r['case']} | {r['stage']} | {r['real_parameters']} | {100*r['original_q_saving']:.2f}% | {len(r['seed_values'])} | {100*r['mean_accuracy']:.2f} | {sd} |")
    lines.extend(["", "### Paired accuracy effects", "", "| Dataset | Norm | Removed/site | Stage | Left minus right | Mean (pp) | Seeds (pp) |", "| --- | --- | ---: | --- | --- | ---: | --- |"])
    for r in effects:
        lines.append(f"| {r['dataset']} | {r['normalization']} | {r['fraction']} | {r['stage']} | {r['left']} - {r['right']} | {r['mean_difference_pp']:+.2f} | " + ", ".join(f"{v['accuracy_difference_pp']:+.2f}" for v in r["seed_values"]) + " |")
    lines.extend(["", "## Interpretation limits and decision rules", "", *[f"- {h}" for h in study["decision_rules"]], "",
                  "BN original masks/results are reused with provenance, not new training. No-BN uses its own fitted masks and only one seed. Full-vs-Q after repair combines map initialization and parameter-family freedom; free-Q-init provides the capacity control. Three epochs, same attempted batch/augmentation schedules, not audited equal successful AMP steps. No latency claim or new-method claim (GRAIL2026/Kim2020). Checkpoint bytes and reconstruction/fold checks are saved per group. No external correspondence."])
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run():
    study = curve.read(ROOT / CONFIG)
    output = ROOT / study["output"]
    with lock("compression_curve_runner.lock"):
        protocol = freeze(study)
        torch.set_num_threads(study["cpu_threads"])
        device = torch.device(study["device"])
        write_json(output / "state.json", {"status": "running", "pid": os.getpid(), "started_utc": now(), "planned_groups": 14, "test_evaluated": False})
        rows, done = [], 0
        try:
            for dataset in ("cifar", "eurosat"):
                data_config, source_rows = sources(study, dataset)
                images, labels, mean, std, unused_sources = curve.load_data(data_config, device)
                for source in source_rows:
                    seed_everything(source["seed"])
                    ids = curve.split_ids(source, None)
                    tensors = {name: torch.as_tensor(images[ids[name]], device=device) for name in ("train", "validation", "check_calibration")}
                    targets = {name: torch.as_tensor(labels[ids[name]], device=device) for name in tensors}
                    fit = prepare_batch(torch.as_tensor(images[ids["fit"]], device=device), mean, std)
                    check = prepare_batch(tensors["check_calibration"], mean, std)
                    baseline = source_model(source, device)
                    base_metrics = measure(baseline, tensors, targets, mean, std, study["batch_size"])
                    expected = curve.read(ROOT / source["path"] / "quaternion/metrics.json")["selected_validation"]
                    if base_metrics["validation"]["accuracy"] != expected["accuracy"] or abs(base_metrics["validation"]["cross_entropy"]-expected["cross_entropy"]) > 1e-4:
                        raise AssertionError("Source does not reproduce selected validation")
                    base_parameters = describe(baseline)["real_parameters"]
                    for fraction in study["fractions"]:
                        folder = output / dataset / source["normalization"] / f"seed_{source['seed']}" / f"f_{round(1000*fraction):03d}"
                        write_json(output / "state.json", {"status": "running", "pid": os.getpid(), "updated_utc": now(), "groups_done": done,
                                   "current": rel(folder), "planned_groups": 14, "test_evaluated": False})
                        print(f"Construct {dataset}/{source['normalization']}/{source['seed']}/{fraction}", flush=True)
                        models, phase = construct(study, source, baseline, fraction, fit, check, folder)
                        for case in study["cases"]:
                            model = models[case]
                            structure = describe(model)
                            before_path = folder / case / "before.json"
                            if not before_path.exists():
                                metrics = measure(model, tensors, targets, mean, std, study["batch_size"])
                                # Replay both exact old Q masks and Q-comp outcomes where they exist.
                                if source["normalization"] == "bn" and not case.startswith("free"):
                                    original = curve.read(ROOT / source["curve_output"] / "phase_a" / f"cnn6_seed_{source['seed']}.json")
                                    policy = "greedy_comp" if case == "q_comp" else "greedy_pruned"
                                    reference = next(r["evaluation"] for r in original["rows"] if r["fraction"] == fraction and r["case"] == policy)
                                    if any(metrics[s]["accuracy"] != reference[s]["accuracy"] or abs(metrics[s]["cross_entropy"]-reference[s]["cross_entropy"]) > 1e-4 for s in metrics):
                                        raise AssertionError("Initial Q case does not reproduce frozen phase A")
                                write_json(before_path, {"dataset": dataset, "normalization": source["normalization"], "seed": source["seed"], "fraction": fraction,
                                           "case": case, "stage": "before", **structure, "source_real_parameters": base_parameters,
                                           "original_q_saving": 1-structure["real_parameters"]/base_parameters, "evaluation": metrics,
                                           "checkpoint_bytes": (folder / f"{case}.pt").stat().st_size, "test_evaluated": False})
                            rows.append(curve.read(before_path))
                            target = folder / case / "metrics.json"
                            if not target.exists():
                                start = time.perf_counter()
                                # Discarded-copy update audit does not change the experimental model.
                                audit_model = deepcopy(model)
                                audit = first_update_audit(audit_model, prepare_batch(tensors["train"][:32], mean, std), targets["train"][:32],
                                                           torch.optim.AdamW(audit_model.parameters(), lr=study["finetune"]["learning_rate"]))
                                del audit_model
                                reused = reused_ft(source, fraction, case)
                                if reused is not None:
                                    original = curve.read(reused / "metrics.json")
                                    history = original["history"]
                                    model.load_state_dict(torch.load(reused / "model.pt", map_location=device, weights_only=True)["model"])
                                else:
                                    model, history = finetune(model, tensors, targets, mean, std, study["finetune"], source["seed"], ("validation", "check_calibration"))
                                if describe(model) != structure: raise AssertionError("Repair changed model structure")
                                evaluation = measure(model, tensors, targets, mean, std, study["batch_size"])
                                if any(evaluation[s]["accuracy"] != history[-1][s]["accuracy"] or abs(evaluation[s]["cross_entropy"]-history[-1][s]["cross_entropy"]) > 1e-4 for s in evaluation):
                                    raise AssertionError("Repaired model differs from reported history")
                                checkpoint = target.parent / "model.pt"
                                torch.save({"model": {n: v.detach().cpu() for n, v in model.state_dict().items()}}, checkpoint)
                                result = {**curve.read(before_path), "stage": "after", "evaluation": evaluation, "history": history,
                                          "discarded_copy_update_audit": audit, "reused_training": reused is not None,
                                          "reused_from": rel(reused) if reused is not None else None,
                                          "checkpoint_bytes": checkpoint.stat().st_size, "checkpoint_sha256": file_sha256(checkpoint),
                                          "case_seconds": time.perf_counter()-start}
                                write_json(target, result)
                            rows.append(curve.read(target))
                            report(study, rows, "running", done)
                            print(f"{dataset}/{source['normalization']}/{source['seed']}/{fraction}/{case}: before {100*rows[-2]['evaluation']['validation']['accuracy']:.2f}%, after {100*rows[-1]['evaluation']['validation']['accuracy']:.2f}%, {structure['real_parameters']} params, reused={rows[-1]['reused_training']}", flush=True)
                        done += 1
                        report(study, rows, "running", done)
                        del models
                    del baseline, tensors, targets, fit, check
                del images
            freeze(study)
            if len(rows) != 112: raise AssertionError(f"Incomplete fixed study: {len(rows)} rows")
            report(study, rows, "completed", done)
            write_json(output / "state.json", {"status": "completed", "completed_utc": now(), "groups_done": done, "rows": len(rows), "test_evaluated": False})
        except BaseException as error:
            write_json(output / "state.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "updated_utc": now(),
                       "pid": os.getpid(), "error": f"{type(error).__name__}: {error}", "groups_done": done, "test_evaluated": False})
            raise


if __name__ == "__main__":
    run()
