"""Read-only audit of frozen provenance, masks and physical checkpoint layouts."""
from pathlib import Path

import torch

from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from scripts.resnet20_compression import CONFIG, ROOT, load_model
from src.data.cifar_cached import file_sha256
from src.models.quaternion_conv_torch import QuaternionConv2d


def run():
    torch.set_num_threads(2)
    study = curve.read(ROOT/CONFIG)
    output = ROOT/study["output"]
    protocol = curve.read(output/"protocol.json")
    for name, digest in {**protocol["source_hashes"], **protocol["input_hashes"]}.items():
        if file_sha256(ROOT/name) != digest: raise AssertionError(f"Frozen input/source changed: {name}")
    summary = curve.read(output/"summary.json")
    if summary["status"] != "completed" or len(summary["results"]) != 12: raise AssertionError("Incomplete study")
    baseline = curve.read(ROOT/study["baseline_config"])
    masks = {}
    folds, local_fits = [], []
    for row in summary["results"]:
        family, fraction, policy = row["family"], row["fraction"], row["policy"]
        key = (family, fraction)
        sites = row["site_records"]
        if len(sites) != 9: raise AssertionError("Wrong number of blocks")
        pattern = [s["victims"] for s in sites]
        if key in masks and masks[key] != pattern: raise AssertionError("Victim mismatch between policies")
        masks[key] = pattern
        for site in sites:
            folds.append(site["fold_max_abs_error"])
            if site["direct_audit"]:
                local = site["direct_audit"]
                if local["fit_direct_mse"] > local["fit_prior_mse"]+1e-8: raise AssertionError("Local fit worsened")
                local_fits.append(local)
        folder = output/family/f"f_{round(1000*fraction):03d}"/policy
        for stage in ("before", "after"):
            path = folder/(stage+".pt")
            if file_sha256(path) != row[stage+"_sha256"]: raise AssertionError("Checkpoint hash mismatch")
            model = load_model(path, baseline, family, torch.device("cpu"))
            if sum(p.numel() for p in model.parameters()) != row["parameters"]: raise AssertionError("Physical parameter mismatch")
            if [b.conv1.out_channels for b in model.blocks] != row["hidden_widths"]: raise AssertionError("Wrong compact widths")
            q_layers = sum(isinstance(m, QuaternionConv2d) for m in model.modules())
            if q_layers != (19 if family == "quaternion" else 0): raise AssertionError("Unexpected convolution family")
        if row["counts"]["attempted"] != 1125 or len(row["history"]) != 3: raise AssertionError("Wrong repair budget")
    if max(folds) > .001 or len(local_fits) != 36: raise AssertionError("Fold/direct check failed")
    result = {"status": "passed", "source_count": len(protocol["source_hashes"]), "input_count": len(protocol["input_hashes"]),
              "cases": 12, "checkpoint_layouts": 24, "same_mask_groups": len(masks), "fold_checks": len(folds),
              "max_fold_error": max(folds), "direct_local_fits": len(local_fits), "test_evaluated": False,
              "scope": "Checkpoint layouts/counts, source/input/checkpoint hashes, within-family masks, fold records and local fit; no independent full classification replay."}
    write_json(output/"independent_audit.json", result)
    print(f"PASSED: {result['cases']} cases, {result['checkpoint_layouts']} physical checkpoints, {result['fold_checks']} fold records, {result['direct_local_fits']} fits", flush=True)


if __name__ == "__main__": run()
