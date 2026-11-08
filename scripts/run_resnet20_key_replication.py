"""Orchestrate the preregistered two-seed ResNet compression replication."""
import argparse
from copy import deepcopy
from pathlib import Path

import torch

from scripts import compression_curve as curve
from scripts import resnet20_key_replication as stage
from scripts.cifar_cnn import write_json
from scripts.paper_a_controls import now
from src.data.cifar_cached import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def preregister(config):
    sources = sorted(set(stage.NEW + ["scripts/run_resnet20_key_replication.py", "tests_torch/test_resnet20_key_replication.py"]))
    definition = {"config": config, "source_hashes": {p: file_sha256(ROOT / p) for p in sources},
                  "baseline_protocol_template": config["baseline_output_template"] + "/protocol.json",
                  "discovery_summary_sha256": file_sha256(ROOT / "results/resnet20_compression/summary.json"),
                  "test_evaluated": False}
    path = ROOT / config["output"] / "protocol.json"
    if path.exists():
        if curve.read(path) != definition:
            raise ValueError("Replication preregistration changed")
    else:
        write_json(path, definition)
    return definition


def studies(config):
    base = curve.read(ROOT / config["base_compression_config"])
    if config["seeds"] != [2027, 2028] or config["fractions"] != [0.5]:
        raise ValueError("Unexpected replication scope")
    for seed in config["seeds"]:
        study = deepcopy(base)
        baseline_output = config["baseline_output_template"].format(seed=seed)
        study.update(seed=seed, output=config["output_template"].format(seed=seed),
                     baseline_output=baseline_output, parent_protocol=baseline_output + "/protocol.json",
                     baseline_config=config["baseline_config"], fractions=config["fractions"],
                     families=config["families"], policies=config["policies"],
                     scope=config["scope"], prior_art=config["prior_art"])
        yield study


def validate_parent(study):
    root = ROOT / study["baseline_output"]
    summary = curve.read(root / "summary.json")
    parent = curve.read(root / "protocol.json")
    if summary["status"] != "completed" or parent["config"]["seed"] != study["seed"]:
        raise ValueError("Baseline must be complete and match replication seed")
    if set(parent["config"]["models"]) != set(study["families"]):
        raise ValueError("Baseline families do not match")
    expected = curve.read(ROOT / study["baseline_config"])
    for key, value in expected.items():
        if key not in {"seed", "output", "scope", "resume"} and parent["config"].get(key) != value:
            raise ValueError(f"Replication baseline recipe changed: {key}")
    for family in study["families"]:
        row = curve.read(root / family / "metrics.json")
        if file_sha256(root / family / "model.pt") != row["checkpoint_sha256"]:
            raise ValueError("Baseline checkpoint mismatch")


def validate_results(study, summary):
    rows = summary["results"]
    expected = {(f, x, p) for f in study["families"] for x in study["fractions"] for p in study["policies"]}
    actual = {(r["family"], r["fraction"], r["policy"]) for r in rows}
    if summary["status"] != "completed" or len(rows) != len(expected) or actual != expected:
        raise ValueError("Incomplete or duplicate replication cases")
    for family in study["families"]:
        for fraction in study["fractions"]:
            group = [r for r in rows if r["family"] == family and r["fraction"] == fraction]
            masks = [[s["victims"] for s in r["site_records"]] for r in group]
            if any(m != masks[0] for m in masks) or len(masks[0]) != 9:
                raise ValueError("Policies do not share all nine victim sets")
            if len({r["parameters"] for r in group}) != 1 or len({tuple(r["hidden_widths"]) for r in group}) != 1:
                raise ValueError("Paired policies have different compact models")
            for row in group:
                folder = ROOT / study["output"] / family / f"f_{round(1000*fraction):03d}" / row["policy"]
                if file_sha256(folder / "after.pt") != row["after_sha256"]:
                    raise ValueError("Final compact checkpoint mismatch")
                if row["test_evaluated"] or row["counts"]["successful"] != row["counts"]["attempted"]:
                    raise ValueError("Unexpected test evaluation or skipped optimizer updates")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = curve.read(ROOT / stage.CONFIG)
    preregister(config)
    torch.set_num_threads(curve.read(ROOT / config["base_compression_config"])["cpu_threads"])
    combined = []
    for study in studies(config):
        validate_parent(study)
        if not (ROOT / study["output"] / "readiness.json").exists():
            stage.prepare(study)
        if not args.run:
            continue
        stage.run(study)
        state = curve.read(ROOT / study["output"] / "state.json")
        if state["status"] != "completed":
            print(f"Stopped at seed {study['seed']}: {state['status']}", flush=True)
            return
        rows = validate_results(study, curve.read(ROOT / study["output"] / "summary.json"))
        combined.append({"seed": study["seed"], "role": "replication", "results": rows})
    if args.run:
        discovery = curve.read(ROOT / "results/resnet20_compression/summary.json")
        if discovery["status"] != "completed":
            raise ValueError("Discovery comparison incomplete")
        discovery_rows = [r for r in discovery["results"] if r["fraction"] == 0.5]
        combined.insert(0, {"seed": config["discovery_seed"], "role": "discovery", "results": discovery_rows})
        write_json(ROOT / config["output"] / "summary.json",
                   {"status": "completed", "config": config, "studies": combined,
                    "updated_utc": now(), "test_evaluated": False})
        print("Replication complete; discovery and independent seeds are labelled separately.", flush=True)


if __name__ == "__main__":
    main()
