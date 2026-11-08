"""Run two independent-seed ResNet baseline pairs with the frozen training recipe."""
import argparse
from copy import deepcopy
from pathlib import Path

import torch

from scripts import compression_curve as curve
from scripts import resnet20_baselines as baseline
from scripts.cifar_cnn import write_json
from scripts.paper_a_controls import now

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/resnet20_replication_baselines.json"
baseline.NEW = baseline.NEW + [CONFIG, "scripts/resnet20_replication_baselines.py"]
original_report = baseline.report


def report(study, results, status, progress=None):
    original_report(study, results, status, progress)
    path = ROOT / study["output"] / "REPORT.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace("One seed2026 per family", f"One seed{study['seed']} per family")
    path.write_text(text, encoding="utf-8")


baseline.report = report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = curve.read(ROOT / CONFIG)
    base = curve.read(ROOT / config["base_config"])
    torch.set_num_threads(base["cpu_threads"])
    for seed in config["seeds"]:
        study = deepcopy(base)
        study["seed"] = seed
        study["output"] = config["output_template"].format(seed=seed)
        study["scope"] = config["scope"]
        study["resume"] = config["pause"]
        output = ROOT / study["output"]
        if not (output / "readiness.json").exists():
            baseline.prepare(study)
        if not args.run:
            continue
        baseline.run(study)
        state = curve.read(output / "state.json")
        if state["status"] != "completed":
            print(f"Stopped at seed {seed}: {state['status']}", flush=True)
            return
    if args.run:
        write_json(ROOT / "results/resnet20_replication_baselines/completion.json",
                   {"status": "completed", "seeds": config["seeds"], "updated_utc": now(),
                    "test_evaluated": False})


if __name__ == "__main__":
    main()
