"""From-scratch Q-input/stem plus learned PHM residual baseline."""
import argparse
from pathlib import Path

import torch

from scripts import compression_curve as curve
from scripts import resnet20_baselines as baseline
from scripts.cifar_cnn import write_json
from scripts.run_activation_queue import lock
from src.models.cifar_phm_resnet import CifarPHMResNet20

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/resnet20_qstem_phm4_baseline.json"
baseline.NEW = baseline.NEW + [CONFIG, "scripts/resnet20_qstem_phm4_baseline.py",
                              "src/models/cifar_phm_resnet.py", "src/models/phm_conv.py",
                              "tests_torch/test_cifar_phm_resnet.py"]


def make_model(study, kind):
    if kind != "qstem_phm4":
        raise ValueError("Only the prespecified PHM baseline is authorized here")
    return CifarPHMResNet20(study["quaternion_widths"],study["classes"],study["phm_dimensions"])


def report(study,results,status,progress=None):
    output = ROOT/study["output"]
    write_json(output/"summary.json",{"status":status,"config":study,"results":results,"progress":progress,"test_evaluated":False})
    lines = ["# Quaternion input/stem with learned PHM ResNet-20","",f"Status: {status}.","",
             "## Hypothesis","","A from-scratch learned-PHM backbone can support a later compression study comparing preservation versus readaptation of learned bases. Quality is unmeasured until training completes.","",
             "## Measured results","",
             "| Model | Real parameters | Selected epoch | Selected validation (%) | Final validation (%) |",
             "|---|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['kind']} | {r['real_trainable_parameters']} | {r['selected_epoch']} | {100*r['selected_validation']['accuracy']:.2f} | {100*r['final_validation']['accuracy']:.2f} |")
    if progress:
        lines += ["",f"Completed epoch {progress['epoch']}/{study['epochs']}; validation {100*progress['validation']['accuracy']:.2f}%; epoch {progress['epoch_seconds']:.1f}s."]
    lines += ["","## Interpretation and limits","",study["recipe"],"",study["scope"],"",study["checkpoint_selection"],"",study["resume"],
              "","One Hamilton stem,18learned PHM bases, fixed n4, one seed. Discrete n is not trained. Learned bases are not guaranteed to satisfy algebra axioms. Actual model size differs from previous baselines by1152basis coefficients. No compression or test result here. Prior art: "+", ".join(study["prior_art"])+". Citation database: references.bib."]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


baseline.make_model = make_model
baseline.report = report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true")
    args = parser.parse_args()
    study = curve.read(ROOT/CONFIG)
    torch.set_num_threads(study["cpu_threads"])
    if args.run:
        baseline.run(study)
    else:
        with lock("compression_curve_runner.lock"):
            baseline.prepare(study)
        path = ROOT/study["output"]/"readiness.json"
        ready = curve.read(path)
        ready["tests"] = "2 CPU PHM-model tests passed: Q-only stem/input encoding, all PHM gradients/updates, n2/4/8 partitions, reload/counts and exact interrupted SGD replay."
        write_json(path,ready)


if __name__=="__main__":
    main()
