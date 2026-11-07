"""Train two activation controls, then reuse frozen multi-site compression.

No new activation is proposed. Scope: within-Q sensitivity on CIFAR, not a
causal test of quaternion-specific superiority or component coupling alone.
"""

import csv
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import torch

from scripts import cifar_cnn as trainer
from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from src.data.cifar_cached import file_sha256,load_cached_cifar
from src.models.activation_cnn import make_activation_model

ROOT=Path(__file__).resolve().parents[1]
CONFIG="configs/activation_sensitivity.json"
SOURCES=["scripts/activation_sensitivity.py","src/models/activation_cnn.py",*curve.SOURCES,*trainer.SOURCES]


def read(path):
    return curve.read(path)


def training_config(study,activation,seed):
    original=read(ROOT / study["baseline_source"] / "summary.json")["config"]
    if original["epochs"]!=study["training_epochs"]:
        raise ValueError("Activation baselines must retain the registered training budget")
    return {**original,"seed":seed,"models":["quaternion"],"activation":activation,
            "negative_slope":study["negative_slope"],"activation_epsilon":study["activation_epsilon"]}


def train(config,tensors,labels,mean,std,folder,hashes):
    """Use the frozen trainer; handle only its known completed-resume cleanup."""
    with patch.object(trainer,"make_model",make_activation_model),patch.object(trainer,"source_hashes",lambda:hashes):
        try:
            result=trainer.train_one(config,"quaternion",tensors,labels,mean,std,folder,True)
        except UnboundLocalError as error:
            if "saved" not in str(error) or not (folder / "metrics.json").exists():
                raise
            result=read(folder / "metrics.json")
            last=torch.load(folder / "last.pt",map_location="cpu",weights_only=True)
            selected=torch.load(folder / "model.pt",map_location="cpu",weights_only=True)
            if last["epoch"]!=config["epochs"] or result["source_hashes"]!=hashes or not all(torch.equal(v,last["best_state"][k]) for k,v in selected["model"].items()):
                raise
            write_json(folder / "resume_cleanup_compatibility.json",{"handled_exception":str(error),"numeric_artifacts_verified":True,"extra_optimizer_updates":0})
    last=torch.load(folder / "last.pt",map_location="cpu",weights_only=True)
    with (folder / "training.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(last["history"][0]));writer.writeheader();writer.writerows(last["history"])
    return result


def train_baselines(study,activation,hashes):
    source=ROOT / study["baseline_source"]
    original=read(source / "summary.json")["config"]
    (images,targets),unused_test=load_cached_cifar(ROOT / original["archive"]);del unused_test
    with np.load(source / "split_indices.npz") as saved:
        ids={name:saved[name] for name in ("train","validation","calibration")}
    device=torch.device(study["device"]);stats=read(source / "data.json")["statistics"]
    mean=torch.tensor(stats["mean"],dtype=torch.float32,device=device).reshape(1,3,1,1)
    std=torch.tensor(stats["std"],dtype=torch.float32,device=device).reshape(1,3,1,1)
    tensors={name:torch.as_tensor(images[value],device=device) for name,value in ids.items()}
    labels={name:torch.as_tensor(targets[value],device=device) for name,value in ids.items()}
    records=[]
    for seed in study["seeds"]:
        config=training_config(study,activation,seed)
        folder=ROOT / study["output"] / activation / f"seed_{seed}";folder.mkdir(parents=True,exist_ok=True)
        for name in ("data.json","split_indices.npz"):
            if not (folder / name).exists(): shutil.copyfile(source / name,folder / name)
        print(f"ACTIVATION {activation}, seed {seed}: training/resuming",flush=True)
        report=train(config,tensors,labels,mean,std,folder / "quaternion",hashes)
        if not all(report["first_update"]["checks"].values()) or report["test_evaluated"] or report["real_trainable_parameters"]!=73898:
            raise AssertionError("Actual first-update, no-test or parameter-budget gate failed")
        write_json(folder / "summary.json",{"config":config,"models":[report],"official_test_evaluated":False})
        records.append({"architecture":"cnn6","seed":seed,"path":str(folder.relative_to(ROOT)).replace("\\","/"),"config":config})
    return records


def run_curve(study,activation,sources):
    config=read(ROOT / study["curve_configs"][activation])
    original_loader,original_random=curve.load_data,curve.random_victims

    def loader(config,device):
        images,labels,mean,std,_=original_loader(config,device)
        return images,labels,mean,std,sources

    def random_victims(settings,seed,fraction_index,rep,site_index,width,count):
        parent_index=study["parent_random_fraction_indices"][fraction_index]
        return original_random(settings,seed,parent_index,rep,site_index,width,count)

    # All actual fits, folding, inference and fine-tuning retain the original
    # implementations. Patches select the new factory/sources and align masks
    # with the already-computed ReLU reference's fraction indices.
    with patch.object(curve,"factory",lambda dataset,arch:make_activation_model),patch.object(curve,"load_data",loader),patch.object(curve,"random_victims",random_victims),patch.object(curve,"SOURCES",list(dict.fromkeys([*SOURCES,CONFIG,study["curve_configs"][activation]]))):
        curve.run(config)


def compare(study):
    paths={"split_relu":ROOT / study["relu_reference"] / "summary.json"}
    paths.update({name:ROOT / study["output"] / name / "compression_curve/summary.json" for name in study["activations"]})
    tables=[];audits={};folds={}
    for activation,path in paths.items():
        result=read(path)
        if result["status"]!="completed": raise ValueError("Do not compare incomplete activation curves")
        if activation!="split_relu":
            diagnostic=result["diagnostics"]
            folds[activation]={"site_checks":diagnostic["fold_checks"],"maximum_error":diagnostic["max_fold_error"]}
            if not diagnostic["fold_checks"] or diagnostic["max_fold_error"]>result["protocol"]["config"]["equivalence_tolerance"]:
                raise AssertionError("Activation compression fold checks failed")
        rows=[r for r in result["rows"] if r["architecture"]=="cnn6" and r["fraction"] in [0.,*study["fractions"]] and r["stage"] in ("before","after")]
        groups=curve.aggregate(rows,study["seeds"])
        for row in groups:
            baseline=next(r for r in groups if r["fraction"]==0 and r["stage"]==row["stage"])
            selected=[r for r in rows if (r["fraction"],r["policy"],r["stage"])==(row["fraction"],row["policy"],row["stage"])]
            ce=[np.mean([r["evaluation"]["validation"]["cross_entropy"] for r in selected if r["seed"]==seed]) for seed in study["seeds"]]
            tables.append({"activation":activation,**row,"validation_mean_ce":float(np.mean(ce)),
                           "drop_from_own_baseline_pp":100*(row["validation"]["mean_accuracy"]-baseline["validation"]["mean_accuracy"])})
        if activation!="split_relu":
            for seed in study["seeds"]:
                folder=ROOT / study["output"] / activation / f"seed_{seed}/quaternion"
                metrics=read(folder / "metrics.json");last=torch.load(folder / "last.pt",map_location="cpu",weights_only=True)
                exported=torch.load(folder / "model.pt",map_location="cpu",weights_only=True)
                audits[f"{activation}:{seed}"]={"fifty_epochs":last["epoch"]==50 and len(last["history"])==50,
                    "actual_first_update":all(metrics["first_update"]["checks"].values()),
                    "no_test":not metrics["test_evaluated"],"no_added_trainable_parameters":metrics["real_trainable_parameters"]==73898,
                    "selected_state_exported":all(torch.equal(v,last["best_state"][k]) for k,v in exported["model"].items()),
                    "selection_by_validation_ce":last["best_epoch"]==exported["best_epoch"]==metrics["best_epoch"] and last["best_loss"]<=min(r["validation_loss"] for r in last["history"])+1e-12,
                    "full_resume_state":all(k in last for k in ("optimizer","scheduler","scaler","order_rng","augment_rng","torch_rng","cuda_rng")),
                    "optimizer_attempts":last["updates"]==metrics["optimizer_updates_attempted"]==18750}
    if not all(value for row in audits.values() for value in row.values()): raise AssertionError("Activation training audit failed")
    output=ROOT / study["output"]
    effects=[]
    for row in tables:
        if row["activation"]=="split_relu": continue
        reference=next(r for r in tables if (r["activation"],r["fraction"],r["policy"],r["stage"])==("split_relu",row["fraction"],row["policy"],row["stage"]))
        differences=[100*(a["accuracy"]-b["accuracy"]) for a,b in zip(row["validation"]["seed_values"],reference["validation"]["seed_values"])]
        effects.append({"activation":row["activation"],"fraction":row["fraction"],"policy":row["policy"],"stage":row["stage"],
                        "seed_differences_pp":differences,"mean_difference_pp":float(np.mean(differences))})
    summary={"status":"completed","config":study,"aggregates":tables,"paired_effects_vs_relu":effects,"training_audits":audits,"fold_checks":folds,"official_test_evaluated":False}
    write_json(output / "summary.json",summary)
    lines=["# Activation sensitivity: split LeakyReLU and coupled MagnitudeTanh","","## Hypothesis and fixed scope","",
           "Test whether the earlier compression conclusions depend on activation choice, within quaternion CNN6 on CIFAR. "
           "Three seeds, unchanged baseline 50-epoch trainer and parameter count; remove 25%/50% at every convolution output. "
           "Fit/folding policies and three-epoch fine-tuning match the existing ReLU reference, including the random victim streams. "
           "No activation-specific scratch runs or new official-test inference. MagnitudeTanh is prior art "
           "(PoppelbaumSchwung2024Activations, Eq. 21); no new activation/method is proposed.","","## Measured results","",
           "| Activation | Fraction | Policy | Stage | Real params | Val mean ± seed SD | Mean CE | Change from own baseline (pp) |",
           "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in tables:
        lines.append(f"| {row['activation']} | {row['fraction']} | {row['policy']} | {row['stage']} | {row['real_parameters']} | {100*row['validation']['mean_accuracy']:.2f}% ± {100*row['validation']['sample_sd_accuracy']:.2f} | {row['validation_mean_ce']:.6f} | {row['drop_from_own_baseline_pp']:+.2f} |")
    lines += ["","## Paired differences from ReLU, including unfavorable seeds","",
              "| Activation | Fraction | Policy | Stage | Per-seed difference (pp) | Mean (pp) |",
              "| --- | ---: | --- | --- | --- | ---: |"]
    for row in effects:
        differences=", ".join(f"{v:+.2f}" for v in row["seed_differences_pp"])
        lines.append(f"| {row['activation']} | {row['fraction']} | {row['policy']} | {row['stage']} | {differences} | {row['mean_difference_pp']:+.2f} |")
    lines += ["","## Interpretation limits","",
              "Activation changes baseline learning as well as compression behavior. Compare both absolute accuracy and loss from "
              "each activation's own baseline; preserve unfavorable seeds. The coupled Tanh also changes boundedness/saturation, "
              "so its contrast with ReLU does not isolate component coupling. There is no matched split-Tanh or real-valued "
              "coupled control in this bounded pilot. BN and pooling remain component-wise; no full-model equivariance is implied. "
              "One dataset/two removal levels and one fixed slope/epsilon do not establish general superiority. "
              "All per-policy artifacts and feature-fold checks remain in the two compression_curve subdirectories.",""]
    (output / "REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    return summary


def run():
    study=read(ROOT / CONFIG);output=ROOT / study["output"]
    if study["official_test_evaluation"] or study["dataset"]!="cifar" or study["fractions"]!=[.25,.5] or study["activations"]!=["split_leaky_relu","magnitude_tanh"]:
        raise ValueError("Only the registered bounded two-activation sensitivity study is implemented")
    parent=read(ROOT / "configs/compression_curve.json")
    if [parent["fractions"].index(f) for f in study["fractions"]]!=study["parent_random_fraction_indices"]:
        raise ValueError("Random-victim alignment with the ReLU reference changed")
    hashes={name:file_sha256(ROOT / name) for name in sorted(set([*SOURCES,CONFIG,*study["curve_configs"].values()]))}
    source=ROOT / study["baseline_source"]
    protected={str((source / name).relative_to(ROOT)).replace("\\","/"):file_sha256(source / name) for name in ("summary.json","data.json","split_indices.npz")}
    archive=read(source / "summary.json")["config"]["archive"]
    protected[archive]=file_sha256(ROOT / archive)
    if protected[archive]!=read(source / "data.json")["archive_sha256"]:
        raise ValueError("CIFAR data archive changed")
    protected[str(Path(study["relu_reference"]) / "summary.json").replace("\\","/")]=file_sha256(ROOT / study["relu_reference"] / "summary.json")
    protocol={"config":study,"source_hashes":hashes,"protected_inputs":protected,"official_test_evaluated":False}
    if (output / "protocol.json").exists() and read(output / "protocol.json")!=protocol:
        raise ValueError("Activation protocol/inputs changed; preserve this study")
    write_json(output / "protocol.json",protocol)
    torch.set_num_threads(study["cpu_threads"])
    for activation in study["activations"]:
        sources=train_baselines(study,activation,hashes)
        run_curve(study,activation,sources)
    if any(file_sha256(ROOT / name)!=digest for name,digest in {**hashes,**protected}.items()):
        raise AssertionError("Protected activation source changed during the study")
    compare(study)


if __name__=="__main__":
    run()
