"""Low-label POST-compression repair; never an end-to-end few-shot claim.

Online hard targets come from the CURRENT compact model. A frozen full
teacher is a separate resource control. No new selection or compensation fit.
"""

import argparse
from copy import deepcopy
import importlib.metadata as metadata
from pathlib import Path
import platform
import sys

import numpy as np
import torch

from scripts import cifar_cnn as trainer
from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from src.cnn_training import evaluate, prepare_batch, seed_everything
from src.compression.multilayer import compact_site, units
from src.data.cifar_cached import file_sha256, load_cached_cifar
from src.pseudolabel_repair import budget_partition, repair

ROOT=Path(__file__).resolve().parents[1]
CONFIG="configs/pseudolabel_repair.json"
SOURCES=list(dict.fromkeys(["scripts/pseudolabel_repair.py","src/pseudolabel_repair.py",*trainer.SOURCES,*curve.SOURCES]))
read=curve.read


def compact_path(study,seed):
    return f"{study['parent_curve']}/compact/cnn6/seed_{seed}/f_{round(1000*study['fraction']):03d}/greedy_comp.pt"


def input_paths(study):
    names=[f"{study['parent_curve']}/summary.json"]
    for seed in study["seeds"]:
        source=study["sources"][str(seed)]
        names.extend(f"{source}/{name}" for name in ("summary.json","data.json","split_indices.npz","quaternion/model.pt","quaternion/metrics.json"))
        names.extend([f"{study['parent_curve']}/phase_a/cnn6_seed_{seed}.json",compact_path(study,seed)])
        names.append(read(ROOT / source / "summary.json")["config"]["archive"])
    return sorted(set(names))


def initial_models(study,seed,device):
    source=ROOT / study["sources"][str(seed)]
    config=read(source / "summary.json")["config"]
    teacher=trainer.make_model(config,"quaternion").to(device).eval()
    teacher.load_state_dict(torch.load(source / "quaternion/model.pt",map_location=device,weights_only=True)["model"])
    audit=read(source / "quaternion/metrics.json")["first_update"]["checks"]
    if not all(audit.values()): raise AssertionError("Original quaternion model failed its training audit")
    phase=read(ROOT / study["parent_curve"] / "phase_a" / f"cnn6_seed_{seed}.json")
    records={name:next(r["sites"] for r in phase["records"] if r["fraction"]==study["fraction"] and r["case"]==name)
             for name in study["initializations"]}
    if [[r["victims"] for r in records[name]] for name in study["initializations"]].count([r["victims"] for r in records[study["initializations"][0]]])!=len(records):
        raise AssertionError("Compensation controls must share every victim")
    pruned=deepcopy(teacher)
    for record in records["greedy_pruned"]:
        site=record["site"];victims=set(record["victims"])
        pruned=compact_site(pruned,site,[i for i in range(units(pruned,site)) if i not in victims])
    narrow=curve.reduced_config(config,"cnn6",study["fraction"])
    compact=trainer.make_model(narrow,"quaternion").to(device).eval()
    compact.load_state_dict(torch.load(ROOT / compact_path(study,seed),map_location=device,weights_only=True)["model"])
    restored=trainer.make_model(narrow,"quaternion").to(device).eval();restored.load_state_dict(pruned.state_dict())
    models={"greedy_pruned":restored,"greedy_comp":compact}
    for model in models.values():
        if sum(p.numel() for p in model.parameters())!=42034: raise AssertionError("Expected actual 25% narrowed architecture")
    expected={name:next(r for r in phase["rows"] if r["fraction"]==study["fraction"] and r["case"]==name) for name in models}
    return teacher,models,expected,records


def diagnostic(logits,targets,threshold):
    probabilities=torch.as_tensor(logits).softmax(1)
    confidence,prediction=probabilities.max(1);accepted=confidence>=threshold
    truth=torch.as_tensor(targets,device="cpu")
    return {"coverage":float(accepted.float().mean()),"accepted":int(accepted.sum()),"examples":len(targets),
            "accepted_accuracy":float((prediction[accepted]==truth[accepted]).float().mean()) if bool(accepted.any()) else None,
            "accepted_predicted_class_counts":torch.bincount(prediction[accepted],minlength=10).tolist()}


@torch.no_grad()
def probabilities(model,images,mean,std):
    return torch.cat([model(prepare_batch(images[start:start+128],mean,std)).float().softmax(1) for start in range(0,len(images),128)])


def summarize(study,protocol,rows,status,output):
    aggregates=[]
    for budget in study["repair_labels_per_class"]:
        for method in study["methods"]:
            for initialization in study["initializations"]:
                selected=[r for r in rows if (r["repair_labels_per_class"],r["method"],r["initialization"])==(budget,method,initialization)]
                if not selected: continue
                values=[r["history"][-1]["validation"]["accuracy"] for r in selected]
                aggregates.append({"repair_labels_per_class":budget,"method":method,"initialization":initialization,
                    "seeds":[r["seed"] for r in selected],"seed_validation_accuracy":values,
                    "mean_validation_accuracy":float(np.mean(values)),"sample_sd":float(np.std(values,ddof=1)) if len(values)>1 else None,
                    "actual_optimizer_updates":[r["optimizer_updates"] for r in selected],
                    "pseudo_accepted":[r["pseudo_accepted"] for r in selected]})
    effects=[]
    lookup={(r["seed"],r["repair_labels_per_class"],r["method"],r["initialization"]):r for r in rows}
    for budget in study["repair_labels_per_class"]:
        for method in study["methods"]:
            seeds=[s for s in study["seeds"] if all((s,budget,method,i) in lookup for i in study["initializations"])]
            differences=[100*(lookup[s,budget,method,"greedy_comp"]["history"][-1]["validation"]["accuracy"]-
                              lookup[s,budget,method,"greedy_pruned"]["history"][-1]["validation"]["accuracy"]) for s in seeds]
            if seeds: effects.append({"repair_labels_per_class":budget,"method":method,"seeds":seeds,"comp_minus_pruned_pp":differences,
                                       "mean_pp":float(np.mean(differences)),"positive_seeds":sum(v>0 for v in differences)})
    summary={"status":status,"protocol":protocol,"rows":rows,"aggregates":aggregates,"paired_compensation_effects":effects,
             "run_count":len(rows),"official_test_evaluated":False,"smoke_check":study.get("smoke_check",False)}
    write_json(output / "summary.json",summary)
    lines=["# Low-label POST-compression repair","","## Hypotheses and scope","",
        "Does compensation improve recovery when post-compression repair labels are scarce? Can current-student pseudo-labels help, or reinforce compression errors? No novel algorithm is proposed.","",
        "The original teacher was fully supervised and validation-selected. Labels per class count ONLY labels exposed to repair updates. Diagnostic validation/check labels and inherited normalization/BN statistics are outside that budget. This is not end-to-end few-shot or unsupervised training. Primary budgets: 5/20; zero is a boundary control. Same-distribution CIFAR simulation; domain shift and an actual deployment are untested.","",
        "Same frozen victims and physical 42,034-parameter architecture, with/without compensation, no fitting again. Online self-pseudo-labels come from the current compact model; the frozen full teacher is an additional-resource control. Confidence filtering is established (Sohn2020FixMatch), but crop/flip-only views are not a FixMatch reproduction or the student's exact pipeline.","",
        "Nested balanced labeled subsets and one disjoint 4096-image unlabeled pool per seed; three paired seeds, not many independent dataset replications. Fixed 1000 opportunities, final-step outcome, snapshots 0/200/1000; AdamW 1e-4 cosine to 1e-6, confidence .95. BN buffers and dropout fixed; BN affine remains trainable. Label/unlabeled batches and augmentation streams paired across policies. A batch with no usable target produces no optimizer step (including no weight decay). Teacher inference is an extra cost, not included in the count of update opportunities.","",
        "These repair settings differ from earlier full-label three-epoch train-mode fine-tuning; no direct label-efficiency claim follows from that cross-study comparison.","",
        "## Measured results","",f"State: {status}; completed runs: {len(rows)}. Smoke checks are infrastructure checks only.","",
        "| Repair labels/class | Method | Initialization | Validation mean | Per-seed actual updates |",
        "| ---: | --- | --- | ---: | --- |"]
    for row in aggregates:
        lines.append(f"| {row['repair_labels_per_class']} | {row['method']} | {row['initialization']} | {100*row['mean_validation_accuracy']:.2f}% | {row['actual_optimizer_updates']} |")
    lines.extend(["","## Interpretation limits","",
        "Keep all seeds, unfavorable effects and zero-acceptance/no-update runs. Compare final accuracy and CE with each initialization's step-zero values. Paired differences are descriptive with three seeds; no claim of superiority or deployment value follows merely from running this protocol. Full per-seed histories, pseudo-label coverage/precision and prediction files accompany the summary. No new official-test inference."])
    (output / "REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(study,config_name=CONFIG):
    if study["fraction"]!=.25 or study["architecture"]!="cnn6" or study["activation"]!="split_relu":
        raise ValueError("This registered pilot is limited to ReLU CNN6 at 25% unit removal")
    torch.set_num_threads(study["cpu_threads"]);seed_everything(study["seeds"][0])
    device=torch.device(study["device"]);output=ROOT / study["output"];output.mkdir(parents=True,exist_ok=True)
    if read(ROOT / study["parent_curve"] / "summary.json")["status"]!="completed": raise ValueError("Parent curve is incomplete")
    hashes={name:file_sha256(ROOT / name) for name in sorted(set([*SOURCES,config_name,*input_paths(study)]))}
    protocol={"config":study,"hashes":hashes,"environment":{"python":platform.python_version(),"torch":metadata.version("torch"),
        "numpy":metadata.version("numpy"),"device":str(device),"gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None},
        "official_test_evaluated":False}
    if (output / "protocol.json").exists() and read(output / "protocol.json")!=protocol: raise ValueError("Repair protocol/inputs changed")
    write_json(output / "protocol.json",protocol)
    source=ROOT / study["sources"][str(study["seeds"][0])]
    original=read(source / "summary.json")["config"]
    (images,targets),unused_test=load_cached_cifar(ROOT / original["archive"]);del unused_test
    rows=[];summarize(study,protocol,rows,"running",output)
    for seed in study["seeds"]:
        source=ROOT / study["sources"][str(seed)]
        with np.load(source / "split_indices.npz") as saved:
            ids={"train":saved["train"],"validation":saved["validation"],"check_calibration":saved["calibration"][1::2]}
        label_ids,unlabeled_ids=budget_partition(ids["train"],targets,study["repair_labels_per_class"],study["unlabeled_pool_size"],seed+210000)
        folder=output / f"seed_{seed}";folder.mkdir(parents=True,exist_ok=True)
        partitions={**ids,"unlabeled":unlabeled_ids,**{f"labeled_{b}":v for b,v in label_ids.items()}}
        partition_path=folder / "partition_ids.npz"
        if partition_path.exists():
            with np.load(partition_path) as saved:
                if set(saved.files)!=set(partitions) or any(not np.array_equal(saved[k],v) for k,v in partitions.items()):
                    raise ValueError("Registered data partition changed")
        else: np.savez_compressed(partition_path,**partitions)
        stats=read(source / "data.json")["statistics"]
        mean=torch.tensor(stats["mean"],device=device,dtype=torch.float32).reshape(1,3,1,1)
        std=torch.tensor(stats["std"],device=device,dtype=torch.float32).reshape(1,3,1,1)
        eval_ids={k:v[:16] if study.get("smoke_check") else v for k,v in ids.items() if k!="train"}
        tensors={k:torch.as_tensor(images[v],device=device) for k,v in eval_ids.items()}
        labels={k:torch.as_tensor(targets[v],device=device) for k,v in eval_ids.items()}
        unlabeled=torch.as_tensor(images[unlabeled_ids],device=device)
        teacher,models,expected,records=initial_models(study,seed,device)
        write_json(folder / "initialization_manifest.json",{"records":records,"expected_parent_metrics":expected,
            "partition_sha256":file_sha256(partition_path),"real_trainable_parameters":42034,"fully_supervised_teacher":True})
        teacher_path=folder / "teacher_probabilities.pt"
        if not teacher_path.exists():
            value=probabilities(teacher,unlabeled,mean,std).cpu()
            torch.save({"probabilities":value,"ids":torch.as_tensor(unlabeled_ids.copy()),"teacher_sha256":hashes[f"{study['sources'][str(seed)]}/quaternion/model.pt"]},teacher_path)
        cache=torch.load(teacher_path,map_location="cpu",weights_only=True)
        if cache["teacher_sha256"]!=hashes[f"{study['sources'][str(seed)]}/quaternion/model.pt"] or not np.array_equal(cache["ids"].numpy(),unlabeled_ids):
            raise ValueError("Frozen teacher cache has different provenance")
        teacher_probabilities=cache["probabilities"].to(device);del teacher
        for initialization,initial in models.items():
            initial_metrics={split:evaluate(initial,tensors[split],labels[split],mean,std)[0] for split in eval_ids}
            if not study.get("smoke_check"):
                for split,measured in initial_metrics.items():
                    prior=expected[initialization]["evaluation"][split]
                    if measured["accuracy"]!=prior["accuracy"] or abs(measured["cross_entropy"]-prior["cross_entropy"])>1e-4:
                        raise AssertionError("Initial model does not reproduce its frozen parent outcome")
            for budget in study["repair_labels_per_class"]:
                selected=label_ids[budget]
                labeled=torch.as_tensor(images[selected],device=device);human_targets=torch.as_tensor(targets[selected],device=device)
                for method in study["methods"]:
                    target=folder / initialization / f"labels_{budget}" / method;target.mkdir(parents=True,exist_ok=True)
                    context={"seed":seed,"initialization":initialization,"repair_labels_per_class":budget,"partition_sha256":file_sha256(partition_path),
                        "protocol_sha256":file_sha256(output / "protocol.json"),"teacher_cache_sha256":file_sha256(teacher_path)}
                    completed=target / "metrics.json"
                    if completed.exists():
                        result=read(completed)
                        if result["context"]!=context or result["method"]!=method or result["final_model_sha256"]!=file_sha256(target / "model.pt"):
                            raise ValueError("Completed repair case changed")
                    else:
                        model=deepcopy(initial);buffers={k:v.clone() for k,v in model.named_buffers()}

                        def callback(current,step):
                            measurements={};logits={}
                            for split in eval_ids:
                                measurements[split],logits[split]=evaluate(current,tensors[split],labels[split],mean,std)
                            measurements["pseudo_diagnostic_on_check_only"]=diagnostic(logits["check_calibration"],targets[eval_ids["check_calibration"]],study["repair"]["confidence_threshold"])
                            np.savez_compressed(target / f"predictions_step_{step:04d}.npz",**{f"{k}_logits":v for k,v in logits.items()},
                                                **{f"{k}_ids":v for k,v in eval_ids.items()})
                            return measurements

                        result=repair(model,labeled,human_targets,unlabeled,mean,std,study["repair"],seed,method,target,context,callback,teacher_probabilities)
                        if any(not torch.equal(value,dict(model.named_buffers())[k]) for k,value in buffers.items()):
                            raise AssertionError("BN buffers changed during fixed-statistics repair")
                        if result["first_update"] is not None:
                            kernels=[r for k,r in result["first_update"]["parameters"].items() if k.endswith("kernel")]
                            if len(kernels)!=6 or not any(r["maximum_update"]>0 for r in kernels):
                                raise AssertionError("Quaternion repair failed to update tracked kernels")
                        result.update(context=context,method=method,seed=seed,initialization=initialization,repair_labels_per_class=budget,
                            human_labels_exposed_to_repair=len(selected),real_trainable_parameters=42034,bn_buffers_fixed=True,
                            final_model_sha256=file_sha256(target / "model.pt"))
                        write_json(completed,result)
                    rows.append(result);summarize(study,protocol,rows,"running",output)
                    print(f"repair seed={seed} init={initialization} labels/class={budget} method={method}: "
                          f"val={100*result['history'][-1]['validation']['accuracy']:.2f}%, updates={result['optimizer_updates']}",flush=True)
    expected_count=len(study["seeds"])*len(study["initializations"])*len(study["methods"])*len(study["repair_labels_per_class"])
    if len(rows)!=expected_count: raise AssertionError("Incomplete registered repair grid")
    if any(file_sha256(ROOT / name)!=digest for name,digest in hashes.items()): raise ValueError("Repair sources/inputs changed during execution")
    summarize(study,protocol,rows,"completed",output)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("config",nargs="?",default=CONFIG);args=parser.parse_args()
    run(read(ROOT / args.config),args.config)
