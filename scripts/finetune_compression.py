"""Equal-budget fine-tuning of every compact policy from the multi-source study (resumable per run)."""

import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from scripts.cifar_cnn import make_model as make_cnn6,write_json
from scripts.internal_replication import make_model as make_quaternion,sources_for
from scripts.multisource_compensation import COMPARISONS,aggregate,quaternion_cases,real_cases,to_grid
from src.cnn_training import prepare_batch,seed_everything
from src.compression.multisource import real_pair_fits,select_disjoint
from src.data.cifar_cached import file_sha256,load_cached_cifar
from src.evaluation.block_ablation import capture_blocks
from src.finetuning import finetune
from src.models.quaternion_conv_torch import QuaternionConv2d

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT / ".cache/matplotlib"))
SOURCES=["scripts/finetune_compression.py","src/finetuning.py","scripts/multisource_compensation.py","src/compression/multisource.py"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def same_maps(left,right,tolerance=1e-9):
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys()==right.keys() and all(same_maps(left[key],right[key],tolerance) for key in left)
    if isinstance(left,list):
        return isinstance(right,list) and len(left)==len(right) and all(same_maps(a,b,tolerance) for a,b in zip(left,right))
    if isinstance(left,(int,float)) and not isinstance(left,bool):
        return isinstance(right,(int,float)) and abs(left-right)<=tolerance*max(1.,abs(left))
    return left==right


def structure(model):
    return {"real_parameters":sum(p.numel() for p in model.parameters()),
            "quaternion_layers":sum(isinstance(module,QuaternionConv2d) for module in model.modules())}


def sources_of(ms):
    rep_config=read(ROOT / ms["replication_config"])
    sources=[{**source,"family":"quaternion"} for source in sources_for(rep_config)]
    for row in read(ROOT / ms["real_sources"] / "summary.json")["runs"]:
        path=str(row["path"]).replace("\\","/")
        sources.append({"architecture":"real_cnn6","family":"real","seed":row["seed"],"path":path,"config":read(ROOT / path / "summary.json")["config"]})
    return sources


def paired_effects(aggregates):
    """Same comparisons as the multi-source study, minus the functional oracle that is not fine-tuned."""
    lookup={(row["architecture"],row["removed"],row["policy"]):row for row in aggregates}
    extra={"quaternion":[("multi_q","paired_pruned")],"real":[("multi_real","paired_pruned")]}
    effects=[]
    for architecture,removed in sorted({(row["architecture"],row["removed"]) for row in aggregates}):
        family="real" if architecture=="real_cnn6" else "quaternion"
        for left,right in COMPARISONS[family]+extra[family]:
            if (architecture,removed,left) not in lookup or (architecture,removed,right) not in lookup:
                continue
            a,b=lookup[(architecture,removed,left)],lookup[(architecture,removed,right)]
            differences=[100*(x["accuracy"]-y["accuracy"]) for x,y in zip(a["validation"]["seed_values"],b["validation"]["seed_values"])]
            check=[100*(x["accuracy"]-y["accuracy"]) for x,y in zip(a["check_calibration"]["seed_values"],b["check_calibration"]["seed_values"])]
            effects.append({"architecture":architecture,"removed":removed,"left":left,"right":right,"validation_seed_pp":differences,
                            "validation_mean_pp":float(np.mean(differences)),"check_seed_pp":check,"check_mean_pp":float(np.mean(check)),
                            "positive_seeds":int(sum(value>0 for value in differences))})
    return effects


def wanted(name,policy,rep,removed,levels,study):
    if policy=="baseline":
        return removed==levels[0]
    return rep is None or rep<study["random_repetitions_finetuned"]


def render(output,summary):
    lookup={(row["architecture"],row["removed"],row["policy"]):row for row in summary["aggregates"]}
    before={(row["architecture"],row["removed"],row["policy"]):row for row in summary["before_aggregates"]}
    order=["baseline","random","paired_pruned","quaternion_exact","pair_merge","multi_q_random","multi_real_random","multi_full_projected","multi_q","multi_real"]
    config=summary["protocol"]["config"]
    rows=["# Equal-budget fine-tuning of compact policies","",
          f"Status: completed. Every compact model from results/multisource_compensation (first {config['random_repetitions_finetuned']} random victim sets only) and the "
          f"uncompressed baseline were fine-tuned for {config['epochs']} epochs on the 48,000 training images: AdamW, LR {config['learning_rate']} cosine to "
          f"{config['minimum_learning_rate']} per step, weight decay {config['weight_decay']}, batch {config['batch_size']}, original augmentation, BN in train mode. "
          "All policies of one training seed see identical batch order and augmentations. Primary outcome: final-epoch validation accuracy, "
          "no checkpoint selection. No official-test inference. Hyperparameters were fixed a priori, not tuned.","",
          "## Validation accuracy before and after fine-tuning","",
          "Before-values use the same random victim sets as the fine-tuned rows.","",
          "| Architecture | Removed | Policy | Real parameters | Before | After (mean ± seed SD) | Change (pp) |","| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for key in sorted(lookup,key=lambda key:(key[0],key[1],order.index(key[2]))):
        item,old=lookup[key],before[key]
        a,b=100*item["validation"]["mean_accuracy"],100*old["validation"]["mean_accuracy"]
        rows.append(f"| {key[0]} | {key[1]} | {key[2]} | {item['real_parameters']} | {b:.2f}% | {a:.2f}% ± {100*item['validation']['sample_sd_accuracy']:.2f} pp | {a-b:+.2f} |")
    rows+=["","## Paired effects after fine-tuning (left minus right)","",
           "| Architecture | Removed | Comparison | Validation per seed (pp) | Mean (pp) | Check mean (pp) | Positive seeds |","| --- | ---: | --- | --- | ---: | ---: | ---: |"]
    for item in summary["effects"]:
        rows.append(f"| {item['architecture']} | {item['removed']} | {item['left']} - {item['right']} | {', '.join(f'{v:+.2f}' for v in item['validation_seed_pp'])} | "
                    f"{item['validation_mean_pp']:+.2f} | {item['check_mean_pp']:+.2f} | {item['positive_seeds']}/3 |")
    rows+=["","## Matched-savings comparison after fine-tuning (CNN6)","","Drops are from each family's own fine-tuned baseline; descriptive only.","",
           "| Removal pair | Policy pair | Quaternion drop (pp) | Real drop (pp) |","| --- | --- | ---: | ---: |"]
    for q_removed,r_removed in ((4,8),(8,16)):
        for q_policy,r_policy in (("random","random"),("paired_pruned","paired_pruned"),("quaternion_exact","pair_merge"),("multi_q","multi_real"),("multi_q_random","multi_real_random")):
            q=100*(lookup[("cnn6",q_removed,q_policy)]["validation"]["mean_accuracy"]-lookup[("cnn6",q_removed,"baseline")]["validation"]["mean_accuracy"])
            r=100*(lookup[("real_cnn6",r_removed,r_policy)]["validation"]["mean_accuracy"]-lookup[("real_cnn6",r_removed,"baseline")]["validation"]["mean_accuracy"])
            rows.append(f"| {q_removed}/16 vs {r_removed}/32 | {q_policy} vs {r_policy} | {q:+.2f} | {r:+.2f} |")
    rows+=["","## Verification","",f"{summary['run_count']} fine-tuning runs. Rebuilt compact models matched the frozen multi-source maps; parameter counts and "
           "QuaternionConv2d layer counts are unchanged by fine-tuning in every run.","",
           "## Limits","","Three epochs at one fixed LR schedule; longer or tuned fine-tuning could change the ranking. Three random victim sets per seed. "
           "Validation also selected baseline checkpoints. One block, small CNNs, CIFAR-10. No novelty or general superiority claim.",""]
    (output / "REPORT.md").write_text("\n".join(rows),encoding="utf-8")


def run(study):
    output=ROOT / study["output"]
    if study["official_test_evaluation"]:
        raise ValueError("Official-test evaluation is not part of this study")
    ms=read(ROOT / study["multisource_config"]);ms_output=ROOT / ms["output"]
    frozen={(row["architecture"],row["seed"],row["removed"],row["case"]):row["maps"] for row in read(ms_output / "selection_frozen.json")["maps"]}
    protocol={"config":study,"source_hashes":{name:file_sha256(ROOT / name) for name in SOURCES},
              "multisource_summary_sha256":file_sha256(ms_output / "summary.json"),"official_test_evaluated":False}
    if (output / "protocol.json").exists() and read(output / "protocol.json")!=protocol:
        raise ValueError("Protocol/sources changed; preserve completed runs and use another output")
    write_json(output / "protocol.json",protocol)
    torch.set_num_threads(study["cpu_threads"]);device=torch.device(study["device"])
    (images,labels),unused_test=load_cached_cifar(ROOT / ms["archive"]);del unused_test
    stats=read(ROOT / ms["data_source"] / "data.json")["statistics"]
    mean=torch.tensor(stats["mean"],dtype=torch.float32,device=device).reshape(1,3,1,1)
    std=torch.tensor(stats["std"],dtype=torch.float32,device=device).reshape(1,3,1,1)
    rows=[];count=0
    for source in sources_of(ms):
        architecture,seed,quaternion=source["architecture"],source["seed"],source["family"]=="quaternion"
        seed_everything(seed)
        model=(make_quaternion(source) if quaternion else make_cnn6(source["config"],"real")).to(device).eval()
        model.load_state_dict(torch.load(ROOT / source["path"] / ("quaternion" if quaternion else "real") / "model.pt",map_location=device,weights_only=True)["model"])
        with np.load(ROOT / source["path"] / "split_indices.npz") as split: ids={"train":split["train"],"validation":split["validation"],"check_calibration":split["calibration"][1::2],"calibration":split["calibration"]}
        tensors={name:torch.as_tensor(images[value],device=device) for name,value in ids.items() if name!="calibration"}
        targets={name:torch.as_tensor(labels[value],device=device) for name,value in ids.items() if name!="calibration"}
        fit=to_grid(capture_blocks(model,prepare_batch(torch.as_tensor(images[ids["calibration"]],device=device),mean,std),ms["batch_size"])[1],quaternion)[::2]
        levels=ms["quaternion_removed_units"] if quaternion else ms["real_removed_channels"]
        if not quaternion:
            ordered=select_disjoint(real_pair_fits(fit,model.blocks[2][0].weight.detach().cpu().numpy()),max(levels))
        selected=read(ROOT / ms["replication_output"] / "selection_frozen.json")["pairs"]
        for removed in levels:
            if quaternion:
                cases=quaternion_cases(model,fit,selected[f"{architecture}:{seed}:{removed}"],removed,seed,ms)[0]
            else:
                cases=real_cases(model,fit,ordered[:removed],removed,seed,ms)
            for name,policy,rep,compact,maps in cases:
                if not wanted(name,policy,rep,removed,levels,study):
                    continue
                if maps is not None and not same_maps(maps,frozen[(architecture,seed,removed,name)]):
                    raise AssertionError(f"Rebuilt maps differ from the frozen multi-source study: {architecture}/{seed}/{removed}/{name}")
                stored_removed=0 if policy=="baseline" else removed
                folder=output / architecture / f"seed_{seed}" / f"removed_{stored_removed:02d}" / name
                if not (folder / "metrics.json").exists():
                    start=time.perf_counter();before=structure(compact)
                    tuned,history=finetune(compact,tensors,targets,mean,std,study,seed,("validation","check_calibration"))
                    after=structure(tuned)
                    if before!=after: raise AssertionError(f"Fine-tuning changed the architecture: {folder}")
                    folder.mkdir(parents=True,exist_ok=True)
                    torch.save({"model":{key:value.detach().cpu() for key,value in tuned.state_dict().items()}},folder / "model.pt")
                    write_json(folder / "metrics.json",{"architecture":architecture,"seed":seed,"removed":stored_removed,"case":name,"policy":policy,
                                                       "random_repetition":rep,**after,"history":history,"final":{split:history[-1][split] for split in ("validation","check_calibration")},
                                                       "checkpoint_sha256":file_sha256(folder / "model.pt"),"official_test_evaluated":False})
                    print(f"{architecture}/{seed}/{stored_removed}/{name}: val {100*history[-1]['validation']['accuracy']:.2f}% ({time.perf_counter()-start:.0f} s)",flush=True)
                result=read(folder / "metrics.json");count+=1
                for level in (levels if policy=="baseline" else [removed]):
                    rows.append({"architecture":architecture,"seed":seed,"removed":level,"case":name,"policy":policy,
                                 "real_parameters":result["real_parameters"],"evaluation":result["final"]})
    keep={(row["architecture"],row["seed"],row["removed"],row["case"]) for row in rows}
    before_rows=[row for row in read(ms_output / "summary.json")["rows"] if (row["architecture"],row["seed"],row["removed"],row["case"]) in keep]
    summary={"status":"completed","protocol":protocol,"run_count":count,"rows":rows,"aggregates":aggregate(rows,study["seeds"]),
             "before_aggregates":aggregate(before_rows,study["seeds"]),"official_test_evaluated":False}
    summary["effects"]=paired_effects(summary["aggregates"])
    write_json(output / "summary.json",summary);render(output,summary)
    return summary


if __name__=="__main__":
    run(read(ROOT / "configs/finetune_compression.json"))
