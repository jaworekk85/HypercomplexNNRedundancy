"""Multi-source compensation at block 2: Hamilton-restricted vs unconstrained, plus a real-CNN control."""

from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.cifar_cnn import make_model as make_cnn6,write_json
from scripts.internal_replication import make_model as make_quaternion,sources_for
from src.cnn_training import evaluate,prepare_batch,seed_everything
from src.compression.internal_merging import merge_internal,substitute_features
from src.compression.multisource import (compact_real_block,merge_quaternion_multi,merge_real,pair_maps,quaternion_maps,
    real_maps,real_pair_fits,select_disjoint,substitute_multi)
from src.compression.quaternion_channels import compact_block
from src.data.cifar_cached import file_sha256,load_cached_cifar
from src.evaluation.block_ablation import capture_blocks,classification_metrics,predict_from_block

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT / ".cache/matplotlib"))
SOURCES=["scripts/multisource_compensation.py","src/compression/multisource.py"]
EXACT_FOLDS={"quaternion_exact","multi_q","multi_q_random","pair_merge","multi_real","multi_real_random"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_grid(features,quaternion):
    """(N,C,H,W) post-block tensor -> fitting layout (N,positions,units,4) or (N,positions,channels)."""
    n,channels,height,width=features.shape
    if quaternion:
        return features.reshape(n,4,channels//4,height,width).permute(0,3,4,2,1).reshape(n,height*width,channels//4,4).double().numpy()
    return features.permute(0,2,3,1).reshape(n,height*width,channels).double().numpy()


def quaternion_cases(model,fit,pairs,removed,seed,study):
    victims=[row["victim"] for row in pairs];alpha=study["ridge_alpha"]
    full=quaternion_maps(fit,victims,"full_real",alpha)
    cases=[("baseline","baseline",None,deepcopy(model).eval(),None),
           ("paired_pruned","paired_pruned",None,compact_block(model,1,sorted(set(range(16))-set(victims))).eval(),None),
           ("quaternion_exact","quaternion_exact",None,merge_internal(model,1,pairs,"quaternion_exact")[0],pairs)]
    maps=quaternion_maps(fit,victims,"quaternion_right",alpha)
    cases.append(("multi_q","multi_q",None,merge_quaternion_multi(model,1,maps)[0],maps))
    projected,_,projection=merge_quaternion_multi(model,1,full)
    cases.append(("multi_full_projected","multi_full_projected",None,projected,full))
    for rep in range(study["random_repetitions"]):
        random=np.random.default_rng(study["quaternion_random_seed_offset"]+seed*100+rep).permutation(16)[:removed].tolist()
        cases.append((f"random_r{rep:02d}","random",rep,compact_block(model,1,sorted(set(range(16))-set(random))).eval(),None))
        random_maps=quaternion_maps(fit,random,"quaternion_right",alpha)
        cases.append((f"multi_q_random_r{rep:02d}","multi_q_random",rep,merge_quaternion_multi(model,1,random_maps)[0],random_maps))
    return cases,full,projection


def real_cases(model,fit,pairs,removed,seed,study):
    victims=[row["victim"] for row in pairs];alpha=study["ridge_alpha"]
    cases=[("baseline","baseline",None,deepcopy(model).eval(),None),
           ("paired_pruned","paired_pruned",None,compact_real_block(model,1,sorted(set(range(32))-set(victims))),None)]
    for name,maps in (("pair_merge",pair_maps(pairs,32)),("multi_real",real_maps(fit,victims,alpha))):
        cases.append((name,name,None,merge_real(model,1,maps,32)[0],maps))
    for rep in range(study["random_repetitions"]):
        random=np.random.default_rng(study["real_random_seed_offset"]+seed*100+rep).permutation(32)[:removed].tolist()
        cases.append((f"random_r{rep:02d}","random",rep,compact_real_block(model,1,sorted(set(range(32))-set(random))),None))
        random_maps=real_maps(fit,random,alpha)
        cases.append((f"multi_real_random_r{rep:02d}","multi_real_random",rep,merge_real(model,1,random_maps,32)[0],random_maps))
    return cases


def aggregate(rows,seeds):
    groups={}
    for row in rows:
        groups.setdefault((row["architecture"],row["removed"],row["policy"]),[]).append(row)
    result=[]
    for (architecture,removed,policy),items in groups.items():
        values={}
        for split in ("validation","check_calibration"):
            per_seed=[]
            for seed in seeds:
                chosen=[row["evaluation"][split] for row in items if row["seed"]==seed]
                per_seed.append({"seed":seed,"accuracy":float(np.mean([m["accuracy"] for m in chosen])),"cross_entropy":float(np.mean([m["cross_entropy"] for m in chosen]))})
            accuracies=[row["accuracy"] for row in per_seed]
            values[split]={"seed_values":per_seed,"mean_accuracy":float(np.mean(accuracies)),"sample_sd_accuracy":float(np.std(accuracies,ddof=1)),
                           "mean_cross_entropy":float(np.mean([row["cross_entropy"] for row in per_seed]))}
        result.append({"architecture":architecture,"removed":removed,"policy":policy,"real_parameters":items[0]["real_parameters"],**values})
    return result


COMPARISONS={"quaternion":[("multi_q","quaternion_exact"),("multi_q","multi_full_projected"),("multi_full_substitution","multi_q"),
                           ("multi_q","random"),("multi_q_random","random"),("multi_q","multi_q_random"),("quaternion_exact","random")],
             "real":[("multi_real","pair_merge"),("multi_real","random"),("multi_real_random","random"),("multi_real","multi_real_random"),("pair_merge","random")]}


def paired_effects(aggregates):
    lookup={(row["architecture"],row["removed"],row["policy"]):row for row in aggregates}
    effects=[]
    for architecture,removed in sorted({(row["architecture"],row["removed"]) for row in aggregates}):
        family="real" if architecture=="real_cnn6" else "quaternion"
        for left,right in COMPARISONS[family]:
            a,b=lookup[(architecture,removed,left)],lookup[(architecture,removed,right)]
            differences=[100*(x["accuracy"]-y["accuracy"]) for x,y in zip(a["validation"]["seed_values"],b["validation"]["seed_values"])]
            check=[100*(x["accuracy"]-y["accuracy"]) for x,y in zip(a["check_calibration"]["seed_values"],b["check_calibration"]["seed_values"])]
            ce=[x["cross_entropy"]-y["cross_entropy"] for x,y in zip(a["validation"]["seed_values"],b["validation"]["seed_values"])]
            effects.append({"architecture":architecture,"removed":removed,"left":left,"right":right,"validation_seed_pp":differences,
                            "validation_mean_pp":float(np.mean(differences)),"check_seed_pp":check,"check_mean_pp":float(np.mean(check)),
                            "validation_seed_ce":ce,"positive_seeds":int(sum(value>0 for value in differences))})
    return effects


def render(output,summary):
    aggregates,effects=summary["aggregates"],summary["effects"]
    order=["baseline","random","paired_pruned","quaternion_exact","pair_merge","multi_q_random","multi_real_random","multi_full_projected","multi_q","multi_real","multi_full_substitution"]
    rows=["# Multi-source compensation with and without the Hamilton restriction","",
          "Status: completed pilot. No fine-tuning, BN refit or official-test inference. Configuration: "
          "`configs/multisource_compensation.json`. Unconstrained multi-source ridge reconstruction folded into the "
          "next layer is prior art (GRAIL2026; also Asiaee2026CMR, Kim2020). Under study is only the right-Hamilton "
          "restriction q_v ~= sum_k q_k*a_k, which folds exactly as w_k <- w_k + a_k*w_v and keeps a quaternion consumer.","",
          "## Protocol","",
          "Block 2 output, three training seeds per architecture. Fit on 500 even calibration images, check on 500 odd ones; "
          "validation is the primary held-out split. Ridge alpha fixed a priori at "
          f"{summary['protocol']['config']['ridge_alpha']} (lambda = alpha * mean Gram diagonal). Quaternion selected victims are the frozen nested "
          "pairwise victims from results/internal_replication, so multi_q and quaternion_exact remove identical units. "
          "Random rows: ten victim sets per seed, averaged within seed before the seed SD. multi_*_random applies the "
          "multi-source fit to those same random victim sets. multi_full_substitution is a functional oracle (unconstrained "
          "GRAIL-style substitution); folding it would break the Hamilton consumer, so it has no compact-quaternion budget. "
          "Real CNN6 control: 8/32 and 16/32 block-2 channels, victims by the scalar analogue of the pairwise rule.","",
          "## Validation results","","| Architecture | Removed | Policy | Real parameters | Validation mean ± seed SD | Mean CE | Check mean |",
          "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for item in sorted(aggregates,key=lambda row:(row["architecture"],row["removed"],order.index(row["policy"]))):
        params=str(item["real_parameters"]) if item["real_parameters"] is not None else "functional oracle"
        rows.append(f"| {item['architecture']} | {item['removed']} | {item['policy']} | {params} | {100*item['validation']['mean_accuracy']:.2f}% ± "
                    f"{100*item['validation']['sample_sd_accuracy']:.2f} pp | {item['validation']['mean_cross_entropy']:.4f} | {100*item['check_calibration']['mean_accuracy']:.2f}% |")
    rows+=["","## Paired effects (left minus right), all seeds retained","",
           "| Architecture | Removed | Comparison | Validation per seed (pp) | Mean (pp) | Check mean (pp) | Positive seeds |","| --- | ---: | --- | --- | ---: | ---: | ---: |"]
    for item in effects:
        rows.append(f"| {item['architecture']} | {item['removed']} | {item['left']} - {item['right']} | {', '.join(f'{v:+.2f}' for v in item['validation_seed_pp'])} | "
                    f"{item['validation_mean_pp']:+.2f} | {item['check_mean_pp']:+.2f} | {item['positive_seeds']}/3 |")
    rows+=["","## Structural diagnostics","","| Architecture | Removed | Seed | Discarded energy of unconstrained additions after projection |","| --- | ---: | ---: | ---: |"]
    for item in summary["diagnostics"]:
        rows.append(f"| {item['architecture']} | {item['removed']} | {item['seed']} | {100*item['discarded_energy_fraction']:.2f}% |")
    rows+=["","## Matched-savings comparison across families (CNN6 only)","",
           "Quaternion 4/16 and real 8/32 removals save almost the same number of real parameters (likewise 8/16 vs 16/32). "
           "The families start from different baselines, widths and compute, so only drops from each family's own baseline are compared. "
           "This is descriptive, not a causal claim about quaternion algebra.","",
           "| Removal pair | Policy pair | Quaternion drop (pp) | Real drop (pp) |","| --- | --- | ---: | ---: |"]
    lookup={(row["architecture"],row["removed"],row["policy"]):row for row in aggregates}
    for q_removed,r_removed in ((4,8),(8,16)):
        for q_policy,r_policy in (("random","random"),("paired_pruned","paired_pruned"),("quaternion_exact","pair_merge"),("multi_q","multi_real"),("multi_q_random","multi_real_random")):
            q=100*(lookup[("cnn6",q_removed,q_policy)]["validation"]["mean_accuracy"]-lookup[("cnn6",q_removed,"baseline")]["validation"]["mean_accuracy"])
            r=100*(lookup[("real_cnn6",r_removed,r_policy)]["validation"]["mean_accuracy"]-lookup[("real_cnn6",r_removed,"baseline")]["validation"]["mean_accuracy"])
            rows.append(f"| {q_removed}/16 vs {r_removed}/32 ({lookup[('cnn6',q_removed,q_policy)]['real_parameters']} vs {lookup[('real_cnn6',r_removed,r_policy)]['real_parameters']} params) | {q_policy} vs {r_policy} | {q:+.2f} | {r:+.2f} |")
    rows+=["","## Verification","",f"Exact-fold equivalence checks (physical compact model vs functional substitution, check half): "
           f"{summary['equivalence']['count']} cases, maximum logit error {summary['equivalence']['maximum']:.2e} (tolerance {summary['protocol']['config']['equivalence_tolerance']}). "
           "Every quaternion compact model keeps QuaternionConv2d consumers, so the Hamilton structure is retained by construction.","",
           "## Limits","","One block, two removal levels, small sequential CNNs on CIFAR-10, three seeds. Validation also selected baseline checkpoints. "
           "Ridge alpha was not tuned. No fine-tuning; a separate equal-budget fine-tuning study is required before practical claims. "
           "Weighted projection of the unconstrained ridge solution is not run here. No novelty or general superiority claim.",""]
    (output / "REPORT.md").write_text("\n".join(rows),encoding="utf-8")


@torch.no_grad()
def run(study):
    output=ROOT / study["output"]
    if study["official_test_evaluation"] or study["compression_optimizer_updates"] or study["bn_recalibration"]:
        raise ValueError("Only the no-test, no-fine-tuning study is implemented")
    rep_config=read(ROOT / study["replication_config"])
    selected=read(ROOT / study["replication_output"] / "selection_frozen.json")["pairs"]
    hashes={name:file_sha256(ROOT / name) for name in SOURCES+["src/compression/internal_merging.py","src/compression/quaternion_channels.py"]}
    protocol={"config":study,"source_hashes":hashes,"replication_selection_sha256":file_sha256(ROOT / study["replication_output"] / "selection_frozen.json"),"official_test_evaluated":False}
    if (output / "protocol.json").exists() and read(output / "protocol.json")!=protocol:
        raise ValueError("Protocol/sources changed; preserve the run and use another output")
    write_json(output / "protocol.json",protocol)
    if (output / "summary.json").exists():
        summary=read(output / "summary.json");render(output,summary);return summary
    torch.set_num_threads(study["cpu_threads"]);seed_everything(study["seeds"][0]);device=torch.device(study["device"])
    (images,labels),unused_test=load_cached_cifar(ROOT / study["archive"]);del unused_test
    stats=read(ROOT / study["data_source"] / "data.json")["statistics"]
    mean=torch.tensor(stats["mean"],dtype=torch.float32,device=device).reshape(1,3,1,1)
    std=torch.tensor(stats["std"],dtype=torch.float32,device=device).reshape(1,3,1,1)
    sources=[{**source,"family":"quaternion"} for source in sources_for(rep_config)]
    for row in read(ROOT / study["real_sources"] / "summary.json")["runs"]:
        path=str(row["path"]).replace("\\","/")
        sources.append({"architecture":"real_cnn6","family":"real","seed":row["seed"],"path":path,"config":read(ROOT / path / "summary.json")["config"]})
    built=[];diagnostics=[];frozen={}
    for source in sources:
        architecture,seed,quaternion=source["architecture"],source["seed"],source["family"]=="quaternion"
        model=(make_quaternion(source) if quaternion else make_cnn6(source["config"],"real")).to(device).eval()
        model.load_state_dict(torch.load(ROOT / source["path"] / ("quaternion" if quaternion else "real") / "model.pt",map_location=device,weights_only=True)["model"])
        with np.load(ROOT / source["path"] / "split_indices.npz") as split: cal,val=split["calibration"],split["validation"]
        features=capture_blocks(model,prepare_batch(torch.as_tensor(images[cal],device=device),mean,std),study["batch_size"])[1]
        grid=to_grid(features,quaternion)
        if quaternion:
            with np.load(ROOT / source["cache_path"]) as cached: reference=cached["trained"]
        elif seed==2026:
            with np.load(ROOT / "results/block_analysis/real/block_2/calibration_activations.npz") as cached: reference=cached["trained"]
        else:
            reference=None
        if reference is not None and float(np.max(np.abs(grid-reference)))>study["equivalence_tolerance"]: raise AssertionError(f"Feature cache mismatch: {architecture}/{seed}")
        fit=grid[::2]
        if quaternion:
            levels=study["quaternion_removed_units"]
        else:
            levels=study["real_removed_channels"]
            ordered=select_disjoint(real_pair_fits(fit,model.blocks[2][0].weight.detach().cpu().numpy()),max(levels))
        for removed in levels:
            key=f"{architecture}:{seed}:{removed}"
            if quaternion:
                pairs=selected[key];cases,full,projection=quaternion_cases(model,fit,pairs,removed,seed,study)
                diagnostics.append({"architecture":architecture,"seed":seed,"removed":removed,**projection});frozen[key]={"victims":[row["victim"] for row in pairs],"multi_full":full}
            else:
                pairs=ordered[:removed];cases=real_cases(model,fit,pairs,removed,seed,study)
                frozen[key]={"pairs":pairs}
            for name,policy,rep,compact,maps in cases:
                built.append({"architecture":architecture,"seed":seed,"removed":removed,"case":name,"policy":policy,"random_repetition":rep,
                              "model":compact,"maps":maps,"quaternion":quaternion,"source_model":model,"cal":cal,"val":val,
                              "full":full if quaternion else None})
            print(f"{key}: {len(cases)} compact cases frozen",flush=True)
    write_json(output / "selection_frozen.json",{"selections":frozen,"maps":[{key:value for key,value in item.items() if key in ("architecture","seed","removed","case","maps")} for item in built if item["maps"] is not None],
                                                "all_choices_frozen_before_validation":True})
    rows=[];errors=[];feature_cache={}
    for item in built:
        source_key=(item["architecture"],item["seed"])
        if source_key not in feature_cache:
            original=item["source_model"]
            normalized={"check_calibration":prepare_batch(torch.as_tensor(images[item["cal"][1::2]],device=device),mean,std),
                        "validation":prepare_batch(torch.as_tensor(images[item["val"]],device=device),mean,std)}
            feature_cache={source_key:{split:capture_blocks(original,value,study["batch_size"])[1] for split,value in normalized.items()}}
        ids={"check_calibration":item["cal"][1::2],"validation":item["val"]}
        metrics={split:evaluate(item["model"],torch.as_tensor(images[split_ids],device=device),torch.as_tensor(labels[split_ids],device=device),mean,std,study["batch_size"])[0] for split,split_ids in ids.items()}
        if item["policy"] in EXACT_FOLDS:
            check=feature_cache[source_key]["check_calibration"].to(device)
            if item["policy"]=="quaternion_exact":
                substituted=substitute_features(check,item["maps"],"quaternion_right")
            else:
                substituted=substitute_multi(check,item["maps"],item["quaternion"])
            expected=predict_from_block(item["source_model"],1,substituted,[],study["batch_size"])
            physical=item["model"](prepare_batch(torch.as_tensor(images[ids["check_calibration"]],device=device),mean,std)).cpu().numpy()
            error=float(np.max(np.abs(physical-expected)));errors.append(error)
            if error>study["equivalence_tolerance"]: raise AssertionError(f"Fold check failed: {item['architecture']}/{item['seed']}/{item['removed']}/{item['case']}: {error}")
        rows.append({key:item[key] for key in ("architecture","seed","removed","case","policy","random_repetition")}|
                    {"real_parameters":sum(p.numel() for p in item["model"].parameters()),"evaluation":metrics,"physical":True})
        if item["case"]=="baseline" and item["quaternion"]:
            metrics={}
            for split,split_ids in ids.items():
                logits=predict_from_block(item["source_model"],1,substitute_multi(feature_cache[source_key][split].to(device),item["full"],True),[],study["batch_size"])
                metrics[split]=classification_metrics(logits,labels[split_ids])
            rows.append({"architecture":item["architecture"],"seed":item["seed"],"removed":item["removed"],"case":"multi_full_substitution",
                         "policy":"multi_full_substitution","random_repetition":None,"real_parameters":None,"evaluation":metrics,"physical":False})
    aggregates=aggregate(rows,study["seeds"]);effects=paired_effects(aggregates)
    summary={"status":"completed","protocol":protocol,"rows":rows,"aggregates":aggregates,"effects":effects,"diagnostics":diagnostics,
             "equivalence":{"count":len(errors),"maximum":max(errors)},"compression_optimizer_updates":0,"official_test_evaluated":False}
    write_json(output / "summary.json",summary);render(output,summary)
    return summary


if __name__=="__main__":
    run(read(ROOT / "configs/multisource_compensation.json"))
