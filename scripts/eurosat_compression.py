"""EuroSAT replication of multi-source compensation (phase A) and equal-budget fine-tuning (phase B)."""

from copy import deepcopy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from scripts.cifar_cnn import write_json
from scripts.eurosat_cnn import load_eurosat
from scripts.finetune_compression import paired_effects,structure
from scripts.multisource_compensation import EXACT_FOLDS,aggregate,quaternion_cases,real_cases,to_grid
from src.cnn_training import evaluate,prepare_batch,seed_everything
from src.compression.internal_merging import fit_pairs,select_pairs,substitute_features
from src.compression.multisource import real_pair_fits,select_disjoint,substitute_multi
from src.data.cifar_cached import file_sha256
from src.evaluation.block_ablation import capture_blocks,classification_metrics,predict_from_block
from src.finetuning import finetune
from src.models.eurosat_cnn import make_fourband

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT / ".cache/matplotlib"))
SOURCES=["scripts/eurosat_compression.py","scripts/eurosat_cnn.py","src/models/eurosat_cnn.py","scripts/multisource_compensation.py",
         "scripts/finetune_compression.py","src/compression/multisource.py","src/finetuning.py","src/compression/internal_merging.py"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(rows,aggregates,effects,title):
    order=["baseline","random","paired_pruned","quaternion_exact","pair_merge","multi_q_random","multi_real_random","multi_full_projected","multi_q","multi_real","multi_full_substitution"]
    lines=[f"## {title}","","| Model | Removed | Policy | Real parameters | Validation mean ± seed SD | Mean CE | Check mean |","| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for item in sorted(aggregates,key=lambda row:(row["architecture"],row["removed"],order.index(row["policy"]))):
        params=str(item["real_parameters"]) if item["real_parameters"] is not None else "functional oracle"
        lines.append(f"| {'quaternion' if item['architecture']=='cnn6' else 'real'} | {item['removed']} | {item['policy']} | {params} | {100*item['validation']['mean_accuracy']:.2f}% ± "
                     f"{100*item['validation']['sample_sd_accuracy']:.2f} pp | {item['validation']['mean_cross_entropy']:.4f} | {100*item['check_calibration']['mean_accuracy']:.2f}% |")
    lines+=["","| Model | Removed | Comparison | Validation per seed (pp) | Mean (pp) | Check mean (pp) | Positive seeds |","| --- | ---: | --- | --- | ---: | ---: | ---: |"]
    for item in effects:
        lines.append(f"| {'quaternion' if item['architecture']=='cnn6' else 'real'} | {item['removed']} | {item['left']} - {item['right']} | {', '.join(f'{v:+.2f}' for v in item['validation_seed_pp'])} | "
                     f"{item['validation_mean_pp']:+.2f} | {item['check_mean_pp']:+.2f} | {item['positive_seeds']}/3 |")
    return lines+[""]


def render(output,summary):
    lines=["# EuroSAT replication: multi-source compensation and equal-budget fine-tuning","",
           "Second dataset: EuroSAT_MS (Zenodo 7711810), four bands q = NIR + R i + G j + B k (NIR in the real part is a choice). "
           "Quaternion and real CNN6 baselines were trained with the unchanged CIFAR trainer and budget (results/eurosat_cnn). "
           "All compression and fine-tuning rules were copied unchanged from the CIFAR studies and fixed before any EuroSAT compression "
           "outcome. Official test split (5,400 images) held out and not evaluated. Validation also selected baseline checkpoints. "
           "Unconstrained multi-source folding is prior art (GRAIL2026); only the Hamilton restriction is under study. No novelty claim.",""]
    lines+=table(summary["phase_a"]["rows"],summary["phase_a"]["aggregates"],summary["phase_a"]["effects"],"Phase A: compression without fine-tuning")
    lines+=[f"Exact-fold checks: {summary['phase_a']['equivalence']['count']} cases, maximum logit error {summary['phase_a']['equivalence']['maximum']:.2e}.",""]
    if summary.get("phase_b"):
        lines+=table(summary["phase_b"]["rows"],summary["phase_b"]["aggregates"],summary["phase_b"]["effects"],
                     f"Phase B: after {summary['protocol']['config']['finetune']['epochs']} fine-tuning epochs (final epoch, no selection; first {summary['protocol']['config']['finetune']['random_repetitions_finetuned']} random sets)")
        lines+=[f"{summary['phase_b']['run_count']} fine-tuning runs; parameter and quaternion-layer counts unchanged in every run.",""]
    lines+=["## Limits","","One block, two removal levels, one small CNN family, three seeds, fixed fine-tuning schedule. Descriptive cross-family comparisons only.",""]
    (output / "REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def run(study,phases=("a","b")):
    output=ROOT / study["output"]
    if study["official_test_evaluation"]: raise ValueError("Official-test evaluation is not part of this study")
    training=read(ROOT / study["training_config"]);training_output=ROOT / training["output"]
    protocol={"config":study,"source_hashes":{name:file_sha256(ROOT / name) for name in SOURCES},
              "baselines_summary_sha256":file_sha256(training_output / "summary.json"),"official_test_evaluated":False}
    if (output / "protocol.json").exists() and read(output / "protocol.json")!=protocol:
        raise ValueError("Protocol/sources changed; preserve completed runs and use another output")
    write_json(output / "protocol.json",protocol)
    torch.set_num_threads(study["cpu_threads"]);device=torch.device(study["device"])
    images,targets,indices,meta=load_eurosat(ROOT / training["data"])
    mean=torch.tensor(meta["statistics"]["mean"],dtype=torch.float32,device=device).reshape(1,4,1,1)
    std=torch.tensor(meta["statistics"]["std"],dtype=torch.float32,device=device).reshape(1,4,1,1)
    ids={"train":indices["train"],"validation":indices["validation"],"check_calibration":indices["calibration"][1::2]}
    tensors={key:torch.as_tensor(images[value],device=device) for key,value in ids.items()}
    labels={key:torch.as_tensor(targets[value],device=device) for key,value in ids.items()}
    calibration=prepare_batch(torch.as_tensor(images[indices["calibration"]],device=device),mean,std)
    phase_a=[];errors=[];phase_b=[];frozen={};count=0
    for run_row in read(training_output / "summary.json")["runs"]:
        seed,path=run_row["seed"],ROOT / run_row["path"]
        config=read(path / "summary.json")["config"]
        for kind in ("quaternion","real"):
            architecture,quaternion="cnn6" if kind=="quaternion" else "real_cnn6",kind=="quaternion"
            seed_everything(seed)
            model=make_fourband(config,kind).to(device).eval()
            model.load_state_dict(torch.load(path / kind / "model.pt",map_location=device,weights_only=True)["model"])
            features=capture_blocks(model,calibration,study["batch_size"])[1]
            fit=to_grid(features,quaternion)[::2]
            check={split:capture_blocks(model,prepare_batch(tensors[split],mean,std),study["batch_size"])[1] for split in ("check_calibration","validation")}
            levels=study["quaternion_removed_units"] if quaternion else study["real_removed_channels"]
            if quaternion:
                ordered=select_pairs(fit_pairs(fit,model.blocks[2][0].kernel.detach().cpu().double().numpy()),max(levels))
            else:
                ordered=select_disjoint(real_pair_fits(fit,model.blocks[2][0].weight.detach().cpu().numpy()),max(levels))
            for removed in levels:
                pairs=ordered[:removed]
                if quaternion: cases,full,_=quaternion_cases(model,fit,pairs,removed,seed,study)
                else: cases=real_cases(model,fit,pairs,removed,seed,study)
                frozen[f"{architecture}:{seed}:{removed}"]={"pairs":pairs}
                for name,policy,rep,compact,maps in cases:
                    base={"architecture":architecture,"seed":seed,"removed":removed,"case":name,"policy":policy,"random_repetition":rep}
                    if "a" in phases:
                        metrics={split:evaluate(compact,tensors[split],labels[split],mean,std,study["batch_size"])[0] for split in ("validation","check_calibration")}
                        if policy in EXACT_FOLDS:
                            values=check["check_calibration"].to(device)
                            substituted=substitute_features(values,maps,"quaternion_right") if policy=="quaternion_exact" else substitute_multi(values,maps,quaternion)
                            expected=predict_from_block(model,1,substituted,[],study["batch_size"])
                            physical=compact(prepare_batch(tensors["check_calibration"],mean,std)).detach().cpu().numpy()
                            error=float(np.max(np.abs(physical-expected)));errors.append(error)
                            if error>study["equivalence_tolerance"]: raise AssertionError(f"Fold check failed: {base}: {error}")
                        phase_a.append({**base,"real_parameters":structure(compact)["real_parameters"],"evaluation":metrics})
                        if policy=="baseline" and quaternion:
                            oracle={split:classification_metrics(predict_from_block(model,1,substitute_multi(check[split].to(device),full,True),[],study["batch_size"]),labels[split].cpu().numpy())
                                    for split in ("validation","check_calibration")}
                            phase_a.append({**base,"case":"multi_full_substitution","policy":"multi_full_substitution","real_parameters":None,"evaluation":oracle})
                    tuning=study["finetune"]
                    if "b" in phases and (removed==levels[0] if policy=="baseline" else rep is None or rep<tuning["random_repetitions_finetuned"]):
                        stored=0 if policy=="baseline" else removed
                        folder=output / "finetuned" / architecture / f"seed_{seed}" / f"removed_{stored:02d}" / name
                        if not (folder / "metrics.json").exists():
                            start=time.perf_counter();before=structure(compact)
                            tuned,history=finetune(deepcopy(compact),tensors,labels,mean,std,tuning,seed,("validation","check_calibration"))
                            if structure(tuned)!=before: raise AssertionError(f"Fine-tuning changed the architecture: {folder}")
                            folder.mkdir(parents=True,exist_ok=True)
                            torch.save({"model":{key:value.detach().cpu() for key,value in tuned.state_dict().items()}},folder / "model.pt")
                            write_json(folder / "metrics.json",{**base,"removed":stored,**before,"history":history,
                                                               "final":{split:history[-1][split] for split in ("validation","check_calibration")},"official_test_evaluated":False})
                            print(f"finetuned {architecture}/{seed}/{stored}/{name}: val {100*history[-1]['validation']['accuracy']:.2f}% ({time.perf_counter()-start:.0f} s)",flush=True)
                        result=read(folder / "metrics.json");count+=1
                        for level in (levels if policy=="baseline" else [removed]):
                            phase_b.append({**base,"removed":level,"real_parameters":result["real_parameters"],"evaluation":result["final"]})
                print(f"{architecture} seed {seed}: {removed} removed done",flush=True)
    write_json(output / "selection_frozen.json",{"selections":frozen})
    previous=read(output / "summary.json") if (output / "summary.json").exists() else {}
    summary={"status":"completed","protocol":protocol,"official_test_evaluated":False}
    if "a" in phases:
        aggregates=aggregate(phase_a,study["seeds"])
        summary["phase_a"]={"rows":phase_a,"aggregates":aggregates,"effects":paired_effects(aggregates),"equivalence":{"count":len(errors),"maximum":max(errors)}}
    elif previous.get("phase_a"):
        summary["phase_a"]=previous["phase_a"]
    if "b" in phases:
        aggregates=aggregate(phase_b,study["seeds"])
        summary["phase_b"]={"rows":phase_b,"aggregates":aggregates,"effects":paired_effects(aggregates),"run_count":count}
    write_json(output / "summary.json",summary);render(output,summary)
    return summary


if __name__=="__main__":
    run(read(ROOT / "configs/eurosat_compression.json"))
