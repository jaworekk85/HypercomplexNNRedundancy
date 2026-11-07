"""Quality-size curve: uniform compression of every convolution output, with and without compensation,
before and after equal-budget fine-tuning, against the same compact architectures trained from scratch.

Phase A builds and evaluates every compact model without training (minutes) and stores the cases to
be fine-tuned. Phase B fine-tunes them (src.finetuning, one shared batch stream per training seed).
Phase C trains the compact architectures from scratch with the unchanged baseline trainer and budget.
Every phase is resumable; REPORT.md/summary.json are rewritten after each phase. Everything is fixed in
the config before outcomes. No official-test inference. Usage: python -m scripts.compression_curve [config]
"""

import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

import scripts.cifar_cnn as trainer
from scripts.cifar_cnn import make_model as make_cnn6,write_json
from scripts.eurosat_cnn import load_eurosat
from scripts.finetune_compression import sources_of,structure
from src.cnn_training import evaluate,prepare_batch,seed_everything
from src.compression.multilayer import (SiteMoments,compress_site,consumer_energy,greedy_victims,kept_width,magnitude_victims,
    site_gram,site_maps,sites,substituted_logits,units)
from src.data.cifar_cached import file_sha256,load_cached_cifar
from src.finetuning import finetune
from src.models.cifar_fourblock import make_fourblock
from src.models.eurosat_cnn import make_fourband

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT / ".cache/matplotlib"))
SOURCES=["scripts/compression_curve.py","src/compression/multilayer.py","src/finetuning.py","src/compression/multisource.py",
         "src/compression/internal_merging.py","src/compression/bn_magnitude.py","src/compression/quaternion_channels.py"]
ORDER=["baseline","random","bn_pruned","greedy_pruned","random_comp","greedy_full_projected","greedy_comp","scratch"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def factory(dataset,architecture):
    if dataset=="eurosat": return make_fourband
    return make_fourblock if architecture=="cnn8" else make_cnn6


def kind_of(architecture):
    return "real" if architecture.startswith("real") else "quaternion"


def reduced_config(config,architecture,fraction):
    key="real_widths" if kind_of(architecture)=="real" else "quaternion_widths"
    return {**config,key:[kept_width(width,fraction) for width in config[key]]}


def load_data(study,device):
    """Images, labels, normalization and baseline sources (with config and checkpoint folder)."""
    if study["dataset"]=="cifar":
        ms=read(ROOT / "configs/multisource_compensation.json")
        (images,labels),unused_test=load_cached_cifar(ROOT / ms["archive"]);del unused_test
        stats=read(ROOT / ms["data_source"] / "data.json")["statistics"]
        sources=sources_of(ms)
    elif study["dataset"]=="eurosat":
        images,labels,_,meta=load_eurosat(ROOT / study["data"]);stats=meta["statistics"]
        sources=[{"architecture":architecture,"seed":row["seed"],"path":row["path"],"config":read(ROOT / row["path"] / "summary.json")["config"]}
                 for row in read(ROOT / study["training_output"] / "summary.json")["runs"] for architecture in ("cnn6","real_cnn6")]
    else:
        raise ValueError("Unknown dataset")
    sources=[source for source in sources if source["architecture"] in study["architectures"] and source["seed"] in study["seeds"]]
    channels=len(stats["mean"])
    mean=torch.tensor(stats["mean"],dtype=torch.float32,device=device).reshape(1,channels,1,1)
    std=torch.tensor(stats["std"],dtype=torch.float32,device=device).reshape(1,channels,1,1)
    return images,labels,mean,std,sources


def split_ids(source,limits):
    with np.load(ROOT / source["path"] / "split_indices.npz") as split:
        ids={"train":split["train"],"validation":split["validation"],"fit":split["calibration"][::2],"check_calibration":split["calibration"][1::2],
             "calibration":split["calibration"]}
    # Smoke limits take evenly spaced ids: the EuroSAT split files are sorted by class, so a prefix would be one class.
    return {key:value[np.linspace(0,len(value)-1,limits[key]).round().astype(int)] if limits and key in limits else value for key,value in ids.items()}


def load_source(study,source,device):
    architecture=source["architecture"]
    model=factory(study["dataset"],architecture)(source["config"],kind_of(architecture)).to(device).eval()
    model.load_state_dict(torch.load(ROOT / source["path"] / kind_of(architecture) / "model.pt",map_location=device,weights_only=True)["model"])
    return model


def random_victims(study,seed,fraction_index,rep,site_index,width,count):
    rng=np.random.default_rng(study["random_seed_offset"]+seed*1000+fraction_index*100+rep*10+site_index)
    return sorted(rng.permutation(width)[:count].tolist())


def chain(model,fraction,choose,mode,study,fit,check):
    """Compress every site shallow to deep. choose(index,site,current,moments,count) -> victims; mode None prunes only.
    Exact folds are checked at every site against the functional substitution reference."""
    current,records,alpha=model,[],study["ridge_alpha"]
    for index,site in enumerate(sites(model)):
        width=units(current,site);count=width-kept_width(width,fraction)
        moments=SiteMoments(*site_gram(current,site,fit,study["batch_size"]),current.kind=="quaternion") if mode or choose.needs_moments else None
        victims=choose(index,site,current,moments,count)
        maps=site_maps(moments,victims,alpha,mode) if mode else None
        compact,kept,lost=compress_site(current,site,victims,maps)
        record={"site":list(site),"victims":victims,"kept":len(kept),"projection_discarded_energy_fraction":lost}
        if mode and mode!="full_real":
            with torch.no_grad():
                record["fold_error"]=float((compact(check)-substituted_logits(current,site,maps,check,study["batch_size"])).abs().max())
            if record["fold_error"]>study["equivalence_tolerance"]: raise AssertionError(f"Fold check failed at site {site}: {record['fold_error']}")
        records.append(record);current=compact
    return current,records


def chooser(function,needs_moments=False):
    function.needs_moments=needs_moments
    return function


def build_cases(model,study,seed,fraction_index,fraction,fit,check):
    quaternion=model.kind=="quaternion";mode="quaternion_right" if quaternion else "real";alpha=study["ridge_alpha"]
    greedy=chooser(lambda index,site,current,moments,count:greedy_victims(moments,count,consumer_energy(current,site),alpha,mode),True)
    compact,records=chain(model,fraction,greedy,mode,study,fit,check)
    frozen=[row["victims"] for row in records]
    same=chooser(lambda index,site,current,moments,count:frozen[index])
    cases=[("greedy_comp","greedy_comp",None,compact,records),("greedy_pruned","greedy_pruned",None,*chain(model,fraction,same,None,study,fit,check))]
    if quaternion:
        cases.append(("greedy_full_projected","greedy_full_projected",None,*chain(model,fraction,same,"full_real",study,fit,check)))
    cases.append(("bn_pruned","bn_pruned",None,*chain(model,fraction,chooser(lambda index,site,current,moments,count:magnitude_victims(current,site,count)),None,study,fit,check)))
    for rep in range(study["random_repetitions"]):
        random=chooser(lambda index,site,current,moments,count,rep=rep:random_victims(study,seed,fraction_index,rep,index,units(current,site),count))
        cases.append((f"random_r{rep:02d}","random",rep,*chain(model,fraction,random,None,study,fit,check)))
        cases.append((f"random_comp_r{rep:02d}","random_comp",rep,*chain(model,fraction,random,mode,study,fit,check)))
    return cases


def finetuned(study,policy,rep):
    config=study["finetune"]
    return policy in config["policies"] and (rep is None or rep<config["random_repetitions_finetuned"])


def metrics(model,images,labels,ids,mean,std,study,device):
    return {split:evaluate(model,torch.as_tensor(images[ids[split]],device=device),torch.as_tensor(labels[ids[split]],device=device),mean,std,study["batch_size"])[0]
            for split in ("validation","check_calibration")}


def phase_a(study,source,images,labels,mean,std,device,output):
    architecture,seed=source["architecture"],source["seed"]
    path=output / "phase_a" / f"{architecture}_seed_{seed}.json"
    if path.exists(): return read(path)
    seed_everything(seed)
    model=load_source(study,source,device);ids=split_ids(source,study.get("limits"))
    fit=prepare_batch(torch.as_tensor(images[ids["fit"]],device=device),mean,std)
    check=prepare_batch(torch.as_tensor(images[ids["check_calibration"][:study["equivalence_images"]]],device=device),mean,std)
    make=factory(study["dataset"],architecture)
    rows=[{"architecture":architecture,"seed":seed,"fraction":0.,"case":"baseline","policy":"baseline","random_repetition":None,"stage":"before",
           **structure(model),"evaluation":metrics(model,images,labels,ids,mean,std,study,device)}]
    store=output / "compact" / architecture / f"seed_{seed}";(store / "f_000").mkdir(parents=True,exist_ok=True)
    torch.save({"model":{key:value.detach().cpu() for key,value in model.state_dict().items()}},store / "f_000" / "baseline.pt")
    records=[]
    for fraction_index,fraction in enumerate(study["fractions"]):
        start=time.perf_counter();folder=store / f"f_{round(1000*fraction):03d}";folder.mkdir(parents=True,exist_ok=True)
        narrow=reduced_config(source["config"],architecture,fraction)
        for name,policy,rep,compact,site_records in build_cases(model,study,seed,fraction_index,fraction,fit,check):
            reloaded=make(narrow,kind_of(architecture)).to(device).eval()
            reloaded.load_state_dict(compact.state_dict())
            with torch.no_grad():
                if float((reloaded(check)-compact(check)).abs().max())>1e-6: raise AssertionError("Compact model is not the narrower standard architecture")
            rows.append({"architecture":architecture,"seed":seed,"fraction":fraction,"case":name,"policy":policy,"random_repetition":rep,"stage":"before",
                         **structure(compact),"evaluation":metrics(compact,images,labels,ids,mean,std,study,device)})
            records.append({"architecture":architecture,"seed":seed,"fraction":fraction,"case":name,"sites":site_records})
            if finetuned(study,policy,rep):
                torch.save({"model":{key:value.detach().cpu() for key,value in compact.state_dict().items()}},folder / f"{name}.pt")
        print(f"phase A {architecture}/{seed} fraction {fraction}: {time.perf_counter()-start:.0f} s, "+
              ", ".join(f"{row['case']} {100*row['evaluation']['validation']['accuracy']:.1f}%" for row in rows if row["fraction"]==fraction and row["random_repetition"] in (None,0)),flush=True)
    result={"rows":rows,"records":records}
    write_json(path,result)
    return result


def phase_b(study,source,images,labels,mean,std,device,output):
    architecture,seed=source["architecture"],source["seed"];ids=split_ids(source,study.get("limits"))
    tensors={name:torch.as_tensor(images[ids[name]],device=device) for name in ("train","validation","check_calibration")}
    targets={name:torch.as_tensor(labels[ids[name]],device=device) for name in tensors}
    make,rows=factory(study["dataset"],architecture),[]
    for fraction in [0.]+study["fractions"]:
        folder=output / "compact" / architecture / f"seed_{seed}" / f"f_{round(1000*fraction):03d}"
        config=reduced_config(source["config"],architecture,fraction) if fraction else source["config"]
        for checkpoint in sorted(folder.glob("*.pt")):
            name=checkpoint.stem;target=output / "finetuned" / architecture / f"seed_{seed}" / folder.name / name / "metrics.json"
            policy=name.rsplit("_r",1)[0] if name.startswith("random") else name
            if not target.exists():
                start=time.perf_counter()
                model=make(config,kind_of(architecture)).to(device).eval()
                model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)["model"])
                before=structure(model)
                tuned,history=finetune(model,tensors,targets,mean,std,study["finetune"],seed,("validation","check_calibration"))
                if structure(tuned)!=before: raise AssertionError(f"Fine-tuning changed the architecture: {checkpoint}")
                target.parent.mkdir(parents=True,exist_ok=True)
                torch.save({"model":{key:value.detach().cpu() for key,value in tuned.state_dict().items()}},target.parent / "model.pt")
                write_json(target,{"architecture":architecture,"seed":seed,"fraction":fraction,"case":name,"policy":policy,**before,"history":history,
                                   "final":{split:history[-1][split] for split in ("validation","check_calibration")},"official_test_evaluated":False})
                print(f"finetuned {architecture}/{seed}/{fraction}/{name}: val {100*history[-1]['validation']['accuracy']:.2f}% ({time.perf_counter()-start:.0f} s)",flush=True)
            result=read(target)
            rows.append({"architecture":architecture,"seed":seed,"fraction":fraction,"case":name,"policy":policy,
                         "random_repetition":int(name.rsplit("_r",1)[1]) if name.startswith("random") else None,"stage":"after",
                         "real_parameters":result["real_parameters"],"quaternion_layers":result["quaternion_layers"],"evaluation":result["final"]})
    return rows


def phase_c(study,source,images,labels,mean,std,device,output):
    architecture,seed=source["architecture"],source["seed"];ids=split_ids(source,study.get("limits"))
    tensors={name:torch.as_tensor(images[ids[name]],device=device) for name in ("train","validation","calibration","check_calibration")}
    targets={name:torch.as_tensor(labels[ids[name]],device=device) for name in tensors}
    trainer.make_model=factory(study["dataset"],architecture);rows=[]
    for fraction in study["scratch"]["fractions"]:
        config=reduced_config(source["config"],architecture,fraction)
        if study["scratch"]["epochs"]: config={**config,"epochs":study["scratch"]["epochs"]}
        folder=output / "scratch" / architecture / f"seed_{seed}" / f"f_{round(1000*fraction):03d}"
        report=trainer.train_one(config,kind_of(architecture),tensors,targets,mean,std,folder / kind_of(architecture),True)
        model=trainer.make_model(config,kind_of(architecture)).to(device).eval()
        model.load_state_dict(torch.load(folder / kind_of(architecture) / "model.pt",map_location=device,weights_only=True)["model"])
        with (folder / kind_of(architecture) / "training.csv").open(encoding="utf-8") as handle:
            final_epoch=float(handle.read().strip().splitlines()[-1].split(",")[4])
        rows.append({"architecture":architecture,"seed":seed,"fraction":fraction,"case":"scratch","policy":"scratch","random_repetition":None,"stage":"scratch",
                     **structure(model),"evaluation":metrics(model,images,labels,ids,mean,std,study,device),
                     "best_epoch":report["best_epoch"],"final_epoch_validation_accuracy":final_epoch})
        print(f"scratch {architecture}/{seed}/{fraction}: selected val {100*rows[-1]['evaluation']['validation']['accuracy']:.2f}%, final epoch {100*final_epoch:.2f}%",flush=True)
    trainer.make_model=make_cnn6
    return rows


def aggregate(rows,seeds):
    groups={}
    for row in rows:
        groups.setdefault((row["architecture"],row["fraction"],row["policy"],row["stage"]),[]).append(row)
    result=[]
    for (architecture,fraction,policy,stage),items in sorted(groups.items()):
        entry={"architecture":architecture,"fraction":fraction,"policy":policy,"stage":stage,"real_parameters":items[0]["real_parameters"]}
        for split in ("validation","check_calibration"):
            per_seed=[{"seed":seed,"accuracy":float(np.mean([row["evaluation"][split]["accuracy"] for row in items if row["seed"]==seed]))}
                      for seed in seeds if any(row["seed"]==seed for row in items)]
            values=[row["accuracy"] for row in per_seed]
            entry[split]={"seed_values":per_seed,"mean_accuracy":float(np.mean(values)),"sample_sd_accuracy":float(np.std(values,ddof=1)) if len(values)>1 else 0.}
        result.append(entry)
    return result


def paired_effects(aggregates,comparisons):
    lookup={(row["architecture"],row["fraction"],row["policy"],row["stage"]):row for row in aggregates}
    effects=[]
    for architecture,fraction in sorted({(row["architecture"],row["fraction"]) for row in aggregates if row["fraction"]>0}):
        for left,right in comparisons:
            a,b=lookup.get((architecture,fraction,*left.split("@"))),lookup.get((architecture,fraction,*right.split("@")))
            if a is None or b is None: continue
            x,y=({value["seed"]:value["accuracy"] for value in item["validation"]["seed_values"]} for item in (a,b))
            common=sorted(set(x)&set(y));differences=[100*(x[seed]-y[seed]) for seed in common]
            effects.append({"architecture":architecture,"fraction":fraction,"left":left,"right":right,"seeds":common,"validation_seed_pp":differences,
                            "validation_mean_pp":float(np.mean(differences)),"positive_seeds":int(sum(value>0 for value in differences))})
    return effects


def plot(output,aggregates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup={(row["architecture"],row["fraction"],row["policy"],row["stage"]):row for row in aggregates}
    colors={"random":"tab:gray","bn_pruned":"tab:orange","random_comp":"tab:cyan","greedy_comp":"tab:blue","scratch":"tab:green"}
    figure,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
    for axis,stage,title in zip(axes,("before","after"),("Without fine-tuning","After fine-tuning (scratch: trained from scratch)")):
        for architecture in sorted({row["architecture"] for row in aggregates}):
            for policy,color in colors.items():
                if policy=="scratch" and stage=="before": continue
                wanted="scratch" if policy=="scratch" else stage
                points=[(lookup[key]["real_parameters"],100*lookup[key]["validation"]["mean_accuracy"]) for key in sorted(lookup)
                        if key[0]==architecture and key[2]==policy and key[3]==wanted]
                base=lookup.get((architecture,0.,"baseline","before" if policy=="scratch" else stage))
                if base: points.append((base["real_parameters"],100*base["validation"]["mean_accuracy"]))
                if len(points)>1:
                    points.sort()
                    axis.plot(*zip(*points),marker="o",color=color,linestyle="--" if kind_of(architecture)=="real" else "-",label=f"{architecture} {policy}")
        axis.set_title(title);axis.set_xlabel("real parameters (log scale)");axis.set_xscale("log");axis.grid(alpha=.3)
    axes[0].set_ylabel("validation accuracy (%)")
    for axis in axes:
        if axis.get_legend_handles_labels()[0]: axis.legend(fontsize=7,ncol=2)
    figure.tight_layout();figure.savefig(output / "quality_size_curve.png",dpi=150);plt.close(figure)


def render(output,summary):
    config=summary["protocol"]["config"];aggregates=summary["aggregates"]
    lookup={(row["architecture"],row["fraction"],row["policy"],row["stage"]):row for row in aggregates}
    cell=lambda row:f"{100*row['validation']['mean_accuracy']:.2f}% ± {100*row['validation']['sample_sd_accuracy']:.2f} (n={len(row['validation']['seed_values'])})" if row else "—"
    lines=["# Quality-size curve: uniform multi-site compression","",
           f"Status: {summary['status']}. Dataset {config['dataset']}; architectures {', '.join(config['architectures'])}; seeds {config['seeds']}. "
           f"The same fraction {config['fractions']} of units is removed at every convolution output, shallow to deep, so each compact model is the "
           "standard architecture with narrower widths (checked by reloading it into that architecture). Compensation folds a ridge reconstruction "
           f"(alpha {config['ridge_alpha']}) from the kept units of the same site into the consumer: right-Hamilton maps for quaternion models, scalar "
           "maps for real models. Unconstrained multi-source folding is prior art (GRAIL2026); only the Hamilton restriction is under study. "
           f"Fine-tuning: {config['finetune']['epochs']} epochs, final epoch, no selection. Scratch: same compact architecture, baseline trainer "
           "and budget, checkpoint selected by validation cross-entropy as for the baselines. No official-test inference. No novelty claim.","",
           "## Validation accuracy (mean ± seed SD)","","| Architecture | Removed fraction | Policy | Real parameters | Without fine-tuning | After fine-tuning / from scratch |",
           "| --- | ---: | --- | ---: | ---: | ---: |"]
    for architecture,fraction,policy in sorted({key[:3] for key in lookup},key=lambda key:(key[0],key[1],ORDER.index(key[2]))):
        before=lookup.get((architecture,fraction,policy,"before"))
        after=lookup.get((architecture,fraction,policy,"scratch" if policy=="scratch" else "after"))
        params=(before or after)["real_parameters"]
        lines.append(f"| {architecture} | {fraction} | {policy} | {params} | {cell(before)} | {cell(after)} |")
    lines+=["","## Paired effects (left minus right, validation pp, all seeds retained)","",
            "| Architecture | Removed fraction | Comparison | Per seed | Mean | Positive seeds |","| --- | ---: | --- | --- | ---: | ---: |"]
    for item in summary["effects"]:
        lines.append(f"| {item['architecture']} | {item['fraction']} | {item['left']} - {item['right']} | {', '.join(f'{v:+.2f}' for v in item['validation_seed_pp'])} | "
                     f"{item['validation_mean_pp']:+.2f} | {item['positive_seeds']}/{len(item['seeds'])} |")
    pairs=[(q,r) for q,r in (("cnn6","real_cnn6"),) if q in config["architectures"] and r in config["architectures"]]
    if pairs:
        lines+=["","## Quaternion vs real at the same removed fraction (near-equal real parameters)","",
                "Descriptive: the families start from different baselines. Drop = accuracy minus the family's own baseline of the same stage "
                "(scratch is compared with the original baseline).","",
                "| Fraction | Policy and stage | Quaternion params | Real params | Quaternion acc | Real acc | Quaternion drop | Real drop |","| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for q,r in pairs:
            for fraction in config["fractions"]:
                for policy,stage in (("random","before"),("greedy_comp","before"),("random","after"),("bn_pruned","after"),("greedy_comp","after"),("scratch","scratch")):
                    a,b=lookup.get((q,fraction,policy,stage)),lookup.get((r,fraction,policy,stage))
                    base_stage="before" if stage=="scratch" else stage
                    qa,ra=lookup.get((q,0.,"baseline",base_stage)),lookup.get((r,0.,"baseline",base_stage))
                    if not (a and b and qa and ra): continue
                    acc=lambda row:100*row["validation"]["mean_accuracy"]
                    lines.append(f"| {fraction} | {policy}@{stage} | {a['real_parameters']} | {b['real_parameters']} | {acc(a):.2f}% | {acc(b):.2f}% | {acc(a)-acc(qa):+.2f} | {acc(b)-acc(ra):+.2f} |")
    diagnostics=summary["diagnostics"]
    lines+=["","## Verification and diagnostics","",
            f"Exact-fold checks (physical compact model vs functional substitution, per site, {config['equivalence_images']} check images): "
            f"{diagnostics['fold_checks']} checks, maximum logit error {diagnostics['max_fold_error']:.2e} (tolerance {config['equivalence_tolerance']}). "
            "Every compact model was reloaded into the standard narrower architecture with identical logits.",""]
    if diagnostics["projection"]:
        lines+=["Energy of the unconstrained (greedy_full_projected) additions discarded by the Hamilton projection, mean over seeds, per site:","",
                "| Architecture | Fraction | Per site (shallow to deep) |","| --- | ---: | --- |"]
        for item in diagnostics["projection"]:
            lines.append(f"| {item['architecture']} | {item['fraction']} | {', '.join(f'{100*v:.1f}%' for v in item['per_site'])} |")
    lines+=["","Figure: `quality_size_curve.png` (solid quaternion, dashed real).","","## Limits","",
            "Small sequential CNNs, uniform fractions, one fixed ridge alpha and fine-tuning schedule, few seeds. Validation also selected the "
            "baseline and scratch checkpoints. Random rows average their repetitions within seed first.",""]
    (output / "REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def summarize(study,protocol,output,rows,records,status):
    aggregates=aggregate(rows,study["seeds"])
    errors=[site["fold_error"] for record in records for site in record["sites"] if "fold_error" in site]
    projection=[]
    for architecture,fraction in sorted({(r["architecture"],r["fraction"]) for r in records if r["case"]=="greedy_full_projected"}):
        chosen=[r["sites"] for r in records if r["architecture"]==architecture and r["fraction"]==fraction and r["case"]=="greedy_full_projected"]
        projection.append({"architecture":architecture,"fraction":fraction,"per_site":[float(np.mean([s[i]["projection_discarded_energy_fraction"] for s in chosen])) for i in range(len(chosen[0]))]})
    summary={"status":status,"protocol":protocol,"rows":rows,"aggregates":aggregates,"effects":paired_effects(aggregates,study["comparisons"]),
             "diagnostics":{"fold_checks":len(errors),"max_fold_error":max(errors) if errors else 0.,"projection":projection},"official_test_evaluated":False}
    write_json(output / "summary.json",summary);render(output,summary);plot(output,aggregates)


def run(study):
    output=ROOT / study["output"]
    if study["official_test_evaluation"]: raise ValueError("Official-test evaluation is not part of this study")
    protocol={"config":study,"source_hashes":{name:file_sha256(ROOT / name) for name in SOURCES},"official_test_evaluated":False}
    if (output / "protocol.json").exists() and read(output / "protocol.json")!=protocol:
        raise ValueError("Protocol/sources changed; preserve completed runs and use another output")
    write_json(output / "protocol.json",protocol)
    torch.set_num_threads(study["cpu_threads"]);device=torch.device(study["device"])
    images,labels,mean,std,sources=load_data(study,device)
    rows,records=[],[]
    for source in sources:
        result=phase_a(study,source,images,labels,mean,std,device,output);rows+=result["rows"];records+=result["records"]
    summarize(study,protocol,output,rows,records,"phase A done (no fine-tuning yet)")
    if study["finetune"]["enabled"]:
        for source in sources: rows+=phase_b(study,source,images,labels,mean,std,device,output)
        summarize(study,protocol,output,rows,records,"phases A-B done (scratch training pending)")
    if study["scratch"]["enabled"]:
        for source in sources: rows+=phase_c(study,source,images,labels,mean,std,device,output)
    summarize(study,protocol,output,rows,records,"completed")


if __name__=="__main__":
    run(read(ROOT / (sys.argv[1] if len(sys.argv)>1 else "configs/compression_curve.json")))
