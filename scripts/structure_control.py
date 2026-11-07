"""One established non-quaternion structure control at exactly matched widths.

Reuses frozen training, ranking and fine-tuning; source modules stay unchanged.
"""

from copy import deepcopy
import importlib.metadata as metadata
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from scripts import cifar_cnn as trainer
from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from src.cnn_training import evaluate,prepare_batch,seed_everything
from src.compression import multilayer as old
from src.compression.structured_blocks import StructuredMoments,basis_of,compact,energy,magnitude,reference
from src.data.cifar_cached import file_sha256,load_cached_cifar
from src.models.circulant_cnn import CirculantConv2d,make_circulant

ROOT=Path(__file__).resolve().parents[1]
CONFIG="configs/structure_control.json"
SOURCES=list(dict.fromkeys(["scripts/structure_control.py","src/models/circulant_cnn.py","src/compression/structured_blocks.py",*trainer.SOURCES,*curve.SOURCES]))
read=curve.read


def input_paths(study):
    names=[f"{study['parent_curve']}/summary.json","configs/compression_curve.json"]
    for seed in study["seeds"]:
        source=study["quaternion_sources"][str(seed)]
        names.extend(f"{source}/{n}" for n in ("summary.json","data.json","split_indices.npz","quaternion/model.pt","quaternion/metrics.json"))
        names.append(f"{study['parent_curve']}/phase_a/cnn6_seed_{seed}.json")
        names.append(read(ROOT / source / "summary.json")["config"]["archive"])
    return sorted(set(names))


def first_update(model,inputs,labels,optimizer):
    """Audit ACTUAL tracked circulant coefficients, not an empty quaternion check."""
    model.train();before={n:p.detach().clone() for n,p in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True);outputs=model(inputs);loss=F.cross_entropy(outputs,labels);loss.backward()
    rows=[];layers={}
    for name,p in model.named_parameters():
        if p.grad is None or not bool(torch.isfinite(p.grad).all()): raise AssertionError(f"Missing/nonfinite circulant gradient: {name}")
        rows.append({"name":name,"gradient_norm":float(p.grad.norm()),"shape":list(p.shape)})
    for name,layer in model.named_modules():
        if isinstance(layer,CirculantConv2d): layers[name]={"component_gradient_norms":layer.kernel.grad.flatten(1).norm(dim=1).tolist()}
    optimizer.step()
    for row,(name,p) in zip(rows,model.named_parameters()): row["max_abs_update"]=float((p.detach()-before[name]).abs().max())
    for name,layer in model.named_modules():
        if isinstance(layer,CirculantConv2d): layers[name]["component_update_norms"]=(layer.kernel.detach()-before[f"{name}.kernel"]).flatten(1).norm(dim=1).tolist()
    checks={"six_structured_layers":len(layers)==6,"all_variables_updated":all(r["max_abs_update"]>0 for r in rows),
            "all_circulant_components_have_gradients_and_updates":all(v>0 for row in layers.values() for values in row.values() for v in values)}
    if not all(checks.values()): raise AssertionError(f"Circulant first-update audit failed: {checks}")
    return {"parameters":rows,"structured_layers":layers,"checks":checks,"loss_before":float(loss.detach()),"correct_before":int((outputs.argmax(1)==labels).sum())}


@torch.no_grad()
def collect(model,images,mean,std,batch_size=128):
    model.eval();out=[]
    for start in range(0,len(images),batch_size):
        values=model.forward_features(prepare_batch(images[start:start+batch_size],mean,std));n,c,h,w=values.shape
        out.append(values.reshape(n,4,c//4,h,w).permute(0,3,4,2,1).reshape(n,h*w,c//4,4).cpu().numpy())
    return np.concatenate(out)


def train(study,seed,tensors,labels,mean,std,folder,hashes):
    config={**read(ROOT / study["baseline_source"] / "summary.json")["config"],"seed":seed,"models":["circulant"],
            "device":study["device"],"epochs":study["training_epochs"],"structure":"circulant4"}
    with patch.object(trainer,"make_model",make_circulant),patch.object(trainer,"source_hashes",lambda:hashes),patch.object(trainer,"first_update_audit",first_update),patch.object(trainer,"collect",collect):
        try: result=trainer.train_one(config,"circulant",tensors,labels,mean,std,folder / "circulant",True)
        except UnboundLocalError as error:
            # Narrow adapter for the frozen trainer's completed-resume cleanup bug.
            if "saved" not in str(error) or not (folder / "circulant/metrics.json").exists(): raise
            result=read(folder / "circulant/metrics.json")
            last=torch.load(folder / "circulant/last.pt",map_location="cpu",weights_only=True)
            selected=torch.load(folder / "circulant/model.pt",map_location="cpu",weights_only=True)
            if last["epoch"]!=config["epochs"] or result["source_hashes"]!=hashes or any(not torch.equal(v,last["best_state"][k]) for k,v in selected["model"].items()): raise
            write_json(folder / "circulant/resume_cleanup_compatibility.json",{"handled_exception":str(error),"states_verified":True})
    if not all(result["first_update"]["checks"].values()) or result["test_evaluated"] or result["real_trainable_parameters"]!=73898:
        raise AssertionError("Circulant training gates failed")
    last=torch.load(folder / "circulant/last.pt",map_location="cpu",weights_only=True)
    if last["epoch"]!=config["epochs"] or len(last["history"])!=config["epochs"]: raise AssertionError("Baseline training incomplete")
    selected=torch.load(folder / "circulant/model.pt",map_location="cpu",weights_only=True)
    if any(not torch.equal(v,last["best_state"][k]) for k,v in selected["model"].items()): raise AssertionError("Best-state export mismatch")
    write_json(folder / "summary.json",{"config":config,"models":[result],"official_test_evaluated":False})
    return {"architecture":"circulant_cnn6","seed":seed,"path":str(folder.relative_to(ROOT)).replace("\\","/"),"config":config}


def chain(model,fraction,study,fit,check,frozen=None,bn=False):
    current=model;records=[]
    for index,site in enumerate(old.sites(model)):
        width=old.units(current,site);count=round(fraction*width)
        moments=None;maps=None
        if frozen is not None: victims=frozen[index]
        elif bn: victims=magnitude(current,site,count)
        else:
            moments=StructuredMoments(*old.site_gram(current,site,fit,study["batch_size"]),basis_of(current))
            victims=old.greedy_victims(moments,count,energy(current,site),study["ridge_alpha"],"structured")
            maps=old.site_maps(moments,victims,study["ridge_alpha"],"structured")
        narrowed,lost=compact(current,site,victims,maps)
        row={"site":list(site),"victims":victims,"kept":width-count,"projection_discarded_energy_fraction":lost}
        if maps:
            with torch.no_grad(): row["fold_error"]=float((narrowed(check)-reference(current,site,maps,check)).abs().max())
            if row["fold_error"]>study["equivalence_tolerance"]: raise AssertionError("Circulant fold mismatch")
        records.append(row);current=narrowed
    return current,records


def cases(model,study,seed,index,fraction,fit,check):
    compensated,records=chain(model,fraction,study,fit,check);victims=[r["victims"] for r in records]
    return [("greedy_comp","greedy_comp",None,compensated,records),
            ("greedy_pruned","greedy_pruned",None,*chain(model,fraction,study,fit,check,frozen=victims)),
            ("bn_pruned","bn_pruned",None,*chain(model,fraction,study,fit,check,bn=True))]


def add_quaternion_pruned(study,source,images,labels,mean,std,device,output):
    """Missing paired 3-epoch control; reconstruct ONLY the frozen parent victims."""
    seed=source["seed"];phase=read(ROOT / study["parent_curve"] / "phase_a" / f"cnn6_seed_{seed}.json")
    for fraction in study["fractions"]:
        target=output / "compact/cnn6" / f"seed_{seed}/f_{round(1000*fraction):03d}/greedy_pruned.pt"
        if target.exists(): continue
        model=curve.load_source({"dataset":"cifar"},source,device)
        record=next(r for r in phase["records"] if r["fraction"]==fraction and r["case"]=="greedy_pruned")
        for row in record["sites"]: model=compact(model,row["site"],row["victims"])[0]
        ids=curve.split_ids(source,study.get("limits"));actual=curve.metrics(model,images,labels,ids,mean,std,study,device)
        expected=next(r for r in phase["rows"] if r["fraction"]==fraction and r["case"]=="greedy_pruned")
        if not study.get("smoke_check"):
            if any(actual[s]["accuracy"]!=expected["evaluation"][s]["accuracy"] or abs(actual[s]["cross_entropy"]-expected["evaluation"][s]["cross_entropy"])>1e-4 for s in actual):
                raise AssertionError("Reconstructed Q pruning control differs from parent")
        target.parent.mkdir(parents=True,exist_ok=True);torch.save({"model":{k:v.detach().cpu() for k,v in model.state_dict().items()}},target)
    return curve.phase_b(study,source,images,labels,mean,std,device,output)


def summarize(study,protocol,rows,records,status,output):
    groups=curve.aggregate(rows,study["seeds"]);effects=curve.paired_effects(groups,[["greedy_comp@before","greedy_pruned@before"],["greedy_comp@after","greedy_pruned@after"],
                                                                                   ["greedy_comp@before","bn_pruned@before"],["greedy_comp@after","bn_pruned@after"]])
    drops=[]
    for row in rows:
        baseline=next((r for r in rows if r["architecture"]==row["architecture"] and r["seed"]==row["seed"] and r["fraction"]==0 and r["stage"]==row["stage"]),None)
        if baseline:
            drops.append({"architecture":row["architecture"],"seed":row["seed"],"fraction":row["fraction"],"policy":row["policy"],"stage":row["stage"],
                          "loss_from_own_baseline_pp":100*(baseline["evaluation"]["validation"]["accuracy"]-row["evaluation"]["validation"]["accuracy"])})
    gaps=[]
    for seed in study["seeds"]:
        q=next((r for r in rows if (r["architecture"],r["seed"],r["fraction"],r["stage"])==("cnn6",seed,0,"before")),None)
        c=next((r for r in rows if (r["architecture"],r["seed"],r["fraction"],r["stage"])==("circulant_cnn6",seed,0,"before")),None)
        if q and c: gaps.append({"seed":seed,"circulant_minus_quaternion_pp":100*(c["evaluation"]["validation"]["accuracy"]-q["evaluation"]["validation"]["accuracy"])})
    errors=[s["fold_error"] for r in records for s in r["sites"] if "fold_error" in s]
    summary={"status":status,"protocol":protocol,"rows":rows,"aggregates":groups,"paired_policy_effects":effects,"own_baseline_losses":drops,
        "baseline_quality_gaps":gaps,"diagnostics":{"circulant_fold_checks":len(errors),"max_fold_error":max(errors,default=0.)},"official_test_evaluated":False}
    write_json(output / "summary.json",summary)
    lines=["# QNN compression and one matched structured control","","## Hypotheses","",
        "Primary: characterize structure-preserving compression of an existing QNN family, including quality/size, repair and failure regimes. Secondary: does one established non-quaternion sharing pattern show comparable behavior? No new algorithm, algebra or quaternion superiority is claimed.","",
        "Circulant 4x4 blocks (Ding2017CirCNN context) have four trainable coefficients, like Hamilton blocks. Widths, input information (0,R,G,B), coefficient initialization, BN/ReLU/dropout and parameter counts match exactly. Circulant groups are not quaternion units. Both structured spaces permit exact linear folding; our expanded Conv2d does not implement FFT acceleration or reproduce CirCNN. Selected after Q/real outcomes; exploratory follow-up.","",
        "CIFAR CNN6 only, 3 seeds, 50-epoch baselines selected by validation CE; f=.25/.5 at all six sites. Identical 3-epoch repair and same-victim controls. Reuse original Q/real outcomes and add the previously missing Q same-victim pruning repair. No new official-test inference, latency claim, Circulant scratch or EuroSAT sweep.","",
        "## Measured results","",f"State: {status}. Scientific interpretation requires completed scientific runs; CPU smoke values are not results.","",
        "| Family | Fraction removed | Policy | Stage | Actual real params | Validation mean | Seed SD (pp) |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |"]
    for r in groups:
        lines.append(f"| {r['architecture']} | {r['fraction']} | {r['policy']} | {r['stage']} | {r['real_parameters']} | {100*r['validation']['mean_accuracy']:.2f}% | {100*r['validation']['sample_sd_accuracy']:.2f} |")
    lines.extend(["","## Interpretation limits","",
        "Report absolute accuracy/CE, own-baseline losses and every paired seed, including unfavorable outcomes. Baseline quality may differ despite identical parameter budgets; own-baseline normalization does not eliminate this confound. This single contrast cannot isolate one quaternion property or prove a generic structure-only explanation. No hyperparameter retuning or dropping a weak control after inspecting results.","",
        "The original paper evidence includes Q-vs-real quality-size curves on CIFAR/EuroSAT and matched compact scratch controls. Those 50-epoch scratch runs have a different student-training budget from 3-epoch repairs. Unrestricted functional substitution is not a compact quaternion architecture. Fit-then-project versus directly constrained fitting is a different contrast from unrestricted substitution.","",
        "Full per-seed rows, own-baseline drops, baseline gaps, paired compensation effects and exact-fold diagnostics are in summary.json. A negative result is retained. Pseudo-label repair remains a separate supplementary question."])
    (output / "REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(study,config_name=CONFIG):
    torch.set_num_threads(study["cpu_threads"]);device=torch.device(study["device"])
    output=ROOT / study["output"];output.mkdir(parents=True,exist_ok=True)
    hashes={n:file_sha256(ROOT / n) for n in sorted(set([*SOURCES,config_name,*input_paths(study)]))}
    protocol={"config":study,"hashes":hashes,"environment":{"torch":metadata.version("torch"),"numpy":metadata.version("numpy"),"python":sys.version,
        "device":str(device),"gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None},"official_test_evaluated":False}
    if (output / "protocol.json").exists() and read(output / "protocol.json")!=protocol: raise ValueError("Structure control sources/inputs changed")
    write_json(output / "protocol.json",protocol)
    original=read(ROOT / study["baseline_source"] / "summary.json")["config"]
    (images,labels),unused=load_cached_cifar(ROOT / original["archive"]);del unused
    stats=read(ROOT / study["baseline_source"] / "data.json")["statistics"]
    mean=torch.tensor(stats["mean"],dtype=torch.float32,device=device).reshape(1,3,1,1);std=torch.tensor(stats["std"],dtype=torch.float32,device=device).reshape(1,3,1,1)
    sources=[]
    for seed in study["seeds"]:
        source={"path":study["quaternion_sources"][str(seed)]}
        ids=curve.split_ids(source,study.get("limits"));tensors={k:torch.as_tensor(images[ids[k]],device=device) for k in ("train","validation","calibration")}
        targets={k:torch.as_tensor(labels[ids[k]],device=device) for k in tensors}
        folder=output / f"baselines/seed_{seed}";folder.mkdir(parents=True,exist_ok=True)
        for name in ("data.json","split_indices.npz"):
            if not (folder / name).exists(): shutil.copyfile(ROOT / source["path"] / name,folder / name)
        sources.append(train(study,seed,tensors,targets,mean,std,folder,hashes))
        del tensors,targets
    rows=[];records=[]
    # These patches are scoped to this process; immutable original files remain intact.
    with patch.object(curve,"factory",lambda dataset,arch:make_circulant),patch.object(curve,"kind_of",lambda arch:"circulant"),patch.object(curve,"build_cases",cases):
        for source in sources:
            result=curve.phase_a(study,source,images,labels,mean,std,device,output / "circulant")
            rows+=result["rows"];records+=result["records"]
            summarize(study,protocol,rows,records,"circulant compression",output)
        for source in sources:
            rows+=curve.phase_b(study,source,images,labels,mean,std,device,output / "circulant")
            summarize(study,protocol,rows,records,"circulant repair",output)
    parent=read(ROOT / study["parent_curve"] / "summary.json")
    if parent["status"]!="completed": raise ValueError("Original reference curve incomplete")
    rows.extend(r for r in parent["rows"] if r["architecture"] in ("cnn6","real_cnn6") and r["seed"] in study["seeds"] and
                r["fraction"] in [0.,*study["fractions"]] and r["policy"] in ["baseline",*study["policies"]] and r["stage"] in ("before","after"))
    for seed in study["seeds"]:
        path=study["quaternion_sources"][str(seed)];source={"architecture":"cnn6","seed":seed,"path":path,"config":read(ROOT / path / "summary.json")["config"]}
        rows+=add_quaternion_pruned(study,source,images,labels,mean,std,device,output / "quaternion_paired_repair")
        summarize(study,protocol,rows,records,"paired quaternion repair",output)
    if len(records)!=len(study["seeds"])*len(study["fractions"])*3: raise AssertionError("Incomplete circulant grid")
    if any(file_sha256(ROOT / n)!=h for n,h in hashes.items()): raise ValueError("Structure-control inputs changed during run")
    summarize(study,protocol,rows,records,"completed",output)


if __name__=="__main__":
    config_name=sys.argv[1] if len(sys.argv)>1 else CONFIG;run(read(ROOT / config_name),config_name)
