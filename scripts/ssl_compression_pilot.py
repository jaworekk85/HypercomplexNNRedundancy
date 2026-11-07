"""Narrow bridge: genuine low-label SSL -> compression -> paired continuation.

No fully supervised initialization, new loss, official-test inference or
automatic email/title decision. Sibling sources and old experiments stay frozen.
"""

from copy import deepcopy
from datetime import datetime,timezone
import importlib.metadata as metadata
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.cifar_cnn import make_model,write_json
from scripts.compression_curve import read,reduced_config
from scripts.structure_control import chain
from src.cnn_training import evaluate,prepare_batch,seed_everything
from src.data.cifar_cached import file_sha256,load_cached_cifar
from src.models.circulant_cnn import make_circulant
from src.ssl_bridge import partition,sibling_api,train

ROOT=Path(__file__).resolve().parents[1]
CONFIG="configs/ssl_compression_pilot.json"
SOURCES=["src/ssl_bridge.py","scripts/ssl_compression_pilot.py","src/models/cifar_cnn_torch.py","src/models/circulant_cnn.py",
    "src/models/quaternion_conv_torch.py","src/cnn_training.py","src/data/cifar_cached.py","scripts/structure_control.py",
    "src/compression/structured_blocks.py","src/compression/multilayer.py","src/compression/multisource.py",
    "src/compression/internal_merging.py","src/compression/quaternion_channels.py","src/compression/quaternion_merging.py",
    "src/compression/bn_magnitude.py","scripts/cifar_cnn.py","scripts/compression_curve.py"]


def make(config,family):
    return make_model(config,"quaternion") if family=="quaternion" else make_circulant(config)


def source_names(study):
    sibling=ROOT / study["sibling_root"]
    names=[*SOURCES,study["model_config"],study["archive"],f"{study['sibling_root']}/{study['source_ssl_config']}"]
    names.extend(f"{study['sibling_root']}/fewshot_metric/{p.name}" for p in (sibling / "fewshot_metric").glob("*.py"))
    return sorted(set(names))


def ssl_config(study,base,api,seed,method,repair=False):
    config={**base,"seed":seed,"epochs":study["baseline_epochs"],"steps_per_epoch":study["steps_per_epoch"],
        "budget_per_class":study["labels_per_class"],"validation_per_class":study["validation_per_class"],
        "batch_size":study["batch_size"],"unlabeled_batch_size":study["batch_size"],"selection":"last"}
    config=api.select_method(config,"ce_2v_pseudo_soft" if method=="soft_ssl" else "ce_2v_u_control")
    if repair: config["learning_rate"]=study["repair_learning_rate"]
    api.validate_pilot(config)
    return config


@torch.no_grad()
def logits_on(model,normalized,batch_size=128):
    model.eval()
    return torch.cat([model(normalized[start:start+batch_size]).float().cpu() for start in range(0,len(normalized),batch_size)])


def pseudo_diagnostic(logits,truth,temperature,state,parent_logits):
    p=logits.softmax(1);confidence,prediction=p.max(1);accepted=confidence>=.95
    correct=prediction==torch.as_tensor(truth);soft=(logits/temperature).softmax(1)
    classes=torch.bincount(prediction,minlength=10).tolist()
    result={"examples":len(truth),"argmax_accuracy":float(correct.float().mean()),"coverage_at_095":float(accepted.float().mean()),
        "precision_at_095":float(correct[accepted].float().mean()) if bool(accepted.any()) else None,
        "mean_confidence":float(confidence.mean()),"predicted_class_counts":classes}
    if state is not None:
        weights=torch.exp(-(confidence-state["mean"]).clamp(max=0).square()/(2*state["std"]**2)) if state["ready"] else accepted.float()
        result.update(confidence_center=state["mean"],confidence_std=state["std"],mean_soft_weight=float(weights.mean()),
            weighted_argmax_accuracy=float((weights*correct).sum()/weights.sum()) if float(weights.sum())>0 else None)
    if parent_logits is not None:
        reference=(parent_logits/temperature).softmax(1)
        result.update(agreement_with_parent=float((prediction==parent_logits.argmax(1)).float().mean()),
            parent_to_current_soft_kl=float((reference*(reference.clamp_min(1e-12).log()-soft.clamp_min(1e-12).log())).sum(1).mean()))
    return result


def summarize(study,protocol,baselines,repairs,status,output):
    effects=[]
    for family in study["families"]:
        for method in study["repair_methods"]:
            selected={r["initialization"]:r for r in repairs if r["family"]==family and r["method"]==method}
            if "greedy_comp" in selected and "greedy_pruned" in selected:
                a,b=selected["greedy_comp"],selected["greedy_pruned"]
                effects.append({"family":family,"method":method,"comp_minus_same_victim_pruned_pp":100*(a["diagnostics"][-1]["validation"]["accuracy"]-b["diagnostics"][-1]["validation"]["accuracy"])})
    summary={"status":status,"protocol":protocol,"baselines":baselines,"repairs":repairs,"paired_effects_single_seed":effects,
        "official_test_evaluated":False,"smoke_check":study.get("smoke_check",False),"updated_utc":datetime.now(timezone.utc).isoformat()}
    write_json(output / "summary.json",summary)
    lines=["# Low-label SSL and structure-preserving compression: bounded pilot","","## Hypotheses and registered scope","",
        "Does compensation preserve the pseudo-label signal after compression, and improve subsequent SSL recovery? Do QNN and one matched circulant control show the same pattern? This is a SINGLE-SEED exploratory pilot, not replicated evidence, a new algorithm or a full FixMatch/SoftMatch reproduction (Sohn2020FixMatch, Chen2023SoftMatch, Ding2017CirCNN).", "",
        "Fresh models: only 50 training labels/class from the FIRST update, source sibling soft pseudo-targets/epoch-lagged confidence weights, weak/strong views, joint train-mode BN. No supervised teacher/pretraining. Two baselines, 20x383 SGD steps, last checkpoint; no tuning or official-test inference. Input information/width/parameter budgets match; the encoder differs from the sibling's original 3-convolution/embedding CNN, so old test scores are not directly comparable.","",
        "ADAPTATION: reserve 500 fit and 500 check images from the previously unlabeled pool, never weight training. At the full scope: L=500, U=47,500, validation=1000, fit=500, check=500. Provider labels construct balanced L/validation; only L labels enter weight updates. Validation/check truth is diagnostic only. Constant source mean/std and separate sampler/augmentation streams; fixed diagnostic weak views.","",
        "Uniform 25% unit removal at all six sites leaves 42,034 actual real parameters vs 73,898. Same victims with/without compensation; no mask-only savings or extra inference maps. Uncompressed control receives the same continuation. Repair: 1000 opportunities, fresh SGD .003 and paired restarting sampler streams, inherit parent soft confidence statistics for SSL. CE-U control receives the SAME labeled/unlabeled views and joint BN, but zero pseudo-label loss; it is not a labeled-only baseline. No EMA/contrastive/triplet additions.","",
        "## Measured results","",f"State: {status}. Baselines complete: {len(baselines)}; continuations complete: {len(repairs)}. Tiny CPU checks are NOT scientific results.","",
        "| Family | Initialization | Continuation | Val before | Val after | Weak pseudo precision before/after (threshold .95) |",
        "| --- | --- | --- | ---: | ---: | --- |"]
    fmt=lambda v:f"{100*v:.2f}%" if v is not None else "none accepted"
    for r in repairs:
        a,b=r["diagnostics"][0],r["diagnostics"][-1]
        lines.append(f"| {r['family']} | {r['initialization']} | {r['method']} | {fmt(a['validation']['accuracy'])} | {fmt(b['validation']['accuracy'])} | "
                     f"{fmt(a['weak_check']['precision_at_095'])} / {fmt(b['weak_check']['precision_at_095'])} |")
    lines.extend(["","## Interpretation and decision limits","",
        "Inspect classification CE/accuracy together with pseudo-label coverage, precision, confidence weights and agreement/KL relative to the same family's SSL parent. Uncompressed continuing controls separate ordinary extra learning from recovery after damage. Report negative effects and collapse, not just the best condition. One seed, one label budget and one compression level cannot establish equivalence, robustness or a quaternion-specific mechanism. A small gain does not establish a new method.","",
        "This genuine low-label-from-initialization study is distinct from results/pseudolabel_repair, whose teacher was fully supervised. No transfer of its negative conclusion to the sibling's method. Decide whether SSL contributes a coherent empirical question to the paper only after all registered conditions finish; no draft commitment/title/email is automatically sent."])
    (output / "REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(study,config_name=CONFIG):
    if len(study["seeds"])!=1 or not set(study["families"])<={"quaternion","circulant"} or study["fraction"]!=.25:
        raise ValueError("This bounded pilot supports one seed, Q/circulant and 25% unit removal")
    torch.set_num_threads(study["cpu_threads"]);device=torch.device(study["device"])
    api,balanced=sibling_api(ROOT / study["sibling_root"])
    output=ROOT / study["output"];output.mkdir(parents=True,exist_ok=True)
    names=source_names(study)+[config_name];hashes={n:file_sha256(ROOT / n) for n in names}
    protocol={"config":study,"hashes":hashes,"environment":{name:metadata.version(name) for name in ("torch","torchvision","numpy","Pillow")},
        "python":sys.version,"device":str(device),"gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None,"official_test_evaluated":False}
    if (output / "protocol.json").exists() and read(output / "protocol.json")!=protocol: raise ValueError("SSL pilot sources/environment/config changed")
    write_json(output / "protocol.json",protocol)
    (raw,targets),unused_test=load_cached_cifar(ROOT / study["archive"]);del unused_test
    images=raw.transpose(0,2,3,1).copy()
    base=read(ROOT / study["sibling_root"] / study["source_ssl_config"]);model_config=read(ROOT / study["model_config"])
    mean=torch.tensor(base["mean"],device=device,dtype=torch.float32).reshape(1,3,1,1);std=torch.tensor(base["std"],device=device,dtype=torch.float32).reshape(1,3,1,1)
    baselines=[];repairs=[];summarize(study,protocol,baselines,repairs,"running",output)
    for seed in study["seeds"]:
        ids=partition(targets,study,seed,balanced)
        if study.get("smoke_check"):
            ids["unlabeled"]=ids["unlabeled"][:32]
            # Remove unused source images entirely from the adapter's U eligibility.
            eligible=np.r_[ids["labeled"],ids["validation"],ids["fit"],ids["check"],ids["unlabeled"]]
            images_current=images[eligible];targets_current=targets[eligible]
            offset=0;local_ids={}
            for key,value in ids.items():
                local_ids[key]=np.arange(offset,offset+len(value),dtype=np.int64);offset+=len(value)
            ids=local_ids
        else: images_current,targets_current=images,targets
        seed_folder=output / f"seed_{seed}";seed_folder.mkdir(parents=True,exist_ok=True)
        path=seed_folder / "partition_ids.npz"
        if path.exists():
            with np.load(path) as saved:
                if set(saved.files)!=set(ids) or any(not np.array_equal(saved[k],v) for k,v in ids.items()): raise ValueError("Frozen SSL partition changed")
        else: np.savez_compressed(path,**ids)
        write_json(seed_folder / "data.json",{"counts":{k:len(v) for k,v in ids.items()},"partition_sha256":file_sha256(path),
            "optimization_labels":len(ids["labeled"]),"calibration_never_weight_training":True,"official_test_evaluated":False})
        config=ssl_config(study,base,api,seed,"soft_ssl")
        diagnostics=api.KeyedViews(images_current,ids["check"].tolist(),api.WeakStrong(config),seed+970000)
        diagnostic_loader=DataLoader(diagnostics,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(320000+seed))
        weak=torch.cat([views[0] for views in diagnostic_loader]).to(device)
        validation=torch.as_tensor(images_current[ids["validation"]].transpose(0,3,1,2).copy(),device=device)
        validation_truth=torch.as_tensor(targets_current[ids["validation"]],device=device)
        check_truth=targets_current[ids["check"]]

        def callback_for(folder,parent_logits):
            def callback(model,step,pilot):
                metric,clean_logits=evaluate(model,validation,validation_truth,mean,std,128)
                value=logits_on(model,weak)
                state=pilot.soft_state.state_dict() if pilot.soft_state is not None else None
                pseudo=pseudo_diagnostic(value,check_truth,config.get("pseudo_temperature",.5),state,parent_logits)
                folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder / f"predictions_step_{step:05d}.npz",validation_logits=clean_logits,weak_check_logits=value.numpy(),
                    validation_ids=ids["validation"],check_ids=ids["check"])
                return {"validation":metric,"weak_check":pseudo,"real_trainable_parameters":sum(p.numel() for p in model.parameters())}
            return callback

        for family in study["families"]:
            family_folder=seed_folder / family;family_folder.mkdir(parents=True,exist_ok=True)
            baseline_folder=family_folder / "baseline";seed_everything(seed);model=make(model_config,family).to(device)
            identity={"protocol_sha256":file_sha256(output / "protocol.json"),"family":family,"seed":seed,"config":config,"phase":"baseline"}
            metric_file=baseline_folder / "metrics.json"
            if metric_file.exists():
                parent=read(metric_file)
                if parent["identity"]!=identity or parent["model_sha256"]!=file_sha256(baseline_folder / "model.pt"): raise ValueError("Completed SSL baseline changed")
                model.load_state_dict(torch.load(baseline_folder / "model.pt",map_location=device,weights_only=True)["model"])
            else:
                parent=train(model,config,images_current,targets_current,ids,api,baseline_folder,identity,
                    study["baseline_epochs"]*study["steps_per_epoch"],study["steps_per_epoch"],[0],callback_for(baseline_folder,None))
                parent.update(identity=identity,family=family,seed=seed,model_sha256=file_sha256(baseline_folder / "model.pt"));write_json(metric_file,parent)
            baselines.append(parent);summarize(study,protocol,baselines,repairs,"baselines/continuations",output)
            reference_logits=logits_on(model,weak)
            initial_folder=family_folder / "initializations";initial_folder.mkdir(parents=True,exist_ok=True)
            manifest=initial_folder / "manifest.json"
            if not manifest.exists():
                fit_raw=torch.as_tensor(images_current[ids["fit"]].transpose(0,3,1,2).copy(),device=device)
                fit=prepare_batch(fit_raw,mean,std);check=weak[:study["equivalence_images"]]
                corrected,records=chain(model,study["fraction"],study,fit,check)
                pruned,plain_records=chain(model,study["fraction"],study,fit,check,frozen=[r["victims"] for r in records])
                for name,current in (("uncompressed",model),("greedy_comp",corrected),("greedy_pruned",pruned)):
                    torch.save({"model":{k:v.detach().cpu() for k,v in current.state_dict().items()}},initial_folder / f"{name}.pt")
                write_json(manifest,{"compensated_sites":records,"pruned_sites":plain_records,"parent_sha256":parent["model_sha256"],
                    "model_hashes":{n:file_sha256(initial_folder / f"{n}.pt") for n in study["initializations"]}})
                del fit_raw,fit,corrected,pruned
            frozen=read(manifest)
            if frozen["parent_sha256"]!=parent["model_sha256"] or any(file_sha256(initial_folder / f"{n}.pt")!=h for n,h in frozen["model_hashes"].items()):
                raise ValueError("Frozen SSL compression inputs changed")
            for initialization in study["initializations"]:
                narrow=model_config if initialization=="uncompressed" else reduced_config(model_config,"cnn6",study["fraction"])
                for method in study["repair_methods"]:
                    folder=family_folder / "repair" / initialization / method
                    continuation=ssl_config(study,base,api,seed,method,repair=True)
                    identity={"protocol_sha256":file_sha256(output / "protocol.json"),"family":family,"seed":seed,"config":continuation,
                        "phase":"repair","initialization":initialization,"initial_sha256":frozen["model_hashes"][initialization],
                        "parent_sha256":parent["model_sha256"]}
                    metric_file=folder / "metrics.json"
                    if metric_file.exists():
                        result=read(metric_file)
                        if result["identity"]!=identity or result["model_sha256"]!=file_sha256(folder / "model.pt"): raise ValueError("Completed SSL continuation changed")
                    else:
                        current=make(narrow,family).to(device).eval();current.load_state_dict(torch.load(initial_folder / f"{initialization}.pt",map_location=device,weights_only=True)["model"])
                        expected=73898 if initialization=="uncompressed" else 42034
                        if sum(p.numel() for p in current.parameters())!=expected: raise AssertionError("Physical compact budget mismatch")
                        result=train(current,continuation,images_current,targets_current,ids,api,folder,identity,study["repair_steps"],
                            study["steps_per_epoch"],study["repair_snapshots"],callback_for(folder,reference_logits),parent["soft_state"],study["baseline_epochs"])
                        result.update(identity=identity,family=family,seed=seed,initialization=initialization,method=method,model_sha256=file_sha256(folder / "model.pt"))
                        write_json(metric_file,result);del current
                    repairs.append(result);summarize(study,protocol,baselines,repairs,"baselines/continuations",output)
            del model,reference_logits
    expected=len(study["families"])*len(study["seeds"])*len(study["initializations"])*len(study["repair_methods"])
    if len(repairs)!=expected: raise AssertionError("Incomplete registered SSL pilot")
    if any(file_sha256(ROOT / n)!=h for n,h in hashes.items()): raise ValueError("Registered SSL sources changed during training")
    summarize(study,protocol,baselines,repairs,"completed",output)


if __name__=="__main__":
    name=sys.argv[1] if len(sys.argv)>1 else CONFIG;run(read(ROOT / name),name)
