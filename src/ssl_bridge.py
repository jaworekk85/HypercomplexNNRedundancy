"""Reuse the sibling's genuine low-label SSL without modifying its sources.

Known context: Sohn2020FixMatch, Chen2023SoftMatch. This is an adaptation of
the shared pilot, not a reproduction of either full published recipe.
"""

from copy import deepcopy
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.cnn_training import seed_everything


def sibling_api(root):
    # Imported sibling modules stay read-only, including their __pycache__.
    sys.dont_write_bytecode=True
    location=str(Path(root).resolve())
    if location not in sys.path: sys.path.insert(0,location)
    import fewshot_metric.pilot as pilot
    from fewshot_metric.data import balanced_splits
    return pilot,balanced_splits


class SSLAdapter(nn.Module):
    """Expose existing pooled features/logits; add ZERO trainable parameters."""
    def __init__(self,backbone):
        super().__init__();self.backbone=backbone

    def forward(self,images,normalize=False):
        if normalize: raise ValueError("The selected soft-PL/CE pilot uses unnormalized features")
        features=self.backbone.forward_features(images).mean(dim=(2,3))
        return features,self.backbone.readout(features)


def partition(targets,study,seed,balanced_splits):
    labeled,validation=balanced_splits(targets,study["labels_per_class"],study["validation_per_class"],seed)
    remainder=np.setdiff1d(np.arange(len(targets)),np.r_[labeled,validation])
    shuffled=np.random.default_rng(seed+730000).permutation(remainder)
    a,b=study["calibration_fit_images"],study["calibration_check_images"]
    if len(shuffled)<=a+b: raise ValueError("No unlabeled pool remains")
    return {"labeled":np.asarray(labeled,dtype=np.int64),"validation":np.asarray(validation,dtype=np.int64),
            "fit":shuffled[:a],"check":shuffled[a:a+b],"unlabeled":np.sort(shuffled[a+b:])}


def components_api(config,images,targets,ids,api):
    train=api.labeled_dataset(images,ids["labeled"].tolist(),targets,config)
    # The sibling uses this slot ONLY as an exclusion set when forming U.
    # Scientific manifests keep validation and held-out calibration separate.
    excluded=np.r_[ids["validation"],ids["fit"],ids["check"]].tolist()
    pilot=api.PilotTraining(config,train,{"train":ids["labeled"].tolist(),"validation":excluded})
    if pilot.unlabeled is None or pilot.unlabeled.indices!=ids["unlabeled"].tolist(): raise AssertionError("Actual U pool differs from the registered disjoint partition")
    generator=torch.Generator().manual_seed(config["seed"]+610000)
    loader=DataLoader(train,batch_size=config["batch_size"],shuffle=True,generator=generator,num_workers=0,drop_last=False)
    return pilot,loader,generator


def rng_state(generator):
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,"loader":generator.get_state()}


def restore_rng(state,generator):
    random.setstate(state["python"]);np.random.set_state(state["numpy"]);torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if state["cuda"] is not None: torch.cuda.set_rng_state_all(state["cuda"])


def step_audit_before(model):
    before={n:p.detach().clone() for n,p in model.named_parameters()};rows={}
    for name,p in model.named_parameters():
        if p.grad is None or not bool(torch.isfinite(p.grad).all()): raise AssertionError(f"Missing/nonfinite SSL gradient: {name}")
        rows[name]={"gradient_norm":float(p.grad.norm())}
        if name.endswith("kernel"): rows[name]["component_gradient_norms"]=p.grad.flatten(1).norm(dim=1).tolist()
    return before,rows


def step_audit_after(model,before,rows):
    for name,p in model.named_parameters():
        rows[name]["maximum_update"]=float((p.detach()-before[name]).abs().max())
        if name.endswith("kernel"): rows[name]["component_update_norms"]=(p.detach()-before[name]).flatten(1).norm(dim=1).tolist()
    kernels=[r for k,r in rows.items() if k.endswith("kernel")]
    checks={"six_tracked_structured_layers":len(kernels)==6,
            "all_kernel_components_have_gradients_and_updates":all(v>0 for r in kernels for k in ("component_gradient_norms","component_update_norms") for v in r[k]),
            "all_parameter_tensors_updated":all(r["maximum_update"]>0 for r in rows.values())}
    if not all(checks.values()): raise AssertionError(f"Actual SSL first-update gate failed: {checks}")
    return {"parameters":rows,"checks":checks}


def train(backbone,config,images,targets,ids,api,folder,identity,steps,steps_per_epoch,
          snapshots,callback,parent_soft_state=None,start_epoch=0,max_chunks=None):
    """Exact epoch-boundary resume; every method gets identical paired streams.

    Only selected L labels enter api.labeled_dataset. U stores no labels.
    Callback truth is diagnostic; neither threshold nor schedule adapts to it.
    Unsaved partial epochs replay after interruption (at most 382 steps).
    """
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    seed_everything(config["seed"]+810000+start_epoch)
    model=SSLAdapter(backbone);device=next(model.parameters()).device
    pilot,loader,generator=components_api(config,images,targets,ids,api)
    if parent_soft_state is not None and pilot.soft_state is not None: pilot.soft_state.load_state_dict(deepcopy(parent_soft_state))
    optimizer=torch.optim.SGD(model.parameters(),lr=config["learning_rate"],momentum=config["momentum"],weight_decay=config["weight_decay"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=steps)
    done=chunks=0;history=[];diagnostics=[];audit=None;elapsed=0.
    last=folder / "latest.pt"
    if last.exists():
        saved=torch.load(last,map_location="cpu",weights_only=False)
        if saved["identity"]!=identity: raise ValueError("SSL resume identity/source/config changed")
        backbone.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"]);scheduler.load_state_dict(saved["scheduler"])
        pilot.load_state_dict(saved["pilot_state"]);restore_rng(saved["rng"],generator)
        done,chunks=saved["steps"],saved["chunks"];history=saved["history"];diagnostics=saved["diagnostics"];audit=saved["first_update"];elapsed=saved["elapsed_seconds"]
    if not diagnostics: diagnostics.append({"step":0,**callback(backbone,0,pilot)})
    started=time.perf_counter();new_chunks=0
    while done<steps:
        epoch=start_epoch+chunks;pilot.start_epoch(epoch);iterator=iter(loader);count=min(steps_per_epoch,steps-done)
        totals={};loss_total=0.;chunk_start=time.perf_counter()
        for local_step in range(count):
            model.train()
            try: views,labels=next(iterator)
            except StopIteration: iterator=iter(loader);views,labels=next(iterator)
            optimizer.zero_grad(set_to_none=True)
            loss,_,_,extra=pilot.step(model,views,labels,device)
            if not bool(torch.isfinite(loss)): raise FloatingPointError("Nonfinite low-label SSL loss")
            loss.backward()
            captured=step_audit_before(model) if audit is None else None
            optimizer.step()
            if captured is not None: audit=step_audit_after(model,*captured)
            scheduler.step();done+=1;loss_total+=float(loss.detach())
            for key,value in extra.items(): totals[key]=totals.get(key,0)+value
            if done in snapshots and local_step+1<count: diagnostics.append({"step":done,**callback(backbone,done,pilot)})
        totals.update(pilot.finish_epoch())
        diagnostics.append({"step":done,**callback(backbone,done,pilot)})
        chunks+=1;new_chunks+=1
        history.append({"epoch":epoch+1,"steps":done,"mean_loss":loss_total/count,"seconds":time.perf_counter()-chunk_start,
                        "learning_rate_end":optimizer.param_groups[0]["lr"],**totals})
        state={"identity":identity,"model":{k:v.detach().cpu().clone() for k,v in backbone.state_dict().items()},"optimizer":optimizer.state_dict(),
            "scheduler":scheduler.state_dict(),"pilot_state":pilot.state_dict(),"rng":rng_state(generator),"steps":done,"chunks":chunks,
            "history":history,"diagnostics":diagnostics,"first_update":audit,"elapsed_seconds":elapsed+time.perf_counter()-started}
        torch.save(state,folder / "latest.pending.pt");(folder / "latest.pending.pt").replace(last)
        accuracy=diagnostics[-1].get("validation",{}).get("accuracy")
        print(f"{folder.name}: steps={done}/{steps}, val={accuracy}, chunk={history[-1]['seconds']:.1f}s",flush=True)
        if max_chunks is not None and new_chunks>=max_chunks and done<steps: return {"status":"stopped_at_epoch_boundary","steps":done}
    result={"status":"completed","steps":done,"history":history,"diagnostics":diagnostics,"first_update":audit,
            "soft_state":pilot.soft_state.state_dict() if pilot.soft_state is not None else None,
            "elapsed_seconds":elapsed+time.perf_counter()-started,"official_test_evaluated":False}
    torch.save({"model":{k:v.detach().cpu() for k,v in backbone.state_dict().items()}},folder / "model.pt")
    return result
