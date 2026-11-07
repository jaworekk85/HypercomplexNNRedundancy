"""Fixed-step low-label repair with online self-generated hard pseudo-labels.

Confidence filtering is established SSL background (Sohn2020FixMatch), but
this two-view crop/flip pilot is not a FixMatch reproduction. BN statistics
and dropout are fixed in eval mode throughout; trainable parameters still
receive gradients. Label budgets apply to post-compression repair only.
"""

from pathlib import Path
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.cnn_training import prepare_batch,seed_everything


def budget_partition(train_ids,targets,budgets,pool_size,seed):
    """Nested class-balanced labeled subsets and a common disjoint U pool.

    Dataset-provider labels construct the stratified subsets. Only the
    selected labeled targets are exposed to the optimizer.
    """
    rng=np.random.default_rng(seed)
    maximum=max(budgets);reserved=[];label_ids={int(b):[] for b in budgets}
    for value in np.unique(targets[train_ids]):
        candidates=rng.permutation(train_ids[targets[train_ids]==value])
        if len(candidates)<maximum: raise ValueError("Not enough examples per class")
        reserved.extend(candidates[:maximum].tolist())
        for budget in budgets: label_ids[int(budget)].extend(candidates[:budget].tolist())
    remaining=np.setdiff1d(train_ids,np.asarray(reserved,dtype=np.int64))
    if len(remaining)<pool_size: raise ValueError("Not enough disjoint unlabeled examples")
    unlabeled=rng.permutation(remaining)[:pool_size]
    return {k:np.asarray(v,dtype=np.int64) for k,v in label_ids.items()},unlabeled


def hard_targets(logits,threshold):
    probabilities=logits.detach().float().softmax(dim=1)
    confidence,labels=probabilities.max(dim=1)
    return labels.detach(),(confidence>=threshold).detach()


def repair(model,labeled,targets,unlabeled,mean,std,config,seed,method,folder,context,
           evaluate_callback,teacher_probabilities=None):
    """Deterministic paired streams; epoch-free schedule and resumable steps.

    targets contains ONLY selected human labels. U targets never enter this
    interface. Diagnostic callbacks cannot alter the fixed schedule/threshold.
    """
    if method not in ("supervised","self_pseudo","teacher_pseudo"):
        raise ValueError("Unknown repair method")
    if len(labeled)!=len(targets) or not len(unlabeled): raise ValueError("Bad data partition")
    if method=="teacher_pseudo" and (teacher_probabilities is None or len(teacher_probabilities)!=len(unlabeled)):
        raise ValueError("Frozen teacher probabilities required")
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    seed_everything(seed);model.eval()  # Freeze running BN and disable dropout; gradients remain enabled.
    device=mean.device
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    generators={"label_order":torch.Generator().manual_seed(seed+110000),
                "unlabeled_order":torch.Generator().manual_seed(seed+120000),
                "label_augmentation":torch.Generator(device=device).manual_seed(seed+130000),
                "unlabeled_augmentation":torch.Generator(device=device).manual_seed(seed+140000)}
    identity={"config":config,"seed":seed,"method":method,"context":context}
    history=[];updates=0;accepted_total=0;seen_total=0;first_update=None;start_step=0
    last=folder / "last.pt"
    if last.exists():
        saved=torch.load(last,map_location="cpu",weights_only=True)
        if saved["identity"]!=identity: raise ValueError("Repair resume identity changed")
        model.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"])
        start_step=saved["step"];history=saved["history"];updates=saved["updates"]
        accepted_total=saved["accepted_total"];seen_total=saved["seen_total"];first_update=saved["first_update"]
        for name,generator in generators.items(): generator.set_state(saved["generator_states"][name])
        torch.set_rng_state(saved["torch_rng"])
        if device.type=="cuda": torch.cuda.set_rng_state_all(saved["cuda_rng"])

    def snapshot(step):
        measured=evaluate_callback(model,step)
        return {"step":step,"optimizer_updates":updates,"pseudo_accepted":accepted_total,"pseudo_seen":seen_total,**measured}

    def save(step):
        state={"identity":identity,"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
               "optimizer":optimizer.state_dict(),"step":step,"history":history,"updates":updates,
               "accepted_total":accepted_total,"seen_total":seen_total,"first_update":first_update,
               "generator_states":{k:g.get_state() for k,g in generators.items()},"torch_rng":torch.get_rng_state(),
               "cuda_rng":torch.cuda.get_rng_state_all() if device.type=="cuda" else []}
        torch.save(state,folder / "last.tmp");(folder / "last.tmp").replace(last)

    if not history:
        history.append(snapshot(0));save(0)
    started=time.perf_counter()
    for step in range(start_step+1,config["steps"]+1):
        model.eval();optimizer.zero_grad(set_to_none=True)
        progress=(step-1)/max(1,config["steps"]-1)
        lr=config["minimum_learning_rate"]+.5*(config["learning_rate"]-config["minimum_learning_rate"])*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups: group["lr"]=lr
        loss=None
        if len(labeled):
            index=torch.randint(len(labeled),(config["batch_size"],),generator=generators["label_order"]).to(device)
            x=prepare_batch(labeled[index],mean,std,True,generators["label_augmentation"])
            loss=F.cross_entropy(model(x),targets[index])
        uindex=torch.randint(len(unlabeled),(config["batch_size"],),generator=generators["unlabeled_order"]).to(device)
        if method!="supervised":
            raw=unlabeled[uindex]
            with torch.no_grad():
                if method=="self_pseudo":
                    pseudo,mask=hard_targets(model(prepare_batch(raw,mean,std)),config["confidence_threshold"])
                else:
                    confidence,pseudo=teacher_probabilities[uindex].detach().max(dim=1)
                    mask=confidence>=config["confidence_threshold"]
            augmented=prepare_batch(raw,mean,std,True,generators["unlabeled_augmentation"])
            seen_total+=len(mask);accepted_total+=int(mask.sum())
            if bool(mask.any()):
                # Masked mean over the complete U batch, not just accepted samples.
                unsupervised=(F.cross_entropy(model(augmented),pseudo,reduction="none")*mask).mean()
                loss=config["unlabeled_loss_weight"]*unsupervised if loss is None else loss+config["unlabeled_loss_weight"]*unsupervised
        if loss is not None:
            if not bool(torch.isfinite(loss)): raise RuntimeError("Nonfinite low-label repair loss")
            before={n:p.detach().clone() for n,p in model.named_parameters()} if first_update is None else None
            loss.backward()
            if before is not None:
                gradients={n:{"present":p.grad is not None,"finite":p.grad is not None and bool(torch.isfinite(p.grad).all()),
                              "norm":float(p.grad.norm()) if p.grad is not None else None} for n,p in model.named_parameters()}
                if not all(r["present"] and r["finite"] for r in gradients.values()):
                    raise AssertionError("Missing/nonfinite tracked repair gradient")
            optimizer.step();updates+=1
            if before is not None:
                first_update={"parameters":{n:{**gradients[n],"maximum_update":float((p.detach()-before[n]).abs().max())} for n,p in model.named_parameters()}}
        # A batch with neither labels nor accepted pseudo-labels causes NO AdamW step.
        if step in config["snapshots"]: history.append(snapshot(step))
        if step%config["checkpoint_every"]==0 or step==config["steps"]: save(step)
    torch.save({"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"identity":identity},folder / "model.pt")
    return {"history":history,"optimizer_updates":updates,"attempted_steps":config["steps"],
            "pseudo_accepted":accepted_total,"pseudo_seen":seen_total,"first_update":first_update,
            "elapsed_seconds_this_invocation":time.perf_counter()-started,"official_test_evaluated":False}
