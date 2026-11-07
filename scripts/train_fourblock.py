"""Train three resumable CNN8 baselines with the unchanged original trainer."""

import csv
import gc
import json
from pathlib import Path
import platform
import shutil
import sys
from unittest.mock import patch

import numpy as np
import torch

from scripts import cifar_cnn as original
from scripts.cifar_cnn import write_json
from src.cnn_training import prepare_batch,seed_everything
from src.data.cifar_cached import file_sha256,load_cached_cifar
from src.evaluation.block_ablation import capture_blocks,grouped_spatial
from src.models.cifar_fourblock import make_fourblock

ROOT=Path(__file__).resolve().parents[1]
SOURCES=["scripts/train_fourblock.py","src/models/cifar_fourblock.py"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def train(config,tensors,labels,mean,std,folder,hashes):
    folder=Path(folder)
    def progress(*args,**kwargs):
        print(f"[CNN8 seed {config['seed']}]",*args,**kwargs)
    with patch.object(original,"make_model",make_fourblock),patch.object(original,"source_hashes",lambda:hashes),patch.object(original,"print",progress,create=True):
        try:
            report=original.train_one(config,"quaternion",tensors,labels,mean,std,folder,True)
        except UnboundLocalError as error:
            # The frozen trainer writes its final report before deleting a
            # local 'saved' variable absent on completed-epoch resume. Do not
            # change protected old source or repeat training to fix cleanup.
            if "saved" not in str(error) or not (folder / "metrics.json").exists():
                raise
            report=read(folder / "metrics.json")
            last=torch.load(folder / "last.pt",map_location="cpu",weights_only=True)
            exported=torch.load(folder / "model.pt",map_location="cpu",weights_only=True)
            if last["epoch"]!=config["epochs"] or report["source_hashes"]!=hashes or not all(torch.equal(value,last["best_state"][name]) for name,value in exported["model"].items()):
                raise
            write_json(folder / "resume_cleanup_compatibility.json",{"handled_exception":str(error),"scope":"Frozen trainer cleanup only; completed numeric artifacts verified; no extra optimizer updates"})
    last=torch.load(folder / "last.pt",map_location="cpu",weights_only=True)
    with (folder / "training.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(last["history"][0]))
        writer.writeheader();writer.writerows(last["history"])
    return report


def run(study):
    output=ROOT / study["output"] / "architecture_b"
    output.mkdir(parents=True,exist_ok=True)
    old=read(ROOT / study["data_source"] / "summary.json")
    if old["source_hashes"]!=original.source_hashes():
        raise ValueError("The original numerical training source changed")
    hashes={**original.source_hashes(),**{name:file_sha256(ROOT / name) for name in SOURCES}}
    protocol={"study":study,"source_hashes":hashes,"source_summary_sha256":file_sha256(ROOT / study["data_source"] / "summary.json"),
              "split_sha256":file_sha256(ROOT / study["data_source"] / "split_indices.npz"),"statistics_sha256":file_sha256(ROOT / study["data_source"] / "data.json"),
              "environment":{"python":sys.version,"torch":str(torch.__version__),"numpy":np.__version__,"cuda":torch.version.cuda,"platform":platform.platform()},
              "official_test_evaluated":False}
    if (output / "training_protocol.json").exists() and read(output / "training_protocol.json")!=protocol:
        raise ValueError("CNN8 sources/config changed; preserve the run and use another output")
    write_json(output / "training_protocol.json",protocol)
    if (output / "sources.json").exists():
        print("All three CNN8 trainings already complete",flush=True)
        return read(output / "sources.json")
    torch.set_num_threads(study["cpu_threads"])
    device=torch.device(study["device"])
    (images,targets),unused_test=load_cached_cifar(ROOT / study["archive"])
    del unused_test
    with np.load(ROOT / study["data_source"] / "split_indices.npz") as saved:
        indices={key:saved[key] for key in ("train","validation","calibration")}
    statistics=read(ROOT / study["data_source"] / "data.json")["statistics"]
    mean=torch.tensor(statistics["mean"],dtype=torch.float32,device=device).reshape(1,3,1,1)
    std=torch.tensor(statistics["std"],dtype=torch.float32,device=device).reshape(1,3,1,1)
    tensors={key:torch.as_tensor(images[index],device=device) for key,index in indices.items()}
    labels={key:torch.as_tensor(targets[index],device=device) for key,index in indices.items()}
    sources=[]
    for seed in study["seeds"]:
        config={**old["config"],"seed":seed,"models":["quaternion"],"quaternion_widths":study["architecture_b_widths"],"dropout":study["architecture_b_dropout"]}
        for key in ("epochs","batch_size","learning_rate","minimum_learning_rate","weight_decay","mixed_precision"):
            config[key]=study["architecture_b_epochs"] if key=="epochs" else study[key]
        source=output / f"seed_{seed}"
        source.mkdir(parents=True,exist_ok=True)
        for name in ("split_indices.npz","data.json"):
            if not (source / name).exists():
                shutil.copyfile(ROOT / study["data_source"] / name,source / name)
        result=train(config,tensors,labels,mean,std,source / "quaternion",hashes)
        if not all(result["first_update"]["checks"].values()) or result["test_evaluated"]:
            raise ValueError("CNN8 first-update/test-use gate failed")
        cache=source / "block2_calibration.npz"
        if not cache.exists():
            seed_everything(seed)
            model=make_fourblock(config).to(device).eval()
            model.load_state_dict(torch.load(source / "quaternion/model.pt",map_location=device,weights_only=True)["model"])
            captured=capture_blocks(model,prepare_batch(tensors["calibration"],mean,std),256)[1]
            np.savez_compressed(cache,trained=grouped_spatial(captured,True))
            del model,captured
        record={"architecture":"cnn8","seed":seed,"path":str(source.relative_to(ROOT)).replace("\\","/"),"config":config,
                "checkpoint_sha256":file_sha256(source / "quaternion/model.pt"),"cache_path":str(cache.relative_to(ROOT)).replace("\\","/"),
                "cache_sha256":file_sha256(cache),"metrics":result,"official_test_evaluated":False}
        write_json(source / "source.json",record)
        sources.append(record)
        gc.collect();torch.cuda.empty_cache()
    result={"status":"completed","protocol":protocol,"sources":sources,"official_test_evaluated":False}
    write_json(output / "sources.json",result)
    return result


if __name__=="__main__":
    run(read(ROOT / "configs/internal_replication.json"))
