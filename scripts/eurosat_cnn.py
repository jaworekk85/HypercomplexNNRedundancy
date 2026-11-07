"""Train quaternion and real four-band CNN6 baselines on EuroSAT with the unchanged CIFAR trainer (resumable).

The frozen trainer `scripts.cifar_cnn.train_one` builds models through its module-level `make_model`;
it is pointed at the four-band factory here instead of editing that hashed source. No test evaluation.
"""

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

import scripts.cifar_cnn as trainer
from src.data.cifar_cached import file_sha256
from src.models.eurosat_cnn import make_fourband

ROOT=Path(__file__).resolve().parents[1]
SOURCES=["scripts/eurosat_cnn.py","src/models/eurosat_cnn.py","scripts/prepare_eurosat.py",*trainer.SOURCES]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_eurosat(folder):
    """Images as int16 (lossless: maximum value 28,003) because torch has limited uint16 support."""
    with np.load(folder / "eurosat_nrgb.npz") as data:
        images,labels=data["images"],data["labels"]
    if images.max()>np.iinfo(np.int16).max: raise ValueError("Values exceed int16")
    with np.load(folder / "split_indices.npz") as split:
        indices={key:split[key] for key in split.files}
    return images.astype(np.int16),labels,indices,read(folder / "data.json")


def run(study):
    output=ROOT / study["output"];data=ROOT / study["data"]
    meta=read(data / "data.json")
    if file_sha256(data / "eurosat_nrgb.npz")!=meta["arrays_sha256"] or file_sha256(data / "split_indices.npz")!=meta["split_sha256"]:
        raise ValueError("Prepared EuroSAT arrays do not match their recorded hashes")
    protocol={"config":study,"source_hashes":{name:file_sha256(ROOT / name) for name in SOURCES},"data":meta,"official_test_evaluated":False}
    if (output / "protocol.json").exists() and read(output / "protocol.json")!=protocol:
        raise ValueError("EuroSAT sources/config changed; preserve the run and use another output")
    trainer.write_json(output / "protocol.json",protocol)
    trainer.make_model=make_fourband
    torch.set_num_threads(study["cpu_threads"]);device=torch.device(study["device"])
    images,targets,indices,meta=load_eurosat(data)
    used={key:indices[key] for key in ("train","validation","calibration")}
    tensors={key:torch.as_tensor(images[value],device=device) for key,value in used.items()}
    labels={key:torch.as_tensor(targets[value],device=device) for key,value in used.items()}
    del images
    mean=torch.tensor(meta["statistics"]["mean"],dtype=torch.float32,device=device).reshape(1,4,1,1)
    std=torch.tensor(meta["statistics"]["std"],dtype=torch.float32,device=device).reshape(1,4,1,1)
    keys=("quaternion_widths","real_widths","dropout","epochs","batch_size","learning_rate","minimum_learning_rate","weight_decay","optimizer","device","mixed_precision","cpu_threads","augmentation","checkpoint_selection")
    runs=[]
    for seed in study["seeds"]:
        folder=output / f"seed_{seed}";folder.mkdir(parents=True,exist_ok=True)
        config={"seed":seed,"models":study["models"],**{key:study[key] for key in keys},"dataset":"EuroSAT four-band (NIR,R,G,B)"}
        for name in ("split_indices.npz","data.json"): shutil.copyfile(data / name,folder / name)
        records=[trainer.train_one(config,kind,tensors,labels,mean,std,folder / kind,True) for kind in study["models"]]
        trainer.write_json(folder / "summary.json",{"config":config,"models":records,"official_test_evaluated":False})
        runs.append({"seed":seed,"path":str(folder.relative_to(ROOT)).replace("\\","/")})
        print("seed "+str(seed)+": "+", ".join(f"{r['kind']} val {100*r['selected_validation']['accuracy']:.2f}% (epoch {r['best_epoch']}, {r['real_trainable_parameters']} params)" for r in records),flush=True)
    trainer.write_json(output / "summary.json",{"status":"completed","runs":runs,"official_test_evaluated":False})


if __name__=="__main__":
    run(read(ROOT / (sys.argv[1] if len(sys.argv)>1 else "configs/eurosat_cnn.json")))
