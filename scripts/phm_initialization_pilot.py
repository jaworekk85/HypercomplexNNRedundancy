"""Bounded CPU PHM4 initialization study; --prepare checks, --run resumes."""
import argparse
from copy import deepcopy
from pathlib import Path
import platform
import time

import numpy as np
import torch

from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from scripts.paper_a_controls import now
from scripts.resnet20_baselines import make_model
from scripts.run_activation_queue import lock
from src.cnn_training import prepare_batch
from src.compression.phm_activation_fit import effective, gradient_audit, make_initialization, normalized, objective
from src.compression.structure_catalogue import collect, mse, raw_moments
from src.data.cifar_cached import file_sha256, load_cached_cifar
from src.trainable_map_repair import atomic_save

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/phm_initialization_pilot.json"
NEW = [CONFIG, "scripts/phm_initialization_pilot.py", "src/compression/phm_activation_fit.py",
       "tests_torch/test_phm_activation_fit.py"]


def register(config):
    parent = ROOT / config["parent"]
    old = curve.read(parent / "protocol.json")
    for name, digest in old["source_hashes"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f"Frozen scientific source changed: {name}")
    summary = curve.read(parent / "summary.json")
    if summary["status"] != "completed":
        raise ValueError("Parent catalogue incomplete")
    paths = [parent / n for n in ("protocol.json", "summary.json", "scientific_inputs.json")]
    for i in config["blocks"]:
        paths += [parent / f"catalogue_block_{i}.{extension}" for extension in ("pt", "json")]
    baseline = curve.read(ROOT / summary["config"]["baseline_config"])
    paths += [ROOT / summary["config"]["baseline_checkpoint"], ROOT / baseline["split_source"],
              ROOT / baseline["data_statistics"], ROOT / baseline["archive"]]
    definition = {"config": config, "source_hashes": {**old["source_hashes"], **{p: file_sha256(ROOT/p) for p in NEW}},
                  "input_hashes": {str(p.relative_to(ROOT)).replace("\\", "/"): file_sha256(p) for p in paths},
                  "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
                  "test_evaluated": False}
    output = ROOT / config["output"]
    path = output / "protocol.json"
    if path.exists() and curve.read(path) != definition:
        raise ValueError("Initialization study protocol changed")
    write_json(path, definition)
    return definition, summary


def prepare(config):
    from src.models.quaternion_conv_torch import QuaternionConv2d
    from src.models.phm_conv import PHMConv2d
    torch.manual_seed(17)
    q = QuaternionConv2d(2, 3).double()
    x, y = torch.randn(8,8,8,8), torch.randn(8,12,8,8)
    moments = raw_moments(x,y)
    audits = {}
    for init in config["initializations"]:
        layer = make_initialization(q, PHMConv2d(8,12).double(), init["name"], init["seed"])
        audits[f"{init['name']}_{init['seed']}"] = gradient_audit(layer, moments)
        if init["name"] == "from_q":
            torch.testing.assert_close(effective(q), effective(layer), atol=1e-12, rtol=1e-12)
    definition, _ = register(config)
    write_json(ROOT/config["output"]/"readiness.json",
               {"status": "passed", "source_hashes": definition["source_hashes"], "synthetic_gradient_audits": audits,
                "scientific_results": False, "updated_utc": now()})
    print("READY: all Q/PHM factors receive finite gradients and updates; no scientific fitting.", flush=True)


def data_packs(config, summary):
    output = ROOT/config["output"]
    parent = ROOT/config["parent"]
    baseline = curve.read(ROOT/summary["config"]["baseline_config"])
    source = make_model(baseline,"quaternion").eval()
    source.load_state_dict(torch.load(ROOT/summary["config"]["baseline_checkpoint"], map_location="cpu", weights_only=True)["model"])
    (images,_), unused_test = load_cached_cifar(ROOT/baseline["archive"])
    del unused_test
    with np.load(ROOT/baseline["split_source"]) as split:
        fit_ids, check_ids = split["calibration"][::2], split["calibration"][1::2]
        assert not set(fit_ids) & set(check_ids)
        assert not (set(fit_ids)|set(check_ids)) & (set(split["train"])|set(split["validation"]))
        assert len(fit_ids)==len(check_ids)==500
    stats = curve.read(ROOT/baseline["data_statistics"])["statistics"]
    mean,std = [torch.tensor(stats[k]).reshape(1,3,1,1) for k in ("mean","std")]
    selections = summary["selections"][config["selected_policy"]]["choices"]
    for index, identity in zip(config["blocks"],selections):
        rows = torch.load(parent/f"catalogue_block_{index}.pt",map_location="cpu",weights_only=False)
        qrow = next(r for r in rows if r["id"]==identity)
        svdrow = next(r for r in rows if r["family"]=="phm" and r["n"]==4 and r["kept"]==qrow["kept"])
        width = source.blocks[index].conv1.out_channels
        scalar = [c*width+k for c in range(4) for k in qrow["kept"]]
        cache = output/f"moments_block_{index}.pt"
        cache_record = output/f"moments_block_{index}.json"
        if cache.exists():
            if not cache_record.exists() or file_sha256(cache) != curve.read(cache_record)["sha256"]:
                raise ValueError("Moment cache incomplete or changed")
            pack = torch.load(cache,map_location="cpu",weights_only=True)
        else:
            pack = {}
            for name,ids in (("fit",fit_ids),("check",check_ids)):
                x = prepare_batch(torch.as_tensor(images[ids]),mean,std)
                inputs,targets = collect(source,index,x,16)
                pack[name] = raw_moments(inputs[:,scalar],targets)
                del x,inputs,targets
            atomic_save(pack,cache)
            write_json(cache_record,{"sha256":file_sha256(cache),"fit_images":len(fit_ids),"check_images":len(check_ids)})
        yield index,qrow["block"].conv2,svdrow["block"].conv2,pack


def fit_one(config, output, layer, moments, identity):
    pack,scale = normalized(moments)
    optimizer = torch.optim.Adam(layer.parameters(),lr=config["learning_rate"])
    initial = float(objective(layer,pack,scale).detach())
    best_loss, best_step, step = initial,0,0
    best = deepcopy(layer.state_dict())
    checkpoint = output/"resume.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint,map_location="cpu",weights_only=True)
        layer.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        initial,best_loss,best_step,step,best = (saved[k] for k in ("initial","best_loss","best_step","step","best"))
    def save():
        atomic_save({"model":layer.state_dict(),"optimizer":optimizer.state_dict(),"initial":initial,
                     "best_loss":best_loss,"best_step":best_step,"step":step,"best":best},checkpoint)
    if not checkpoint.exists():
        save()
    start = time.perf_counter()
    for step in range(step+1,config["steps"]+1):
        if (ROOT/config["output"]/"STOP").exists():
            step -= 1
            save()
            return None
        optimizer.zero_grad(set_to_none=True)
        loss = objective(layer,pack,scale)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite fit loss")
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in layer.parameters()):
            raise RuntimeError("Missing/nonfinite factor gradient")
        optimizer.step()
        value = float(objective(layer,pack,scale).detach())
        if not np.isfinite(value):
            raise RuntimeError("Nonfinite post-update loss")
        if value < best_loss:
            best_loss,best_step,best = value,step,deepcopy(layer.state_dict())
        if step % config["checkpoint_interval"]==0 or step==config["steps"]:
            save()
            write_json(ROOT/config["output"]/"state.json",
                       {"status":"running","case":identity,"step":step,"best_fit_normalized_mse":best_loss,"updated_utc":now()})
            print(f"{identity} step {step}/{config['steps']} best_fit={best_loss:.6g}",flush=True)
    final = float(objective(layer,pack,scale).detach())
    layer.load_state_dict(best)
    return {"initial_fit_normalized_mse":initial,"best_fit_normalized_mse":best_loss,"best_step":best_step,
            "final_fit_normalized_mse":final,"steps":config["steps"],"current_attempt_seconds":time.perf_counter()-start}


def report(config,rows,status):
    output = ROOT/config["output"]
    write_json(output/"summary.json",{"status":status,"config":config,"results":rows,"test_evaluated":False})
    lines = ["# PHM4 initialization diagnostic","",f"Status: {status}.","",
             "## Hypothesis","","Different initializations and activation fitting may explain the original PHM deficit.","",
             "## Measured results","",
             "| Block | Initialization | Seed | Consumer real parameters | Initial fit MSE | Best artifact fit MSE | Check MSE |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['block']} | {r['initialization']} | {r['seed']} | {r['real_parameters']} | {r['initial_fit_mse']:.6g} | {r['artifact_fit_mse']:.6g} | {r['artifact_check_mse']:.6g} |")
    lines += ["","## Interpretation and limits","",config["scope"],"",config["objective"],"",config["selection"],
              "","PHM4 contains Q but has 64 extra stored real basis coefficients per layer. This is not a classification result or an equal-budget improvement over Q. Three random initializations do not prove an optimizer optimum. Retain failures and all starts. Prior art: "+", ".join(config["prior_art"])+". See references.bib."]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(config):
    definition,summary = register(config)
    output = ROOT/config["output"]
    ready = curve.read(output/"readiness.json")
    if ready["status"]!="passed" or ready["source_hashes"]!=definition["source_hashes"]:
        raise ValueError("Readiness mismatch")
    rows = []
    with lock("compression_curve_runner.lock"):
        try:
            write_json(output/"state.json",{"status":"preparing_cpu_moments","updated_utc":now()})
            for index,q,svd,pack in data_packs(config,summary):
                for init in config["initializations"]:
                    identity = f"block_{index}_{init['name']}_{init['seed']}"
                    folder = output/identity
                    folder.mkdir(parents=True,exist_ok=True)
                    if (folder/"result.json").exists():
                        row = curve.read(folder/"result.json")
                        if file_sha256(folder/"best.pt")!=row["checkpoint_sha256"]:
                            raise ValueError("Completed artifact changed")
                        rows.append(row)
                        continue
                    if (output/"STOP").exists():
                        report(config,rows,"paused")
                        write_json(output/"state.json",{"status":"paused","case":identity,"updated_utc":now()})
                        return
                    layer = make_initialization(q,svd,init["name"],init["seed"])
                    audit = gradient_audit(layer,pack["fit"])
                    initial = mse(effective(layer).detach(),pack["fit"])
                    result = fit_one(config,folder,layer,pack["fit"],identity)
                    if result is None:
                        report(config,rows,"paused")
                        write_json(output/"state.json",{"status":"paused","case":identity,"updated_utc":now()})
                        return
                    layer = layer.float()
                    artifact_fit = mse(effective(layer).detach(),pack["fit"])
                    artifact_check = mse(effective(layer).detach(),pack["check"])
                    atomic_save({"model":layer.state_dict(),"initialization":init,"block":index},folder/"best.pt")
                    row = {"block":index,"initialization":init["name"],"seed":init["seed"],**result,
                           "initial_fit_mse":initial,"artifact_fit_mse":artifact_fit,"artifact_check_mse":artifact_check,
                           "real_parameters":sum(p.numel() for p in layer.parameters()),"factor_update_audit":audit,
                           "checkpoint_sha256":file_sha256(folder/"best.pt"),"test_evaluated":False}
                    write_json(folder/"result.json",row)
                    rows.append(row)
                    report(config,rows,"running")
            register(config)
            report(config,rows,"completed")
            write_json(output/"state.json",{"status":"completed","completed_cases":len(rows),"updated_utc":now()})
        except BaseException as error:
            report(config,rows,"failed")
            write_json(output/"state.json",{"status":"failed","error":f"{type(error).__name__}: {error}","updated_utc":now()})
            raise


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true")
    args = parser.parse_args()
    config = curve.read(ROOT/CONFIG)
    torch.set_num_threads(config["cpu_threads"])
    if args.run:
        run(config)
    else:
        prepare(config)
