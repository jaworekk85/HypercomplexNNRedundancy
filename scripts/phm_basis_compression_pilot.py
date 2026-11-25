"""Bounded learned-PHM compression pilot; prepare first, then run/resume.

See the registered configuration for hypotheses, controls, limits and prior art.
No scientific fitting occurs in --prepare (synthetic data/copies only).
"""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch

from scripts.run_activation_queue import lock
from src.cnn_training import evaluate, prepare_batch, seed_everything
from src.compression.phm_activation_fit import normalized, objective
from src.compression.phm_basis_pilot import (change_audit, compact_block, counts, fit_consumer,
                                          initialize_consumer, kept_groups, set_repair_policy, update_audit)
from src.compression.structure_catalogue import collect, mse, raw_moments
from src.data.cifar_cached import file_sha256, load_cached_cifar
from src.models.cifar_phm_resnet import CifarPHMResNet20
from src.models.phm_conv import PHMConv2d
from src.resnet_training import train
from src.trainable_map_repair import atomic_save, repair

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/phm_basis_compression_pilot.json"
NEW = [CONFIG, "scripts/phm_basis_compression_pilot.py", "src/compression/phm_basis_pilot.py",
       "tests_torch/test_phm_basis_pilot.py"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_model(baseline, hidden=None):
    return CifarPHMResNet20(baseline["quaternion_widths"], baseline["classes"], baseline["phm_dimensions"], hidden)


def save_model(model, path):
    atomic_save({"model": {n:v.detach().cpu() for n,v in model.state_dict().items()},
                 "hidden_widths": [b.conv1.out_channels//4 for b in model.blocks]}, path)
    return file_sha256(path)


def load_model(path, baseline, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(baseline, saved["hidden_widths"]).to(device)
    model.load_state_dict(saved["model"])
    return model.eval()


def register(study):
    parent = ROOT/study["parent"]
    old = read(parent/"protocol.json")
    for name, digest in {**old["source_hashes"], **old["input_hashes"]}.items():
        if file_sha256(ROOT/name) != digest:
            raise ValueError(f"Frozen predecessor changed: {name}")
    if read(parent/"state.json")["status"] != "completed":
        raise ValueError("PHM parent baseline has not completed")
    metrics = read(parent/"qstem_phm4/metrics.json")
    if file_sha256(parent/"qstem_phm4/model.pt") != metrics["checkpoint_sha256"]:
        raise ValueError("Selected parent checkpoint changed")
    baseline = read(ROOT/study["baseline_config"])
    paths = [parent/"protocol.json", parent/"summary.json", parent/"qstem_phm4/model.pt",
             parent/"qstem_phm4/metrics.json", ROOT/study["baseline_config"]]
    definition = {
        "config": study,
        "source_hashes": {**old["source_hashes"], **{p:file_sha256(ROOT/p) for p in NEW}},
        "input_hashes": {**old["input_hashes"], **{p.relative_to(ROOT).as_posix():file_sha256(p) for p in paths}},
        "environment": {"python":platform.python_version(), "torch":torch.__version__,
                        "numpy":np.__version__, "cuda":torch.version.cuda},
        "test_evaluated": False,
    }
    output = ROOT/study["output"]
    output.mkdir(parents=True, exist_ok=True)
    path = output/"protocol.json"
    if path.exists():
        if read(path) != definition:
            raise ValueError("Registered pilot changed; add a new study instead")
    else:
        write(path, definition)
    return definition, baseline


def load_data(baseline, device):
    (images, labels), unused_test = load_cached_cifar(ROOT/baseline["archive"])
    del unused_test
    with np.load(ROOT/baseline["split_source"]) as split:
        ids = {"train":split["train"], "validation":split["validation"],
               "fit":split["calibration"][::2], "check_calibration":split["calibration"][1::2]}
    sets = [set(v.tolist()) for v in ids.values()]
    expected = {"train":48000, "validation":1000, "fit":500, "check_calibration":500}
    if any(len(ids[k]) != n for k,n in expected.items()) or sum(map(len, sets)) != 50000 or sorted(set.union(*sets)) != list(range(50000)):
        raise ValueError("Data split is overlapping, duplicated, or incomplete")
    stats = read(ROOT/baseline["data_statistics"])["statistics"]
    mean, std = [torch.tensor(stats[k], device=device).reshape(1,3,1,1) for k in ("mean","std")]
    tensors = {s:torch.as_tensor(images[v], device=device) for s,v in ids.items()}
    targets = {s:torch.as_tensor(labels[v], device=device) for s,v in ids.items()}
    return tensors, targets, mean, std


class Paused(Exception):
    pass


def evaluate_splits(model, tensors, targets, mean, std, batch):
    return {s:evaluate(model, tensors[s], targets[s], mean, std, batch)[0]
            for s in ("validation", "check_calibration")}


def decision(study, local, rows):
    if len(local) < 2*len(study["policies"]) or len(rows) < 5:
        return {"status":"pending"}
    gain = []
    for index in study["blocks"]:
        values = {r["policy"]:r["check_mse"] for r in local if r["block"] == index}
        gain.append((values["preserve"]-values["adapt"])/max(values["preserve"], 1e-12))
    models = {r["case"]:r for r in rows}
    def ce(name):
        return models[name]["after"]["validation"]["cross_entropy"]
    ce_gain = ce("compressed_preserve")-ce("compressed_adapt")
    control_gain = ce("uncompressed_preserve")-ce("uncompressed_adapt")
    accuracy_gain = models["compressed_adapt"]["after"]["validation"]["accuracy"]-models["compressed_preserve"]["after"]["validation"]["accuracy"]
    gate = study["continuation"]
    local_pass = float(np.mean(gain)) >= gate["minimum_mean_relative_check_mse_gain"] and min(gain) >= -gate["maximum_single_block_relative_check_mse_increase"]
    checks = {"local_heldout_signal":local_pass, "accuracy_gain":accuracy_gain >= gate["minimum_accuracy_gain"],
              "lower_compressed_ce":ce_gain > 0, "compression_specific_ce_gain":ce_gain-control_gain >= 0}
    return {"status":"evaluated", "continue_to_replication":all(checks.values()), "checks":checks,
            "relative_check_mse_gains":gain, "mean_relative_check_mse_gain":float(np.mean(gain)),
            "validation_accuracy_gain":accuracy_gain, "compressed_ce_gain":ce_gain,
            "uncompressed_ce_gain":control_gain, "compression_specific_ce_gain":ce_gain-control_gain,
            "statistical_significance_claimed":False, "automatic_followup":False}


def report(study, output, local, rows, status, synthetic=False):
    result = {"status":status, "config":study, "local_results":local, "results":rows,
              "decision":decision(study,local,rows), "scientific_results":not synthetic,
              "test_evaluated":False, "updated_utc":now()}
    write(output/"summary.json", result)
    lines = ["# Learned PHM basis compression pilot", "", f"Status: {status}.", "",
             "## Hypothesis", "", study["hypothesis"], "", "## Measured results", "",
             "| Case | Stored reals | Trainable reals | Validation before (%) | Validation after (%) | CE after |",
             "|---|---:|---:|---:|---:|---:|" ]
    for row in rows:
        before = row["before"]["validation"]["accuracy"]
        after = row["after"]["validation"]
        lines.append(f"| {row['case']} | {row['counts']['stored_real_parameters']} | {row['counts']['trainable_real_parameters']} | {100*before:.2f} | {100*after['accuracy']:.2f} | {after['cross_entropy']:.6g} |")
    lines += ["", "| Block | Policy | Initial check MSE | Fitted check MSE | Best fit step |",
              "|---|---|---:|---:|---:|"]
    for row in local:
        lines.append(f"| {row['block']} | {row['policy']} | {row['initial_check_mse']:.6g} | {row['check_mse']:.6g} | {row['best_step']} |")
    lines += ["", "## Interpretation and limits", "", study["scope"], "", study["repair_scope"],
              "", study["scratch_scope"], "", study["evaluation"], "", study["decision"],
              "", "Gate outcome: `"+json.dumps(result["decision"])+"`.",
              "", "Scratch row uses the CE-selected checkpoint; its final validation is separately retained in result.json.",
              "One parent seed, two local sites, one split; no significance, general superiority or novelty claim.",
              "Prior art: "+", ".join(study["prior_art"])+" (references.bib).", ""]
    (output/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def execute(study, baseline, source, tensors, targets, mean, std, output, synthetic=False):
    output.mkdir(parents=True, exist_ok=True)
    local, rows = [], []
    device = mean.device
    stop = output/"STOP"
    def state(**fields):
        write(output/"state.json", {"status":"running", "pid":os.getpid(), **fields, "updated_utc":now()})
    def check_stop():
        if stop.exists():
            raise Paused()
    try:
        check_stop()
        masks = {str(i):kept_groups(source.blocks[i], study["retained_fraction"]) for i in study["blocks"]}
        pruned = deepcopy(source).eval()
        for i in study["blocks"]:
            pruned.blocks[i] = compact_block(source.blocks[i], masks[str(i)])
        manifest = {"masks":masks, "parent_counts":counts(source), "compact_counts":counts(pruned),
                    "hidden_widths":[b.conv1.out_channels//4 for b in pruned.blocks], "n":4, "rank":4}
        if (output/"manifest.json").exists() and read(output/"manifest.json") != manifest:
            raise ValueError("Pruning manifest changed")
        write(output/"manifest.json",manifest)
        state(stage="local_reconstruction", completed_local=0)
        for index in study["blocks"]:
            check_stop()
            cache = output/f"block_{index}_moments.pt"
            meta = output/f"block_{index}_moments.json"
            if cache.exists():
                if not meta.exists() or file_sha256(cache) != read(meta)["sha256"]:
                    raise ValueError("Moment cache changed/incomplete")
                pack = torch.load(cache, map_location="cpu", weights_only=True)
            else:
                pack = {}
                channels = component_indices_for(source.blocks[index], masks[str(index)])
                for split, key in (("fit","fit"),("check_calibration","check")):
                    state(stage="collecting_moments",block=index,split=split)
                    images = prepare_batch(tensors[split],mean,std)
                    inputs, outputs = collect(source,index,images,study["collection_batch_size"])
                    pack[key] = raw_moments(inputs[:,channels],outputs)
                    del images,inputs,outputs
                    check_stop()
                atomic_save(pack,cache)
                write(meta,{"sha256":file_sha256(cache), "masks":masks[str(index)],
                            "fit_images":len(tensors["fit"]),"check_images":len(tensors["check_calibration"])})
            original = pruned.blocks[index].conv2
            for policy in study["policies"]:
                check_stop()
                identity = f"block_{index}_{policy}"
                folder = output/"local"/identity
                folder.mkdir(parents=True,exist_ok=True)
                if (folder/"result.json").exists():
                    row = read(folder/"result.json")
                    if file_sha256(folder/"model.pt") != row["checkpoint_sha256"]:
                        raise ValueError("Local fitted artifact changed")
                else:
                    layer = initialize_consumer(original,policy,study["reinitialization_seed"]+index)
                    normalized_pack,scale = normalized(pack["fit"])
                    audit = update_audit(layer,lambda probe: objective(probe,normalized_pack,scale))
                    initial = {k:mse(layer.effective_kernel().detach(),v) for k,v in pack.items()}
                    def progress(step,loss):
                        state(stage="local_fit",case=identity,step=step,best_normalized_fit_mse=loss)
                        print(f"LOCAL {identity}: {step}/{study['local_fit']['steps']} best_fit={loss:.6g}",flush=True)
                    fitted = fit_consumer(layer,pack["fit"],study["local_fit"],folder,stop,progress)
                    if fitted is None:
                        raise Paused()
                    layer = layer.float()
                    atomic_save({"model":layer.state_dict()},folder/"model.pt")
                    row = {"block":index,"policy":policy,**fitted,"counts":counts(layer),
                           "initial_fit_mse":initial["fit"],"initial_check_mse":initial["check"],
                           "fit_mse":mse(layer.effective_kernel().detach(),pack["fit"]),
                           "check_mse":mse(layer.effective_kernel().detach(),pack["check"]),
                           "gradient_update_audit":audit,"checkpoint_sha256":file_sha256(folder/"model.pt")}
                    write(folder/"result.json",row)
                local.append(row)
                report(study,output,local,rows,"running",synthetic)
            del pack
        for case,policy,compressed in [
            (f"compressed_{p}",p,True) for p in study["policies"]
        ] + [(f"uncompressed_{p}",p,False) for p in ("preserve","adapt")]:
            check_stop()
            folder = output/case
            folder.mkdir(parents=True,exist_ok=True)
            if (folder/"result.json").exists():
                row = read(folder/"result.json")
                if file_sha256(folder/"model.pt") != row["checkpoint_sha256"]:
                    raise ValueError("Completed repair artifact changed")
            else:
                model = deepcopy(pruned if compressed else source)
                if compressed:
                    for index in study["blocks"]:
                        fitted = torch.load(output/"local"/f"block_{index}_{policy}"/"model.pt", map_location=device, weights_only=True)
                        model.blocks[index].conv2.load_state_dict(fitted["model"])
                set_repair_policy(model,policy,study["blocks"])
                before_model = deepcopy(model).cpu()
                before = evaluate_splits(model,tensors,targets,mean,std,study["repair"]["batch_size"])
                previous_successful = [0]
                if (folder/"last.pt").exists():
                    previous_successful[0] = torch.load(folder/"last.pt",map_location="cpu",weights_only=False)["count"]["successful"]
                def on_epoch(row,counters):
                    if counters["successful"] <= previous_successful[0]:
                        raise RuntimeError("Repair epoch had no optimizer updates")
                    previous_successful[0] = counters["successful"]
                    state(stage="repair",case=case,epoch=row["epoch"],counts=counters,validation=row["validation"])
                    print(f"REPAIR {case}: {row['epoch']}/{study['repair']['epochs']} val={100*row['validation']['accuracy']:.2f}%",flush=True)
                state(stage="repair",case=case,epoch=0)
                model,history,counters,done = repair(model,tensors,targets,mean,std,study["repair"],study["seed"],folder/"last.pt",stop,on_epoch)
                if not done:
                    raise Paused()
                after = evaluate_splits(model,tensors,targets,mean,std,study["repair"]["batch_size"])
                sha = save_model(model,folder/"model.pt")
                replay = load_model(folder/"model.pt",baseline,device)
                replay_metrics = evaluate_splits(replay,tensors,targets,mean,std,study["repair"]["batch_size"])
                assert_metrics(after,replay_metrics)
                changes = change_audit(before_model,model)
                permitted = {f"blocks.{i}.conv2" for i in study["blocks"]} if policy != "preserve" else set()
                if any(value != 0 for name,value in changes["basis_max_abs_change"].items() if name not in permitted):
                    raise AssertionError("A frozen basis changed during repair")
                row = {"case":case,"policy":policy,"compressed":compressed,"counts":counts(model),
                       "before":before,"after":after,"history":history,"optimizer_counts":counters,
                       "changes":changes,"checkpoint_sha256":sha,"checkpoint_bytes":(folder/"model.pt").stat().st_size,
                       "reload_replay":replay_metrics,"test_evaluated":False}
                write(folder/"result.json",row)
                del model,replay,before_model
                gc.collect()
            rows.append(row)
            report(study,output,local,rows,"running",synthetic)
        check_stop()
        folder = output/"compact_scratch"
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/"result.json").exists():
            row = read(folder/"result.json")
            if file_sha256(folder/"model.pt") != row["checkpoint_sha256"]:
                raise ValueError("Scratch checkpoint changed")
        else:
            config = deepcopy(baseline)
            config["epochs"] = study["scratch_epochs"]
            seed_everything(study["seed"])
            model = make_model(baseline,manifest["hidden_widths"]).to(device)
            set_repair_policy(model,"scratch",study["blocks"])
            before = evaluate_splits(model,tensors,targets,mean,std,config["batch_size"])
            def on_scratch_epoch(row,best):
                state(stage="compact_scratch",epoch=row["epoch"],target_epochs=config["epochs"],validation=row["validation"],best_epoch=best["epoch"])
                print(f"SCRATCH: {row['epoch']}/{config['epochs']} val={100*row['validation']['accuracy']:.2f}% time={row['epoch_seconds']:.1f}s",flush=True)
            state(stage="compact_scratch",epoch=0,target_epochs=config["epochs"])
            model,details,done = train(model,tensors,targets,mean,std,config,folder/"last.pt",stop,on_scratch_epoch)
            if not done:
                raise Paused()
            after = evaluate_splits(model,tensors,targets,mean,std,config["batch_size"])
            if after["validation"]["accuracy"] != details["selected_validation"]["accuracy"] or abs(after["validation"]["cross_entropy"]-details["selected_validation"]["cross_entropy"]) > 1e-6:
                raise AssertionError("Scratch selected validation replay failed")
            sha = save_model(model,folder/"model.pt")
            replica = load_model(folder/"model.pt",baseline,device)
            replay = evaluate_splits(replica,tensors,targets,mean,std,config["batch_size"])
            assert_metrics(after,replay)
            row = {**details,"case":"compact_scratch","policy":"scratch","compressed":True,"counts":counts(model),
                   "optimizer_counts":details["counts"],"before":before,"after":after,"checkpoint_sha256":sha,
                   "checkpoint_bytes":(folder/"model.pt").stat().st_size,"reload_replay":replay,"test_evaluated":False}
            write(folder/"result.json",row)
        rows.append(row)
        expected_count = manifest["compact_counts"]["stored_real_parameters"]
        if any(r["counts"]["stored_real_parameters"] != expected_count for r in rows if r["compressed"]):
            raise AssertionError("Compact architecture stored counts differ across conditions")
        report(study,output,local,rows,"completed",synthetic)
        write(output/"state.json",{"status":"completed","completed_local_fits":len(local),"completed_models":len(rows),"updated_utc":now()})
        return read(output/"summary.json")
    except Paused:
        report(study,output,local,rows,"paused",synthetic)
        write(output/"state.json",{"status":"paused","updated_utc":now()})
        return None
    except BaseException as error:
        write(output/"state.json",{"status":"failed","error":f"{type(error).__name__}: {error}","updated_utc":now()})
        report(study,output,local,rows,"failed",synthetic)
        raise


def component_indices_for(block, kept):
    width = block.conv1.out_channels//4
    return [component*width+k for component in range(4) for k in kept]


def assert_metrics(expected, actual):
    for split in expected:
        if expected[split]["accuracy"] != actual[split]["accuracy"] or abs(expected[split]["cross_entropy"]-actual[split]["cross_entropy"]) > 1e-6:
            raise AssertionError(f"Serialized validation replay failed: {split}")


def smoke(study):
    tiny = deepcopy(study)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    output = ROOT/".cache"/f"phm_basis_smoke_{stamp}"
    tiny.update(device="cpu",scratch_epochs=2,local_fit={"steps":3,"learning_rate":.01,"checkpoint_interval":1})
    tiny["repair"].update(epochs=1,batch_size=4,mixed_precision=False)
    baseline = read(ROOT/study["baseline_config"])
    baseline.update(quaternion_widths=[2,4,8],classes=2,batch_size=4,epochs=2,mixed_precision=False)
    seed_everything(19)
    source = make_model(baseline).eval()
    sizes = {"train":8,"validation":4,"fit":4,"check_calibration":4}
    tensors = {s:torch.randint(256,(n,3,8,8),dtype=torch.uint8) for s,n in sizes.items()}
    targets = {s:torch.arange(n)%2 for s,n in sizes.items()}
    result = execute(tiny,baseline,source,tensors,targets,torch.zeros(1,3,1,1),torch.ones(1,3,1,1),output,True)
    assert result["status"] == "completed" and len(result["local_results"]) == 6 and len(result["results"]) == 6
    return {"status":"passed","output":output.relative_to(ROOT).as_posix(),"scientific_results":False}


def prepare(study):
    tests = subprocess.run([sys.executable,"-B","-m","pytest","-q","tests_torch/test_phm_basis_pilot.py"],cwd=ROOT,capture_output=True,text=True)
    print(tests.stdout,flush=True)
    if tests.returncode:
        raise RuntimeError(tests.stdout+tests.stderr)
    with lock("compression_curve_runner.lock"):
        checked_smoke = smoke(study)
        audits = {}
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        baseline = read(ROOT/study["baseline_config"])
        seed_everything(study["seed"])
        full = make_model(baseline).cuda().train()
        for index in study["blocks"]:
            full.blocks[index] = compact_block(full.blocks[index],kept_groups(full.blocks[index],study["retained_fraction"]))
        for policy in study["policies"]:
            model = deepcopy(full).train()
            set_repair_policy(model,policy,study["blocks"])
            x = torch.randn(study["repair"]["batch_size"],3,32,32,device="cuda")
            y = torch.arange(len(x),device="cuda")%baseline["classes"]
            audits[policy] = {"counts":counts(model),"updates":update_audit(model,lambda probe:torch.nn.functional.cross_entropy(probe(x[:8]),y[:8]))}
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=study["repair"]["learning_rate"])
            scaler = torch.amp.GradScaler("cuda")
            successful = []
            optimizer.register_step_post_hook(lambda *args:successful.append(True))
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    loss = torch.nn.functional.cross_entropy(model(x),y)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite preflight AMP loss")
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            if not successful:
                raise RuntimeError("All GPU preflight updates skipped")
            audits[policy]["amp_successful_updates"] = len(successful)
            del model,optimizer,scaler,x,y
        definition,_ = register(study)
        write(ROOT/study["output"]/"readiness.json",{"status":"passed","source_hashes":definition["source_hashes"],
              "tests":tests.stdout.strip(),"synthetic_smoke":checked_smoke,"gpu_discarded_copy_audits":audits,
              "scientific_results":False,"updated_utc":now()})
        write(ROOT/study["output"]/"state.json",{"status":"ready_not_started","updated_utc":now()})
        print("READY: protocol frozen; no scientific fitting or test inference",flush=True)


def run(study):
    with lock("compression_curve_runner.lock"):
        definition,baseline = register(study)
        ready = read(ROOT/study["output"]/"readiness.json")
        if ready["status"] != "passed" or ready["source_hashes"] != definition["source_hashes"]:
            raise ValueError("Readiness does not cover current registered sources")
        device = torch.device(study["device"])
        seed_everything(study["seed"])
        source = make_model(baseline).to(device).eval()
        source.load_state_dict(torch.load(ROOT/study["parent"]/"qstem_phm4/model.pt",map_location=device,weights_only=True)["model"])
        tensors,targets,mean,std = load_data(baseline,device)
        parent = evaluate_splits(source,tensors,targets,mean,std,study["repair"]["batch_size"])
        expected = read(ROOT/study["parent"]/"qstem_phm4/metrics.json")["selected_validation"]
        assert_metrics({"validation":expected},{"validation":parent["validation"]})
        write(ROOT/study["output"]/"parent_validation_replay.json",parent)
        result = execute(study,baseline,source,tensors,targets,mean,std,ROOT/study["output"])
        if result is not None:
            register(study)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true")
    args = parser.parse_args()
    config = read(ROOT/CONFIG)
    torch.set_num_threads(config["cpu_threads"])
    if args.run:
        run(config)
    else:
        prepare(config)
