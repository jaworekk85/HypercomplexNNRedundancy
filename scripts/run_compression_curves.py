"""Run the two prepared curve studies serially, with protected input snapshots."""

from datetime import datetime,timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

from scripts import compression_curve as curve
from scripts.cifar_cnn import write_json
from scripts.finetune_compression import sources_of
from src.data.cifar_cached import file_sha256

ROOT=Path(__file__).resolve().parents[1]
CONFIGS=["configs/compression_curve.json","configs/compression_curve_eurosat.json"]
STATUS=ROOT / "results/compression_curve_runner.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def snapshot(config_path):
    study=curve.read(ROOT / config_path)
    files={config_path,*curve.SOURCES,*curve.trainer.SOURCES,"scripts/run_compression_curves.py",
           "scripts/finetune_compression.py","scripts/multisource_compensation.py",
           "src/models/cifar_cnn_torch.py","src/models/cifar_fourblock.py","src/models/eurosat_cnn.py",
           "scripts/eurosat_cnn.py","src/data/cifar_cached.py"}
    if study["dataset"]=="cifar":
        ms=curve.read(ROOT / "configs/multisource_compensation.json")
        sources=sources_of(ms)
        files.update(("configs/multisource_compensation.json",ms["replication_config"],ms["archive"],
                      str(Path(ms["data_source"]) / "data.json")))
    else:
        sources=[{"architecture":arch,"seed":row["seed"],"path":row["path"]}
                 for row in curve.read(ROOT / study["training_output"] / "summary.json")["runs"]
                 for arch in ("cnn6","real_cnn6")]
        files.update(str(Path(study["data"]) / name) for name in ("eurosat_nrgb.npz","data.json","split_indices.npz"))
        files.update(str(Path(study["training_output"]) / name) for name in ("summary.json","protocol.json"))
    sources=[row for row in sources if row["architecture"] in study["architectures"] and row["seed"] in study["seeds"]]
    expected={(arch,seed) for arch in study["architectures"] for seed in study["seeds"]}
    if {(row["architecture"],row["seed"]) for row in sources}!=expected:
        raise ValueError("Missing or duplicate baseline source")
    audits=[]
    for row in sources:
        folder=ROOT / row["path"];kind=curve.kind_of(row["architecture"])
        report=curve.read(folder / kind / "metrics.json")
        if not all(report["first_update"]["checks"].values()):
            raise ValueError("A baseline did not pass its actual first-update audit")
        for name,digest in report["source_hashes"].items():
            if file_sha256(ROOT / name)!=digest:
                raise ValueError(f"Frozen baseline numerical source changed: {name}")
            files.add(name)
        files.update(str(Path(row["path"]) / name) for name in ("summary.json","split_indices.npz","data.json",f"{kind}/metrics.json",f"{kind}/model.pt"))
        with np.load(folder / "split_indices.npz") as saved:
            used=[saved[name] for name in ("train","calibration","validation")]
            if len(np.unique(np.concatenate(used)))!=sum(map(len,used)):
                raise ValueError("Overlapping baseline train/fit/validation indices")
        audits.append({"architecture":row["architecture"],"seed":row["seed"],"path":row["path"],
                       "baseline_first_update_passed":True,"historical_baseline_test_use":report["test_evaluated"]})
    return {"config":study,"hashes":{str(Path(name)).replace("\\","/"):file_sha256(ROOT / name) for name in sorted(files)},
            "baselines":audits,"environment":{"python":sys.version,"torch":str(torch.__version__),"numpy":np.__version__,
                                            "cuda_build":torch.version.cuda,"platform":platform.platform()},
            "official_test_evaluated_by_runner":False}


def validate_smoke():
    folder=ROOT / ".cache/smoke/compression_curve_review_20261104"
    summary=curve.read(folder / "summary.json")
    if summary["status"]!="completed":
        raise ValueError("The fresh small pipeline check is incomplete")
    for name,digest in summary["protocol"]["source_hashes"].items():
        if file_sha256(ROOT / name)!=digest:
            raise ValueError(f"Numerical source changed after the fresh smoke check: {name}")


def run():
    import msvcrt
    lock_path=ROOT / ".cache/compression_curve_runner.lock"
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open("a+b") as lock:
        lock.seek(0);lock.write(b"1");lock.flush();lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            raise RuntimeError("The curve runner is already active; do not start a second copy") from None
        state={"status":"checking_inputs","runner_pid":os.getpid(),"started_utc":now(),"datasets":[],"official_test_evaluated":False}
        write_json(STATUS,state)
        try:
            validate_smoke()
            for config_path in CONFIGS:
                study=curve.read(ROOT / config_path);output=ROOT / study["output"]
                manifest=snapshot(config_path);path=output / "input_manifest.json"
                if path.exists() and curve.read(path)!=manifest:
                    raise ValueError(f"Curve inputs changed; preserve existing results: {study['output']}")
                write_json(path,manifest)
                log=ROOT / "results" / f"compression_curve{'_eurosat' if study['dataset']=='eurosat' else ''}_run.log"
                state.update(status="running",dataset=study["dataset"],log=str(log.relative_to(ROOT)).replace("\\","/"),updated_utc=now())
                with log.open("ab",buffering=0) as handle:
                    handle.write(f"\n=== Curve runner {now()}: {config_path} ===\n".encode("utf-8"))
                    child=subprocess.Popen([sys.executable,"-u","-m","scripts.compression_curve",config_path],cwd=ROOT,
                                           stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    state["child_pid"]=child.pid;write_json(STATUS,state)
                    print(f"Started {study['dataset']}, child PID {child.pid}, log {log.relative_to(ROOT)}",flush=True)
                    code=child.wait()
                if code:
                    raise RuntimeError(f"{study['dataset']} curve exited {code}; inspect {log.relative_to(ROOT)}")
                for name,digest in manifest["hashes"].items():
                    if file_sha256(ROOT / name)!=digest:
                        raise ValueError(f"A protected input changed during the curve: {name}")
                result=curve.read(output / "summary.json")
                if result["status"]!="completed":
                    raise ValueError("Curve process ended without a completed summary")
                state["datasets"].append({"dataset":study["dataset"],"status":"completed","finished_utc":now(),"rows":len(result["rows"])})
                write_json(STATUS,state)
            state.update(status="completed",updated_utc=now());write_json(STATUS,state)
        except Exception as error:
            state.update(status="failed",error=str(error),updated_utc=now());write_json(STATUS,state)
            raise


if __name__=="__main__":
    run()
