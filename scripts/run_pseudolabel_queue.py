"""Append low-label repair after the immutable curve and activation queues."""

import os
from pathlib import Path
import subprocess
import sys
import time

from scripts import run_activation_queue as previous
from scripts import pseudolabel_repair as experiment
from scripts.cifar_cnn import write_json
from src.data.cifar_cached import file_sha256

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT / "results/pseudolabel_repair_queue.json"


def previous_completed():
    path=ROOT / previous.STATUS
    if not path.exists() or experiment.read(path)["status"]!="completed": return False
    study=experiment.read(ROOT / previous.experiment.CONFIG)
    result=ROOT / study["output"] / "summary.json"
    return previous.previous_completed() and result.exists() and experiment.read(result)["status"]=="completed"


def run():
    study=experiment.read(ROOT / experiment.CONFIG)
    readiness=experiment.read(ROOT / ".cache/pseudolabel_repair_readiness.json")
    if readiness["status"]!="passed" or any(file_sha256(ROOT / name)!=digest for name,digest in readiness["source_hashes"].items()):
        raise ValueError("Repair readiness does not cover the current numerical sources")
    names=set([*experiment.SOURCES,experiment.CONFIG,"scripts/run_pseudolabel_queue.py",
               "scripts/run_activation_queue.py","results/activation_sensitivity_queue_protocol.json",
               ".cache/pseudolabel_repair_readiness.json",*experiment.input_paths(study)])
    definition={"config":study,"hashes":{name:file_sha256(ROOT / name) for name in sorted(names)}}
    with previous.lock("pseudolabel_repair_queue.lock"):
        protocol=ROOT / "results/pseudolabel_repair_queue_protocol.json"
        if protocol.exists() and experiment.read(protocol)!=definition: raise ValueError("Registered repair queue changed")
        write_json(protocol,definition)
        state={"status":"waiting_for_activation_queue","queue_pid":os.getpid(),"started_utc":previous.previous.base.now(),
               "repair_runs":54,"labels_per_class_for_repair_only":study["repair_labels_per_class"],"official_test_evaluated":False}
        write_json(STATUS,state)
        try:
            while True:
                if any(previous.busy(name) for name in ("activation_sensitivity_queue.lock","compression_curve_queue.lock","compression_curve_runner.lock")):
                    state.update(status="waiting_for_activation_queue",updated_utc=previous.previous.base.now());write_json(STATUS,state);time.sleep(30)
                elif previous_completed():
                    break
                else:
                    state.update(status="resuming_previous_queue",updated_utc=previous.previous.base.now());write_json(STATUS,state)
                    try:
                        previous.run()
                    except RuntimeError as error:
                        if "already active" not in str(error).lower(): raise
            with previous.lock("compression_curve_runner.lock"):
                if any(file_sha256(ROOT / name)!=digest for name,digest in definition["hashes"].items()):
                    raise ValueError("Repair queue inputs changed while waiting")
                state.update(status="running",updated_utc=previous.previous.base.now());write_json(STATUS,state)
                log=ROOT / "results/pseudolabel_repair_run.log"
                with log.open("ab",buffering=0) as handle:
                    handle.write(f"\n=== Low-label post-compression repair {previous.previous.base.now()} ===\n".encode())
                    child=subprocess.Popen([sys.executable,"-u","-m","scripts.pseudolabel_repair"],cwd=ROOT,
                        stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    state["child_pid"]=child.pid;write_json(STATUS,state);code=child.wait()
                if code: raise RuntimeError(f"Repair exited {code}; inspect {log.relative_to(ROOT)}")
                if experiment.read(ROOT / study["output"] / "summary.json")["status"]!="completed":
                    raise ValueError("Repair output is incomplete")
                if any(file_sha256(ROOT / name)!=digest for name,digest in definition["hashes"].items()):
                    raise ValueError("Registered repair inputs changed during training")
            state.update(status="completed",updated_utc=previous.previous.base.now());write_json(STATUS,state)
        except Exception as error:
            state.update(status="failed",error=str(error),updated_utc=previous.previous.base.now());write_json(STATUS,state)
            raise


if __name__=="__main__":
    run()
