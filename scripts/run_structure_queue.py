"""Run a bounded structure control AFTER all previously authorized jobs."""

import os
from pathlib import Path
import subprocess
import sys
import time

from scripts import run_pseudolabel_queue as previous
from scripts import structure_control as experiment
from scripts.cifar_cnn import write_json
from src.data.cifar_cached import file_sha256

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT / "results/structure_control_queue.json"
lock=previous.previous.lock
busy=previous.previous.busy
now=previous.previous.previous.base.now


def previous_completed():
    if not previous.previous_completed(): return False
    if not (ROOT / previous.STATUS).exists() or experiment.read(ROOT / previous.STATUS)["status"]!="completed": return False
    config=experiment.read(ROOT / previous.experiment.CONFIG);path=ROOT / config["output"] / "summary.json"
    return path.exists() and experiment.read(path)["status"]=="completed"


def run():
    study=experiment.read(ROOT / experiment.CONFIG)
    readiness=experiment.read(ROOT / ".cache/structure_control_readiness.json")
    if readiness["status"]!="passed" or any(file_sha256(ROOT / n)!=h for n,h in readiness["source_hashes"].items()):
        raise ValueError("Structure-control readiness is stale")
    names=set([*experiment.SOURCES,experiment.CONFIG,*experiment.input_paths(study),"scripts/run_structure_queue.py",
        "scripts/review_qnn_compression.py","results/pseudolabel_repair_queue_protocol.json",".cache/structure_control_readiness.json"])
    definition={"config":study,"hashes":{n:file_sha256(ROOT / n) for n in sorted(names)}}
    with lock("structure_control_queue.lock"):
        protocol=ROOT / "results/structure_control_queue_protocol.json"
        if protocol.exists() and experiment.read(protocol)!=definition: raise ValueError("Registered structure control changed")
        write_json(protocol,definition)
        state={"status":"waiting_for_existing_queue","queue_pid":os.getpid(),"started_utc":now(),"new_baseline_trainings":3,"official_test_evaluated":False}
        write_json(STATUS,state)
        try:
            while True:
                if any(busy(n) for n in ("pseudolabel_repair_queue.lock","activation_sensitivity_queue.lock","compression_curve_queue.lock","compression_curve_runner.lock")):
                    state.update(status="waiting_for_existing_queue",updated_utc=now());write_json(STATUS,state);time.sleep(30)
                elif previous_completed(): break
                else:
                    state.update(status="resuming_existing_queue",updated_utc=now());write_json(STATUS,state)
                    try: previous.run()
                    except RuntimeError as error:
                        if "already active" not in str(error).lower(): raise
            with lock("compression_curve_runner.lock"):
                if any(file_sha256(ROOT / n)!=h for n,h in definition["hashes"].items()): raise ValueError("Structure inputs changed while waiting")
                state.update(status="running",updated_utc=now());write_json(STATUS,state)
                log=ROOT / "results/structure_control_run.log"
                with log.open("ab",buffering=0) as handle:
                    handle.write(f"\n=== Matched structure control {now()} ===\n".encode())
                    child=subprocess.Popen([sys.executable,"-u","-m","scripts.structure_control"],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    state["child_pid"]=child.pid;write_json(STATUS,state);code=child.wait()
                if code: raise RuntimeError(f"Structure control exited {code}; inspect {log.relative_to(ROOT)}")
                if experiment.read(ROOT / study["output"] / "summary.json")["status"]!="completed": raise ValueError("Incomplete structure control")
                if any(file_sha256(ROOT / n)!=h for n,h in definition["hashes"].items()): raise ValueError("Structure inputs changed during execution")
                subprocess.run([sys.executable,"-u","-m","scripts.review_qnn_compression"],cwd=ROOT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
            state.update(status="completed",updated_utc=now());write_json(STATUS,state)
        except Exception as error:
            state.update(status="failed",error=str(error),updated_utc=now());write_json(STATUS,state);raise


if __name__=="__main__": run()
