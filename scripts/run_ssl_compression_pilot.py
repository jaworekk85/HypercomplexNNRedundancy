"""Resume the completed old chain, then run the registered genuine SSL pilot."""

import os
from pathlib import Path
import subprocess
import sys
import time

from scripts import run_structure_queue as previous
from scripts import ssl_compression_pilot as experiment
from scripts.cifar_cnn import write_json
from src.data.cifar_cached import file_sha256

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT / "results/ssl_compression_pilot_runner.json"


def previous_completed():
    path=ROOT / previous.STATUS
    if not previous.previous_completed() or not path.exists() or experiment.read(path)["status"]!="completed": return False
    config=experiment.read(ROOT / previous.experiment.CONFIG);summary=ROOT / config["output"] / "summary.json"
    return summary.exists() and experiment.read(summary)["status"]=="completed"


def run():
    study=experiment.read(ROOT / experiment.CONFIG)
    readiness=experiment.read(ROOT / ".cache/ssl_bridge_readiness.json")
    if readiness["status"]!="passed" or any(file_sha256(ROOT / n)!=h for n,h in readiness["source_hashes"].items()):
        raise ValueError("SSL pilot readiness is stale")
    names=set([*experiment.source_names(study),experiment.CONFIG,"scripts/run_ssl_compression_pilot.py",
               ".cache/ssl_bridge_readiness.json","results/structure_control_queue_protocol.json"])
    definition={"config":study,"hashes":{n:file_sha256(ROOT / n) for n in sorted(names)}}
    with previous.lock("ssl_compression_pilot.lock"):
        protocol=ROOT / "results/ssl_compression_pilot_runner_protocol.json"
        if protocol.exists() and experiment.read(protocol)!=definition: raise ValueError("Registered SSL pilot changed")
        write_json(protocol,definition)
        state={"status":"waiting_for_existing_queue","runner_pid":os.getpid(),"started_utc":previous.now(),
               "fresh_ssl_baselines":2,"continuations":12,"seeds":study["seeds"],"official_test_evaluated":False}
        write_json(STATUS,state)
        try:
            while True:
                if any(previous.busy(n) for n in ("structure_control_queue.lock","pseudolabel_repair_queue.lock",
                        "activation_sensitivity_queue.lock","compression_curve_queue.lock","compression_curve_runner.lock")):
                    state.update(status="waiting_for_existing_queue",updated_utc=previous.now());write_json(STATUS,state);time.sleep(30)
                elif previous_completed(): break
                else:
                    state.update(status="resuming_existing_queue",updated_utc=previous.now());write_json(STATUS,state)
                    try: previous.run()
                    except RuntimeError as error:
                        if "already active" not in str(error).lower(): raise
            with previous.lock("compression_curve_runner.lock"):
                if any(file_sha256(ROOT / n)!=h for n,h in definition["hashes"].items()): raise ValueError("SSL inputs changed while waiting")
                state.update(status="running",updated_utc=previous.now());write_json(STATUS,state)
                log=ROOT / "results/ssl_compression_pilot_run.log"
                with log.open("ab",buffering=0) as handle:
                    handle.write(f"\n=== Genuine low-label SSL compression pilot {previous.now()} ===\n".encode())
                    child=subprocess.Popen([sys.executable,"-B","-u","-m","scripts.ssl_compression_pilot"],cwd=ROOT,
                        stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    state["child_pid"]=child.pid;write_json(STATUS,state);code=child.wait()
                if code: raise RuntimeError(f"SSL pilot exited {code}; inspect {log.relative_to(ROOT)}")
                if experiment.read(ROOT / study["output"] / "summary.json")["status"]!="completed": raise ValueError("SSL pilot incomplete")
                if any(file_sha256(ROOT / n)!=h for n,h in definition["hashes"].items()): raise ValueError("SSL sources changed during training")
            state.update(status="completed",updated_utc=previous.now());write_json(STATUS,state)
        except Exception as error:
            state.update(status="failed",error=str(error),updated_utc=previous.now());write_json(STATUS,state);raise


if __name__=="__main__": run()
