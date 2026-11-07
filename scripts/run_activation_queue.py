"""Run the activation pilot after the existing, immutable curve queue."""

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts import run_curve_queue as previous
from scripts import activation_sensitivity as experiment
from scripts.cifar_cnn import write_json
from src.data.cifar_cached import file_sha256

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT / "results/activation_sensitivity_queue.json"


@contextmanager
def lock(name):
    import msvcrt
    path=ROOT / ".cache" / name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0);handle.write(b"1");handle.flush();handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            raise RuntimeError(f"Already active: {name}") from None
        try:
            yield
        finally:
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)


def busy(name):
    import msvcrt
    path=ROOT / ".cache" / name
    if not path.exists(): return False
    with path.open("r+b") as handle:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            return True
        handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        return False


def previous_completed():
    path=ROOT / previous.STATUS
    if not path.exists() or previous.base.curve.read(path)["status"]!="completed": return False
    queue=previous.base.curve.read(ROOT / previous.CONFIG)
    for cfg in [*queue["main_configs"],*queue["followups"]]:
        output=ROOT / previous.base.curve.read(ROOT / cfg)["output"]
        if not (output / "summary.json").exists() or previous.base.curve.read(output / "summary.json")["status"]!="completed": return False
    return True


def run():
    study=experiment.read(ROOT / experiment.CONFIG)
    readiness=experiment.read(ROOT / ".cache/activation_sensitivity_readiness.json")
    if readiness["status"]!="passed" or any(file_sha256(ROOT / name)!=digest for name,digest in readiness["source_hashes"].items()):
        raise ValueError("Activation checks do not cover the current numerical sources")
    names=set([*experiment.SOURCES,experiment.CONFIG,*study["curve_configs"].values(),
               "scripts/run_activation_queue.py","scripts/run_curve_queue.py","scripts/run_compression_curves.py",
               "configs/compression_curve_queue.json"])
    definition={"config":study,"hashes":{name:file_sha256(ROOT / name) for name in sorted(names)}}
    with lock("activation_sensitivity_queue.lock"):
        protocol=ROOT / "results/activation_sensitivity_queue_protocol.json"
        if protocol.exists() and experiment.read(protocol)!=definition:
            raise ValueError("Activation queue sources/config changed; preserve the registered queue")
        write_json(protocol,definition)
        state={"status":"waiting_for_curve_queue","queue_pid":os.getpid(),"started_utc":previous.base.now(),
               "activations":study["activations"],"baseline_trainings":6,"official_test_evaluated":False}
        write_json(STATUS,state)
        try:
            while True:
                if busy("compression_curve_queue.lock") or busy("compression_curve_runner.lock"):
                    state.update(status="waiting_for_curve_queue",updated_utc=previous.base.now());write_json(STATUS,state);time.sleep(30)
                elif previous_completed():
                    break
                else:
                    state.update(status="resuming_curve_queue",updated_utc=previous.base.now());write_json(STATUS,state)
                    try:
                        previous.run()
                    except RuntimeError as error:
                        if "already active" not in str(error): raise
            with lock("compression_curve_runner.lock"):
                if any(file_sha256(ROOT / name)!=digest for name,digest in definition["hashes"].items()):
                    raise ValueError("Registered activation sources changed while waiting")
                state.update(status="running",updated_utc=previous.base.now());write_json(STATUS,state)
                log=ROOT / "results/activation_sensitivity_run.log"
                with log.open("ab",buffering=0) as handle:
                    handle.write(f"\n=== Activation study {previous.base.now()} ===\n".encode())
                    child=subprocess.Popen([sys.executable,"-u","-m","scripts.activation_sensitivity"],cwd=ROOT,
                                           stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    state["child_pid"]=child.pid;write_json(STATUS,state)
                    code=child.wait()
                if code: raise RuntimeError(f"Activation study exited {code}; inspect {log.relative_to(ROOT)}")
                result=experiment.read(ROOT / study["output"] / "summary.json")
                if result["status"]!="completed": raise ValueError("Activation summary is incomplete")
                if any(file_sha256(ROOT / name)!=digest for name,digest in definition["hashes"].items()):
                    raise ValueError("Registered activation sources changed during training")
            state.update(status="completed",updated_utc=previous.base.now());write_json(STATUS,state)
        except Exception as error:
            state.update(status="failed",error=str(error),updated_utc=previous.base.now());write_json(STATUS,state)
            raise


if __name__=="__main__":
    run()
