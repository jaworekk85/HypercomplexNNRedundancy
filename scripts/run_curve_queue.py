"""Append the registered curve refinement after the unchanged two-study runner."""

from contextlib import contextmanager
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

from scripts import run_compression_curves as base
from scripts.cifar_cnn import write_json
from src.data.cifar_cached import file_sha256

ROOT=Path(__file__).resolve().parents[1]
CONFIG="configs/compression_curve_queue.json"
STATUS="results/compression_curve_queue.json"
PROTOCOL="results/compression_curve_queue_protocol.json"


def frozen_definition(queue):
    names={CONFIG,"scripts/run_curve_queue.py","scripts/run_compression_curves.py",*base.curve.SOURCES,
           *queue["main_configs"],*queue["followups"]}
    return {"queue":queue,"source_hashes":{name:file_sha256(ROOT / name) for name in sorted(names)}}


def validate(queue):
    if queue["additional_fractions"]!=[0.375,0.625] or queue["main_configs"]!=base.CONFIGS or len(queue["followups"])!=2:
        raise ValueError("Only the registered two-point follow-up is supported")
    for parent_path,followup_path in zip(queue["main_configs"],queue["followups"]):
        parent,followup=base.curve.read(ROOT / parent_path),base.curve.read(ROOT / followup_path)
        if followup["fractions"]!=queue["additional_fractions"] or followup["scratch"]["fractions"]!=queue["additional_fractions"]:
            raise ValueError("Both new fractions must include compression, fine-tuning and scratch controls")
        allowed={"output","fractions","scratch","scope","followup"}
        if {k:v for k,v in followup.items() if k not in allowed}!={k:v for k,v in parent.items() if k not in allowed}:
            raise ValueError("Follow-up changed policies, seeds, data or fine-tuning settings")
        if {k:v for k,v in parent["scratch"].items() if k!="fractions"}!={k:v for k,v in followup["scratch"].items() if k!="fractions"}:
            raise ValueError("Follow-up changed the scratch training budget")
        if followup["followup"]["parent_config_sha256"]!=file_sha256(ROOT / parent_path) or followup["output"]==parent["output"]:
            raise ValueError("Follow-up needs the frozen parent and a separate output")


@contextmanager
def queue_lock():
    import msvcrt
    path=ROOT / ".cache/compression_curve_queue.lock"
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0);handle.write(b"1");handle.flush();handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            raise RuntimeError("The curve follow-up queue is already active") from None
        try:
            yield
        finally:
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)


def main_runner_busy():
    import msvcrt
    path=ROOT / ".cache/compression_curve_runner.lock"
    if not path.exists():
        return False
    with path.open("r+b") as handle:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            return True
        handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        return False


def main_completed(queue):
    state=ROOT / "results/compression_curve_runner.json"
    if not state.exists() or base.curve.read(state)["status"]!="completed":
        return False
    for config_path in queue["main_configs"]:
        output=ROOT / base.curve.read(ROOT / config_path)["output"]
        if not (output / "summary.json").exists() or base.curve.read(output / "summary.json")["status"]!="completed":
            return False
    return True


def wait_for_main(queue,state,sleep=time.sleep):
    while True:
        if main_runner_busy():
            state.update(status="waiting_for_main",updated_utc=base.now())
            write_json(ROOT / STATUS,state)
            sleep(30)
        elif main_completed(queue):
            return
        else:
            state.update(status="resuming_main",updated_utc=base.now());write_json(ROOT / STATUS,state)
            try:
                base.run()
            except RuntimeError as error:
                # Another launcher may have acquired the shared lock after our probe.
                if "already active" not in str(error):
                    raise


def run():
    queue=base.curve.read(ROOT / CONFIG);validate(queue)
    with queue_lock():
        definition=frozen_definition(queue);path=ROOT / PROTOCOL
        if path.exists() and base.curve.read(path)!=definition:
            raise ValueError("Registered queue/config sources changed; preserve this queue and use a new definition")
        write_json(path,definition)
        state={"status":"starting","queue_pid":os.getpid(),"started_utc":base.now(),"pending":queue["followups"],
               "additional_fractions":queue["additional_fractions"],"official_test_evaluated":False}
        write_json(ROOT / STATUS,state)
        try:
            wait_for_main(queue,state)
            for name,digest in definition["source_hashes"].items():
                if file_sha256(ROOT / name)!=digest:
                    raise ValueError(f"A registered queue input changed while waiting: {name}")
            state.update(status="running_followups",updated_utc=base.now());write_json(ROOT / STATUS,state)
            snapshot=base.snapshot

            def followup_snapshot(config_path):
                manifest=snapshot(config_path)
                manifest["hashes"].update({name:file_sha256(ROOT / name) for name in (CONFIG,PROTOCOL,"scripts/run_curve_queue.py")})
                manifest["queue_definition_sha256"]=file_sha256(ROOT / PROTOCOL)
                return manifest

            # Reuse the existing runner without editing any of its hash-frozen files.
            with patch.object(base,"CONFIGS",queue["followups"]),patch.object(base,"STATUS",ROOT / "results/compression_curve_refinement_runner.json"),patch.object(base,"snapshot",followup_snapshot):
                base.run()
            state.update(status="completed",pending=[],updated_utc=base.now());write_json(ROOT / STATUS,state)
        except Exception as error:
            state.update(status="failed",error=str(error),updated_utc=base.now());write_json(ROOT / STATUS,state)
            raise


if __name__=="__main__":
    if "--check-only" in sys.argv:
        queue=base.curve.read(ROOT / CONFIG);validate(queue)
        print({"status":"ready","after":queue["main_configs"],"followups":queue["followups"],"fractions":queue["additional_fractions"]})
    else:
        run()
