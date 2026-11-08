"""Start the existing combined hybrid/variable-n pilot after replication."""
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.cifar_cnn import write_json
from scripts.paper_a_controls import now
from scripts.queue_resnet20_replication import read_json
from scripts.run_activation_queue import busy, lock

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/resnet_structure_catalogue_queue"


def predecessor_state():
    queue = read_json(ROOT / "results/resnet20_replication_queue/state.json")
    if queue and queue.get("status") in {"failed", "blocked"}:
        return "failed"
    if not queue or queue.get("status") != "completed":
        return "waiting"
    config = read_json(ROOT / "configs/resnet_structure_catalogue.json")
    if not config:
        return "waiting"
    for path in config["required_completed_studies"]:
        summary = read_json(ROOT / path)
        if not summary or summary.get("status") != "completed":
            return "waiting"
    return "completed"


def update(status, **extra):
    write_json(OUTPUT / "state.json", {"status": status, "pid": os.getpid(), "updated_utc": now(), **extra})


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with lock("resnet_structure_catalogue_queue.lock"):
        while True:
            if (OUTPUT / "STOP").exists():
                update("paused")
                return
            predecessor = predecessor_state()
            if predecessor == "failed":
                update("blocked", reason="Replication queue failed or is blocked; pilot was not started.")
                return
            if predecessor == "completed" and not busy("compression_curve_runner.lock"):
                break
            update("waiting_for_replication")
            time.sleep(30)
        if (OUTPUT / "STOP").exists():
            update("paused")
            return
        pilot = ROOT / "results/resnet_structure_catalogue"
        summary = read_json(pilot / "summary.json")
        if summary and summary.get("status") == "completed":
            update("completed", reason="Combined pilot already completed.")
            return
        if (pilot / "STOP").exists():
            update("paused", reason="Pilot STOP exists; remove before resuming the queue.")
            return
        update("running_pilot")
        result = subprocess.run([sys.executable, "-B", "-u", "-m", "scripts.resnet_structure_catalogue", "--run"],
                                cwd=ROOT, check=False)
        summary = read_json(pilot / "summary.json")
        if result.returncode:
            update("failed", exit_code=result.returncode)
        elif summary and summary.get("status") == "completed":
            update("completed", exit_code=0)
        else:
            update("paused", reason="Pilot returned without completion; inspect pilot state and STOP.")


if __name__ == "__main__":
    main()
