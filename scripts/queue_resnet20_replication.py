"""Wait for both baseline seeds and run the frozen compression replication once."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.cifar_cnn import write_json
from scripts.paper_a_controls import now
from scripts.run_activation_queue import busy, lock

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "results/resnet20_replication_baselines"
OUTPUT = ROOT / "results/resnet20_replication_queue"
REPLICATION = ROOT / "results/resnet20_key_replication"


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return None


def baseline_state():
    states = [read_json(BASELINES / f"seed_{seed}" / "state.json") for seed in (2027, 2028)]
    if any(state and state.get("status") == "failed" for state in states):
        return "baseline_failed"
    if any((BASELINES / f"seed_{seed}" / "STOP").exists() for seed in (2027, 2028)):
        return "baseline_paused"
    marker = read_json(BASELINES / "completion.json")
    if (marker and marker.get("status") == "completed" and marker.get("seeds") == [2027, 2028]
            and all(state and state.get("status") == "completed" for state in states)):
        return "completed"
    if any(state and state.get("status") == "paused" for state in states):
        return "baseline_paused"
    return "waiting"


def update(status, **extra):
    write_json(OUTPUT / "state.json", {"status": status, "pid": os.getpid(), "updated_utc": now(), **extra})


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with lock("resnet20_replication_queue.lock"):
        while True:
            if (OUTPUT / "STOP").exists():
                update("paused")
                return
            state = baseline_state()
            if state == "baseline_failed":
                update("blocked", reason="Baseline training failed; compression was not started.")
                return
            if state == "completed" and not busy("compression_curve_runner.lock"):
                break
            update("waiting_for_baselines", baseline_state=state)
            time.sleep(30)
        if (OUTPUT / "STOP").exists():
            update("paused")
            return
        existing = read_json(REPLICATION / "summary.json")
        if existing and existing.get("status") == "completed":
            update("completed", reason="Replication was already completed.")
            return
        update("running_replication")
        result = subprocess.run([sys.executable, "-B", "-u", "-m", "scripts.run_resnet20_key_replication", "--run"],
                                cwd=ROOT, check=False)
        summary = read_json(REPLICATION / "summary.json")
        if result.returncode != 0:
            update("failed", exit_code=result.returncode)
        elif summary and summary.get("status") == "completed":
            update("completed", exit_code=0)
        else:
            update("paused", reason="Replication stopped before completing; inspect seed state and STOP files.")


if __name__ == "__main__":
    main()
