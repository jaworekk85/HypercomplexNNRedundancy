"""Request a cooperative pause of the PHM baseline at a UTC deadline."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "results/resnet20_qstem_phm4_baseline"
OUTPUT = ROOT / "results/qstem_phm4_timed_pause"


def status(name, deadline):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "state.json"
    temporary = OUTPUT / "state.tmp"
    temporary.write_text(json.dumps({"status": name, "pid": os.getpid(),
                                     "deadline_utc": deadline.isoformat(),
                                     "updated_utc": datetime.now(timezone.utc).isoformat()}, indent=2),
                         encoding="utf-8")
    temporary.replace(path)


def request_pause(training):
    training.mkdir(parents=True, exist_ok=True)
    (training / "STOP").touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--at", required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.at)
    if deadline.tzinfo is None:
        raise ValueError("An explicit UTC offset is required")
    status("waiting", deadline)
    while datetime.now(timezone.utc) < deadline:
        time.sleep(min(10, max(.1, (deadline-datetime.now(timezone.utc)).total_seconds())))
    request_pause(TRAINING)
    status("pause_requested", deadline)
    while True:
        try:
            current = json.loads((TRAINING / "state.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, PermissionError):
            current = {}
        if current.get("status") in ("paused", "completed", "failed"):
            status("training_stopped", deadline)
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
