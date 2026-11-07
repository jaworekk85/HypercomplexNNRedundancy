"""Resume training, verify exports, then execute the prespecified conditional stage."""

import json
from pathlib import Path
import subprocess
import sys

from scripts.learned_grouping import run as train
from scripts.verify_learned_grouping import verify as verify_training
from scripts.learned_grouping_compression import run as compress
from scripts.verify_routed_compression import verify as verify_compression

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    train(json.loads((ROOT / "configs" / "learned_grouping.json").read_text(encoding="utf-8")), ROOT / "results" / "learned_grouping_pilot")
    verify_training(ROOT / "results" / "learned_grouping_pilot")
    subprocess.run([sys.executable, "-m", "scripts.plot_learned_grouping"], cwd=ROOT, check=True)
    compress(json.loads((ROOT / "configs" / "learned_grouping_compression.json").read_text(encoding="utf-8")), ROOT / "results" / "learned_grouping_compression")
    verify_compression(ROOT / "results" / "learned_grouping_compression")
