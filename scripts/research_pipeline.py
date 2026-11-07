"""Resume the authorized three-seed study and conditional physical compression."""

import json
from pathlib import Path

from scripts.cifar_cnn import source_hashes
from scripts.cnn_multiseed import run as run_training
from scripts.cnn_compression import run as run_compression
from scripts.verify_cnn_compression import verify

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    training_folder = ROOT / "results" / "cnn_multiseed"
    completed = training_folder / "summary.json"
    if not completed.exists():
        run_training(json.loads((ROOT / "configs" / "cnn_multiseed.json").read_text(encoding="utf-8")), training_folder)
    else:
        summary = json.loads(completed.read_text(encoding="utf-8"))
        if summary["status"] != "completed" or summary["provenance"]["source_hashes"] != source_hashes():
            raise ValueError("Incomplete or changed frozen training study")
    compression_folder = ROOT / "results" / "cnn_compression"
    run_compression(json.loads((ROOT / "configs" / "cnn_compression.json").read_text(encoding="utf-8")), compression_folder)
    summary = json.loads((compression_folder / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] == "completed":
        verify(compression_folder)
        from scripts.summarize_research import run as summarize
        summarize()
    else:
        status_path = ROOT / "docs" / "RESEARCH_STATUS.md"
        status_path.write_text("# Three-seed diagnostic gate not confirmed\n\nThe frozen study completed. See results/cnn_multiseed/REPORT.md for baseline quality and results/cnn_compression/block_analysis/REPORT.md for every measured diagnostic.\n\n## Measured result\n\nThe selected-block gate did not pass in every seed. The planned compression stage stopped without compact-model test inference. The negative finding is preserved; no new layer, threshold or score was chosen to force H1-H3.\n\n## Interpretation\n\nInfrastructure remains verified. Gate B's scientific claim and Gates C-D remain open. This is not a general proof that quaternion networks lack redundancy.\n", encoding="utf-8")
