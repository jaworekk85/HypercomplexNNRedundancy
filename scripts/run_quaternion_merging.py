"""Resume the prespecified diagnostic, conditional compaction and verification."""

from scripts.quaternion_merge_diagnostic import ROOT, read, run as diagnose
from scripts.quaternion_merge_compression import run as compress
from scripts.verify_quaternion_merging import verify
from scripts.summarize_research import run as summarize


if __name__ == "__main__":
    config = read(ROOT / "configs/quaternion_merging.json")
    diagnostic = diagnose(config)
    if diagnostic["gate"]["supports_physical_comparison"]:
        compress(config)
    verify(ROOT / config["output"])
    summarize()
