"""Resume the bounded intermediate-layer study and verify saved artifacts."""

import torch

from scripts.internal_merging import ROOT, read, run
from scripts.verify_internal_merging import verify
from scripts.summarize_research import run as summarize
from scripts.audit_internal_projection import run as audit_projection


if __name__ == "__main__":
    config = read(ROOT / "configs/internal_merging.json")
    with torch.no_grad():
        run(config)
        verify(ROOT / config["output"])
    audit_projection(ROOT / config["output"])
    summarize()
