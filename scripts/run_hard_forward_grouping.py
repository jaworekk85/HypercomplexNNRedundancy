"""Resume training, report, and verify the single prespecified pilot."""

from scripts.hard_forward_grouping import ROOT, read, run
from scripts.verify_hard_forward_grouping import verify
from scripts.summarize_research import run as summarize
from scripts.plot_hard_forward_groups import plot


if __name__ == "__main__":
    study = read(ROOT / "configs/hard_forward_grouping.json")
    run(study)
    verify(ROOT / study["output"])
    plot(ROOT / study["output"])
    summarize()
