"""Resume training, complete the bounded study, verify it, and update status."""

from scripts.internal_replication import ROOT,read,run as compress
from scripts.train_fourblock import run as train
from scripts.verify_internal_replication import verify
from scripts.summarize_research import run as summarize


def run():
    study=read(ROOT / "configs/internal_replication.json")
    train(study)
    compress(study)
    verify(ROOT / study["output"])
    summarize()


if __name__=="__main__":
    run()
