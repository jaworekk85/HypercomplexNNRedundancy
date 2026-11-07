import csv

import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(2)

from scripts import learned_grouping as pilot


@pytest.mark.parametrize("interrupt_epoch", [1, 3])
def test_soft_and_final_hard_checkpoint_resume_match_uninterrupted(tmp_path, monkeypatch, interrupt_epoch):
    config = {"seed": 2026, "quaternion_widths": [2, 3, 4], "dropout": [.1, .2, .3],
              "initial_strength": 4., "temperature_start": 1., "temperature_end": .25, "sinkhorn_iterations": 20,
              "learning_rate": .001, "router_learning_rate": .01, "weight_decay": .0001,
              "epochs": 3, "soft_epochs": 2, "minimum_learning_rate": .00001,
              "mixed_precision": False, "batch_size": 4}
    generator = torch.Generator().manual_seed(17)
    images = torch.randint(256, (12, 3, 32, 32), generator=generator, dtype=torch.uint8)
    labels = torch.arange(12)%10
    tensors = {name:images for name in ("train", "validation", "calibration")}
    targets = {name:labels for name in tensors}
    mean, std = torch.full((1, 3, 1, 1), .5), torch.full((1, 3, 1, 1), .2)
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    hashes = {"synthetic_test": "unchanged"}
    expected = pilot.train_mode(config, "learned", tensors, targets, mean, std, uninterrupted, hashes)
    original_save = pilot.save_last
    def interrupt(path, state):
        original_save(path, state)
        if state["epoch"] == interrupt_epoch:
            raise RuntimeError("injected checkpoint interruption")
    monkeypatch.setattr(pilot, "save_last", interrupt)
    with pytest.raises(RuntimeError, match="injected checkpoint interruption"):
        pilot.train_mode(config, "learned", tensors, targets, mean, std, resumed, hashes)
    monkeypatch.setattr(pilot, "save_last", original_save)
    actual = pilot.train_mode(config, "learned", tensors, targets, mean, std, resumed, hashes)
    a = torch.load(uninterrupted / "model.pt", weights_only=True)["model"]
    b = torch.load(resumed / "model.pt", weights_only=True)["model"]
    assert all(torch.equal(a[key], b[key]) for key in a)
    assert actual["selected_validation"] == expected["selected_validation"]
    assert actual["optimizer_updates_attempted"] == expected["optimizer_updates_attempted"] == 9
    assert [int(row["epoch"]) for row in csv.DictReader((resumed / "training.csv").open())] == [1, 2, 3]
