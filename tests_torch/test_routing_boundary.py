from copy import deepcopy
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(2)

from scripts import learned_grouping as pilot
from scripts.replay_routing_boundary import capture_boundary


@pytest.mark.parametrize("interrupt", [False, True])
def test_boundary_capture_preserves_original_training_and_resumes(tmp_path, interrupt):
    config = {"seed": 2026, "quaternion_widths": [2, 3, 4], "dropout": [.1, .2, .3],
              "initial_strength": 4., "temperature_start": 1., "temperature_end": .25, "sinkhorn_iterations": 20,
              "learning_rate": .001, "router_learning_rate": .01, "weight_decay": .0001,
              "epochs": 3, "soft_epochs": 2, "minimum_learning_rate": .00001,
              "mixed_precision": False, "batch_size": 4}
    images = torch.randint(256, (12, 3, 32, 32), generator=torch.Generator().manual_seed(17), dtype=torch.uint8)
    tensors = {name: images for name in ("train", "validation", "calibration")}
    labels = {name: torch.arange(12) % 10 for name in tensors}
    mean, std = torch.full((1, 3, 1, 1), .5), torch.full((1, 3, 1, 1), .2)
    original_save, expected = pilot.save_last, {}

    def observe(path, state):
        if state["epoch"] == 2:
            expected.update(deepcopy(state))
        original_save(path, state)

    with patch.object(pilot, "save_last", observe):
        pilot.train_mode(config, "learned", tensors, labels, mean, std, tmp_path / "reference", {})
    output = tmp_path / "replay"

    def stop_early(path, state):
        original_save(path, state)
        if state["epoch"] == 1:
            raise RuntimeError("injected interruption")

    if interrupt:
        with patch.object(pilot, "save_last", stop_early), pytest.raises(RuntimeError, match="injected"):
            capture_boundary(config, tensors, labels, mean, std, output, {})
    capture_boundary(config, tensors, labels, mean, std, output, {})
    hard = torch.load(output / "boundary_hard.pt", weights_only=True)
    soft = torch.load(output / "boundary_soft.pt", weights_only=True)
    assert hard["epoch"] == soft["epoch"] == 2
    assert hard["updates"] == soft["updates"] == expected["updates"] == 6
    assert all(torch.equal(value, hard["model"][name]) for name, value in expected["model"].items())
    assert hard["hardening"] == expected["hardening"]
    for name in hard["model"]:
        if name not in ("router.indices", "router.hard_enabled"):
            assert torch.equal(hard["model"][name], soft["model"][name])
    assert not bool(soft["model"]["router.hard_enabled"])
    assert bool(hard["model"]["router.hard_enabled"])
    for name in ("torch_rng", "order_rng", "augment_rng"):
        assert torch.equal(hard[name], expected[name])
        assert torch.equal(soft[name], expected[name])
    assert soft["scheduler"] == hard["scheduler"] == expected["scheduler"]
    assert len(hard["optimizer"]["state"]) == len(expected["optimizer"]["state"])
    for key, values in expected["optimizer"]["state"].items():
        for name, value in values.items():
            assert torch.equal(value, hard["optimizer"]["state"][key][name])
    assert soft["history"][-1]["validation_accuracy"] == hard["hardening"]["soft_validation"]["accuracy"]
    # Re-entry at the completed boundary must not take another optimizer step.
    assert capture_boundary(config, tensors, labels, mean, std, output, {}) == hard["hardening"]
