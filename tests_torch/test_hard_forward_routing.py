from copy import deepcopy
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(2)

from scripts import learned_grouping as pilot
from scripts.hard_forward_grouping import train
from src.cnn_training import first_update_audit, seed_everything
from src.models.channel_routing import log_sinkhorn
from src.models.hard_forward_routing import HardForwardChannelPermutation, make_model


def config():
    return {"seed": 2026, "quaternion_widths": [2, 3, 4], "dropout": [.1, .2, .3],
            "initial_strength": 4., "temperature_start": 1., "temperature_end": .25, "sinkhorn_iterations": 20,
            "learning_rate": .001, "router_learning_rate": .01, "weight_decay": .0001,
            "epochs": 3, "soft_epochs": 2, "minimum_learning_rate": .00001,
            "mixed_precision": False, "batch_size": 4}


def test_hard_forward_and_surrogate_vector_jacobian_are_correct():
    router = HardForwardChannelPermutation(4, 9, initial_strength=.2).double()
    inputs = torch.randn(3, 4, 2, 2, dtype=torch.float64, requires_grad=True)
    vector = torch.randn_like(inputs)
    outputs = router(inputs)
    assert torch.equal(outputs, inputs[:, router.indices])
    grad_logits, grad_inputs = torch.autograd.grad((outputs*vector).sum(), (router.logits, inputs))
    proxy = router.logits.detach().clone().requires_grad_()
    soft = torch.einsum("ij,njhw->nihw", log_sinkhorn(proxy, router.temperature, router.iterations), inputs.detach())
    expected = torch.autograd.grad((soft*vector).sum(), proxy)[0]
    torch.testing.assert_close(grad_logits, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad_inputs, vector[:, router.indices.argsort()], rtol=0, atol=0)


def test_discrete_router_learns_known_permutation_and_freezes_without_a_gap():
    generator = torch.Generator().manual_seed(3)
    router = HardForwardChannelPermutation(4, 9, initial_strength=.2).double()
    inputs = torch.randn(12, 4, 3, 3, generator=generator, dtype=torch.float64)
    target_indices = torch.tensor([2, 0, 3, 1])
    target = inputs[:, target_indices]
    optimizer = torch.optim.Adam(router.parameters(), lr=.1)
    for _ in range(120):
        optimizer.zero_grad()
        result = router(inputs)
        assert torch.equal(result, inputs[:, router.indices])
        loss = (result-target).square().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        before = router(inputs)
    assert torch.equal(before, target)
    assert int(router.assignment_switch_events) > 0
    router.harden()
    assert torch.equal(router(inputs), before)
    assert torch.equal(router.export()(inputs), before)
    assert not router.logits.requires_grad
    restored = HardForwardChannelPermutation(4, 9, initial_strength=.2).double()
    restored.load_state_dict(router.state_dict())
    assert not restored.logits.requires_grad
    assert torch.equal(restored(inputs), before)


def test_paired_cnn_initialization_forward_audit_and_export():
    settings = config()
    seed_everything(settings["seed"])
    model = make_model(settings, "learned")
    seed_everything(settings["seed"])
    control = make_model(settings, "random")
    values = torch.randn(8, 3, 32, 32)
    model.eval()
    control.eval()
    assert torch.equal(model(values), control(values))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    audit = first_update_audit(model, values, torch.arange(8)%10, optimizer)
    assert all(audit["checks"].values())
    assert next(row for row in audit["parameters"] if row["name"] == "router.logits")["max_abs_update"] > 0
    model.eval()
    before = model(values)
    model.router.harden()
    exported = model.export_hard()
    assert torch.equal(before, model(values))
    assert torch.equal(before, exported(values))
    assert sum(p.numel() for p in model.parameters())-sum(p.numel() for p in exported.parameters()) == 64


@pytest.mark.parametrize("interrupt_epoch", [1, 2, 3])
def test_discrete_training_resume_matches_uninterrupted_including_freeze(tmp_path, interrupt_epoch):
    settings = config()
    images = torch.randint(256, (12, 3, 32, 32), generator=torch.Generator().manual_seed(17), dtype=torch.uint8)
    tensors = {name: images for name in ("train", "validation", "calibration")}
    labels = {name: torch.arange(12)%10 for name in tensors}
    mean, std = torch.full((1, 3, 1, 1), .5), torch.full((1, 3, 1, 1), .2)
    expected = train(settings, tensors, labels, mean, std, tmp_path / "reference", {})
    original_save = pilot.save_last

    def interrupt(path, state):
        original_save(path, state)
        if path.name == "last.pt" and state["epoch"] == interrupt_epoch:
            raise RuntimeError("injected interruption")

    with patch.object(pilot, "save_last", interrupt), pytest.raises(RuntimeError, match="injected"):
        train(settings, tensors, labels, mean, std, tmp_path / "resumed", {})
    actual = train(settings, tensors, labels, mean, std, tmp_path / "resumed", {})
    a = torch.load(tmp_path / "reference" / "last.pt", weights_only=True)
    b = torch.load(tmp_path / "resumed" / "last.pt", weights_only=True)
    assert all(torch.equal(a["model"][name], b["model"][name]) for name in a["model"])
    assert a["assignment_trace"] == b["assignment_trace"]
    assert actual["selected_validation"] == expected["selected_validation"]
    assert actual["hardening"]["max_abs_logit_shift"] == 0
    assert actual["optimizer_updates_attempted"] == 9
    before = torch.load(tmp_path / "resumed" / "freeze_before.pt", weights_only=True)
    after = torch.load(tmp_path / "resumed" / "freeze_after.pt", weights_only=True)
    assert not bool(before["model"]["router.hard_enabled"])
    assert bool(after["model"]["router.hard_enabled"])
    assert all(torch.equal(before["model"][name], after["model"][name]) for name in before["model"] if name != "router.hard_enabled")
