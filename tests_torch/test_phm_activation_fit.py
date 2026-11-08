from copy import deepcopy

import torch

from src.compression.phm_activation_fit import effective, gradient_audit, make_initialization, normalized, objective
from src.compression.structure_catalogue import raw_moments
from src.models.phm_conv import PHMConv2d
from src.models.quaternion_conv_torch import QuaternionConv2d


def test_moment_objective_matches_outputs_and_q_embedding_updates_all_factors():
    torch.set_num_threads(2)
    torch.manual_seed(13)
    q = QuaternionConv2d(2, 3).double()
    x = torch.randn(5, 8, 8, 8, dtype=torch.float64)
    y = torch.randn(5, 12, 8, 8, dtype=torch.float64)
    moments = raw_moments(x, y)
    phm = make_initialization(q, PHMConv2d(8, 12), "from_q", 0)
    torch.testing.assert_close(effective(phm), effective(q), atol=1e-12, rtol=1e-12)
    pack, scale = normalized(moments)
    torch.testing.assert_close(objective(phm, pack, scale), (phm(x)-y).square().mean()/scale, atol=1e-12, rtol=1e-12)
    audit = gradient_audit(phm, moments)
    assert all(audit["updated"].values())
    a = make_initialization(q, phm, "random", 31)
    b = make_initialization(q, phm, "random", 31)
    torch.testing.assert_close(effective(a), effective(b), atol=0, rtol=0)
    assert sum(p.numel() for p in phm.parameters()) == sum(p.numel() for p in q.parameters()) + 64


def test_full_batch_fit_resume_replays_optimizer_and_best_checkpoint(tmp_path, monkeypatch):
    from scripts import phm_initialization_pilot as runner
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    torch.set_num_threads(2)
    torch.manual_seed(21)
    layer = PHMConv2d(4,8).double()
    x,y = torch.randn(4,4,8,8),torch.randn(4,8,8,8)
    moments = raw_moments(x,y)
    config = {"output":"study","learning_rate":.01,"steps":6,"checkpoint_interval":2}
    full = deepcopy(layer)
    a = runner.fit_one(config,tmp_path/"full",full,moments,"full")
    original_write = runner.write_json
    def stop_after_two(path,data):
        original_write(path,data)
        if data.get("step")==2:
            (tmp_path/"study"/"STOP").touch()
    monkeypatch.setattr(runner,"write_json",stop_after_two)
    interrupted = deepcopy(layer)
    assert runner.fit_one(config,tmp_path/"resume",interrupted,moments,"partial") is None
    (tmp_path/"study"/"STOP").unlink()
    monkeypatch.setattr(runner,"write_json",original_write)
    resumed = deepcopy(layer)
    b = runner.fit_one(config,tmp_path/"resume",resumed,moments,"resumed")
    assert a["best_fit_normalized_mse"]==b["best_fit_normalized_mse"]
    assert a["best_step"]==b["best_step"]
    for name,value in full.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[name],atol=0,rtol=0)
