import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(2)

import scripts.cifar_cnn as trainer
from scripts.prepare_eurosat import stratified_split
from src.compression.multisource import compact_real_block, merge_quaternion_multi, quaternion_maps
from src.models.eurosat_cnn import FourBandCNN, make_fourband


def test_fourband_models_use_full_quaternion_input_and_four_real_inputs():
    torch.manual_seed(0)
    quaternion, real = FourBandCNN("quaternion").eval(), FourBandCNN("real").eval()
    x = torch.randn(2, 4, 64, 64)
    assert quaternion(x).shape == real(x).shape == (2, 10)
    assert quaternion.blocks[0][0].in_channels == 1 and real.blocks[0][0].in_channels == 4
    assert sum(p.numel() for p in quaternion.parameters()) == 73898
    assert sum(p.numel() for p in real.parameters()) == 72954+16*9
    with pytest.raises(ValueError):
        quaternion(torch.randn(2, 3, 64, 64))
    # Changing only the NIR (real) component changes the quaternion output: nothing is padded with zeros.
    y = x.clone(); y[:, 0] += 1
    assert not torch.allclose(quaternion(x), quaternion(y))
    fit = np.random.default_rng(1).normal(size=(3, 256, 16, 4))
    compact = merge_quaternion_multi(quaternion, 1, quaternion_maps(fit, [0, 5], "quaternion_right", 1e-3))[0]
    assert compact(x).shape == (2, 10)
    assert compact_real_block(real, 1, list(range(24)))(x).shape == (2, 10)


def test_split_is_stratified_partition():
    labels = np.repeat(np.arange(10), [300, 300, 300, 250, 250, 200, 250, 300, 250, 300])
    split = stratified_split(labels, np.random.default_rng(0))
    joined = np.concatenate(list(split.values()))
    assert len(joined) == len(labels) == len(np.unique(joined))
    assert len(split["validation"]) == 1000 and len(split["calibration"]) == 1000 and len(split["test"]) == round(.2*len(labels))
    assert set(np.unique(labels[split["validation"]])) == set(range(10))


def test_frozen_trainer_runs_with_fourband_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "make_model", make_fourband)
    config = {"seed": 3, "quaternion_widths": [2, 2, 2], "real_widths": [4, 4, 4], "dropout": [0., 0., 0.], "epochs": 1, "batch_size": 8,
              "learning_rate": 1e-3, "minimum_learning_rate": 1e-5, "weight_decay": 0., "mixed_precision": False}
    g = torch.Generator().manual_seed(0)
    tensors = {k: torch.randint(0, 3000, (n, 4, 16, 16), dtype=torch.int16, generator=g) for k, n in (("train", 16), ("validation", 8), ("calibration", 8))}
    labels = {k: torch.randint(0, 10, (len(v),), generator=g) for k, v in tensors.items()}
    mean, std = torch.full((1, 4, 1, 1), 5.), torch.full((1, 4, 1, 1), 3.)
    for kind in ("quaternion", "real"):
        report = trainer.train_one(config, kind, tensors, labels, mean, std, tmp_path / kind, False)
        assert report["first_update"]["checks"]["all_variables_updated"] and not report["test_evaluated"]
        assert json.loads((tmp_path / kind / "metrics.json").read_text())["kind"] == kind
