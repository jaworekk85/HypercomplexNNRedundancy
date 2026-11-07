from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(2)

from scripts.finetune_compression import same_maps, structure
from src.compression.multisource import merge_quaternion_multi, quaternion_maps
from src.finetuning import finetune
from src.models.cifar_cnn_torch import CifarCNN

CONFIG = {"epochs": 2, "batch_size": 8, "learning_rate": 1e-3, "minimum_learning_rate": 1e-5, "weight_decay": 1e-4,
          "mixed_precision": False, "order_seed_offset": 80000, "augment_seed_offset": 90000}


def data():
    generator = torch.Generator().manual_seed(0)
    tensors = {"train": torch.randint(0, 256, (20, 3, 32, 32), dtype=torch.uint8, generator=generator),
               "validation": torch.randint(0, 256, (6, 3, 32, 32), dtype=torch.uint8, generator=generator)}
    labels = {"train": torch.randint(0, 10, (20,), generator=generator), "validation": torch.randint(0, 10, (6,), generator=generator)}
    return tensors, labels, torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)


def test_finetuning_keeps_quaternion_structure_updates_weights_and_is_deterministic():
    torch.manual_seed(1)
    model = CifarCNN(q_widths=(2, 5, 3)).eval()
    fit = np.random.default_rng(2).normal(size=(6, 16, 5, 4))
    compact = merge_quaternion_multi(model, 1, quaternion_maps(fit, [1, 3], "quaternion_right", 1e-3))[0]
    tensors, labels, mean, std = data()
    before = structure(compact)
    first, history = finetune(deepcopy(compact), tensors, labels, mean, std, CONFIG, 2026)
    second, _ = finetune(deepcopy(compact), tensors, labels, mean, std, CONFIG, 2026)
    assert structure(first) == before and before["quaternion_layers"] == 6
    assert len(history) == 2 and history[-1]["learning_rate_end"] == pytest.approx(1e-5)
    assert all(torch.equal(a, b) for a, b in zip(first.state_dict().values(), second.state_dict().values()))
    assert any(not torch.equal(a, b) for a, b in zip(first.state_dict().values(), compact.state_dict().values()))


def test_frozen_map_comparison_tolerates_only_rounding():
    left = [{"victim": 1, "matrices": [[0.5, 1.0]], "mode": "quaternion_right"}]
    assert same_maps(left, [{"victim": 1, "matrices": [[0.5 + 1e-13, 1.0]], "mode": "quaternion_right"}])
    assert not same_maps(left, [{"victim": 1, "matrices": [[0.6, 1.0]], "mode": "quaternion_right"}])
    assert not same_maps(left, [{"victim": 2, "matrices": [[0.5, 1.0]], "mode": "quaternion_right"}])
