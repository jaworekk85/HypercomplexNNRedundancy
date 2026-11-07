from copy import deepcopy

import numpy as np
import pytest
import torch

from src.cnn_training import first_update_audit, seed_everything
from src.compression import multilayer as old
from src.compression.internal_merging import expand_right
from src.compression.structure_cost import compact, describe, make_model, release_consumers

torch.set_num_threads(2)
CONFIG = {"quaternion_widths": [3, 4, 5], "real_widths": [6, 8, 10], "dropout": [0., 0., 0.]}


@pytest.mark.parametrize("norm", ["bn", "none"])
@pytest.mark.parametrize("dataset", ["cifar", "eurosat"])
def test_full_map_surgery_equals_functional_substitution_across_all_six_sites(norm, dataset):
    seed_everything(7)
    model = make_model(CONFIG, dataset, norm, "quaternion").double().eval()
    x = torch.randn(3, 3 if dataset == "cifar" else 4, 16, 16, dtype=torch.float64)
    np_rng = np.random.default_rng(51)
    for site in old.sites(model):
        width = describe(model)["output_group_widths"][2*site[0]+site[1]]
        victims, kept = [1], [i for i in range(width) if i != 1]
        maps = [{"victim": 1, "kept": kept, "mode": "full_real", "matrices": np_rng.normal(size=(len(kept), 4, 4)).tolist()}]
        expected = old.substituted_logits(model, site, maps, x)
        model = compact(model, site, victims, maps, "free")
        torch.testing.assert_close(model(x), expected, atol=1e-10, rtol=1e-10)
    assert describe(model)["dense_real_convolutions"] == 5


@pytest.mark.parametrize("norm", ["bn", "none"])
def test_hamilton_embedding_is_same_function_and_compaction_matches_existing_bn_code(norm):
    seed_everything(11)
    q = make_model(CONFIG, "cifar", norm, "quaternion").double().eval()
    free = release_consumers(q)
    x = torch.randn(4, 3, 16, 16, dtype=torch.float64)
    torch.testing.assert_close(q(x), free(x), atol=0, rtol=0)
    np_rng = np.random.default_rng(16)
    for site in old.sites(q):
        width = old.units(q, site)
        kept = [i for i in range(width) if i != 1]
        maps = [{"victim": 1, "kept": kept, "mode": "quaternion_right", "matrices": expand_right(np_rng.normal(size=(len(kept), 4))).tolist()}]
        expected = old.substituted_logits(q, site, maps, x)
        if norm == "bn":
            reference, _, _ = old.compress_site(q, site, [1], maps)
        q = compact(q, site, [1], maps, "quaternion")
        free = compact(free, site, [1], maps, "free")
        torch.testing.assert_close(q(x), expected, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(q(x), free(x), atol=1e-10, rtol=1e-10)
        if norm == "bn": torch.testing.assert_close(q(x), reference(x), atol=1e-10, rtol=1e-10)


def test_unrestricted_parameters_are_tracked_updated_and_checkpoint_reloads():
    seed_everything(31)
    q = make_model(CONFIG, "cifar", "none", "quaternion")
    model = release_consumers(q)
    assert describe(model)["real_parameters"] > describe(q)["real_parameters"]
    images, labels = torch.randn(8, 3, 16, 16), torch.arange(8)
    audit = first_update_audit(model, images, labels, torch.optim.AdamW(model.parameters(), lr=1e-4))
    assert all(audit["checks"].values())
    restored = make_model(CONFIG, "cifar", "none", "free")
    restored.load_state_dict(deepcopy(model.state_dict()))
    torch.testing.assert_close(model.eval()(images), restored.eval()(images), atol=0, rtol=0)
