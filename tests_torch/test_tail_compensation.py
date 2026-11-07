from copy import deepcopy

import pytest
import torch

from src.compression import multilayer as old
from src.compression.structure_cost import make_model
from src.compression.tail_compensation import capture, logits_from_consumer, refine


@pytest.mark.parametrize("mode", ["local_extra", "tail_logits"])
def test_cached_tail_matches_full_forward_and_updates_only_four_Q_components(mode):
    torch.set_num_threads(2)
    torch.manual_seed(44)
    config = {"quaternion_widths": [2, 3, 4], "real_widths": [4, 6, 8], "dropout": [.1, .2, .3]}
    teacher = make_model(config, "cifar", "bn", "quaternion").double().eval()
    student = deepcopy(teacher)
    with torch.no_grad(): old.consumer(student, (2, 0)).kernel.mul_(.75)
    fit_images = torch.randn(11, 3, 16, 16, dtype=torch.float64)
    check_images = torch.randn(7, 3, 16, 16, dtype=torch.float64)
    fit, check = capture(teacher, student, fit_images, 4), capture(teacher, student, check_images, 4)
    assert fit["tail_replay_max_error"] < 1e-12
    assert check["tail_replay_max_error"] < 1e-12
    assert not fit["inputs"].requires_grad and not fit["logits"].requires_grad
    teacher_before = {k: v.clone() for k, v in teacher.state_dict().items()}
    result, audit = refine(student, fit, check, mode,
                           {"steps": 4, "batch_size": 4, "learning_rate": 1e-5, "relative_ridge": .001, "order_seed": 1})
    assert audit["first_update"]["passed"] and audit["only_target_kernel_changed"]
    assert all(torch.equal(v, teacher_before[k]) for k, v in teacher.state_dict().items())
    assert all(not m.training for m in result.modules())
    with torch.no_grad():
        z = old.consumer(result, (2, 0))(check["inputs"])
        torch.testing.assert_close(logits_from_consumer(result, z), result(check_images), atol=1e-12, rtol=1e-12)
