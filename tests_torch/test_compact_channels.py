import io

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(2)

from src.compression.quaternion_channels import compact_block, component_indices, removal_order
from src.models.cifar_cnn_torch import CifarCNN


@pytest.mark.parametrize("block_index", [0, 1, 2])
@pytest.mark.parametrize("pattern", ["drop_first", "alternating"])
def test_physical_removal_matches_mask_and_roundtrips(block_index, pattern):
    torch.manual_seed(17)
    model = CifarCNN(q_widths=(3, 4, 5)).eval()
    with torch.no_grad():
        # Nontrivial BN affine terms/buffers protect against incomplete slicing.
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.normal_()
                module.running_var.uniform_(0.2, 2)
                module.weight.uniform_(0.3, 2)
                module.bias.normal_()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    inputs = torch.randn(4, 3, 32, 32)
    units = model.blocks[block_index][3].out_channels
    zero = compact_block(model, block_index, list(range(units)))
    torch.testing.assert_close(zero(inputs), model(inputs), atol=0, rtol=0)
    kept = list(range(1, units)) if pattern == "drop_first" else list(range(0, units, 2))
    compact = compact_block(model, block_index, kept)
    omitted = component_indices(sorted(set(range(units)) - set(kept)), units)
    def hook(module, args, output):
        result = output.clone()
        result[:, omitted] = 0
        return result
    handle = model.blocks[block_index].register_forward_hook(hook)
    expected = model(inputs).detach()
    handle.remove()
    actual = compact(inputs).detach()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    old_count = sum(p.numel() for p in model.parameters())
    new_count = sum(p.numel() for p in compact.parameters())
    conv = model.blocks[block_index][3]
    consumer_cost = (4 * model.blocks[block_index + 1][0].out_channels * 9) if block_index < 2 else 40
    assert old_count - new_count == (units - len(kept)) * (4 * conv.in_channels * 9 + 8 + consumer_cost)
    serialized = io.BytesIO()
    torch.save(compact.state_dict(), serialized)
    serialized.seek(0)
    restored = compact_block(model, block_index, kept)
    restored.load_state_dict(torch.load(serialized, weights_only=True), strict=True)
    torch.testing.assert_close(restored(inputs), actual, atol=0, rtol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_policies_are_reproducible_and_use_whole_units():
    matrix = np.array([[1, .9, .1], [.9, 1, .2], [.1, .2, 1]])
    assert removal_order("activation_similarity", [2, 1, .5], matrix) == [1, 2, 0]
    assert removal_order("incoming_norm", [2, 1, .5], matrix) == [2, 1, 0]
    assert removal_order("random", [2, 1, .5], matrix, 471) == removal_order("random", [2, 1, .5], matrix, 471)
    for policy in ("random", "incoming_norm", "activation_similarity"):
        assert sorted(removal_order(policy, [2, 1, .5], matrix)) == [0, 1, 2]
