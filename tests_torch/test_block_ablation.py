import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(2)

from src.models.cifar_cnn_torch import CifarCNN
from src.evaluation.block_ablation import capture_blocks, classification_metrics, grouped_spatial, nearest_neighbor_scores, predict_from_block, spearman, unit_channels


@pytest.mark.parametrize("kind", ["quaternion", "real"])
def test_zero_ablation_and_hooked_ablation_match_and_preserve_state(kind):
    torch.manual_seed(18)
    model = CifarCNN(kind, (2, 3, 4), (4, 6, 8)).eval()
    inputs = torch.randn(5, 3, 32, 32)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    blocks = capture_blocks(model, inputs, batch_size=2)
    baseline = model(inputs).detach().numpy()
    for index, features in enumerate(blocks):
        np.testing.assert_allclose(predict_from_block(model, index, features, batch_size=2), baseline, atol=1e-6)
        channels = unit_channels(features.shape[1], 1, kind == "quaternion")
        def zero_hook(module, args, output):
            result = output.clone()
            result[:, channels] = 0
            return result
        handle = model.blocks[index].register_forward_hook(zero_hook)
        expected = model(inputs).detach().numpy()
        handle.remove()
        actual = predict_from_block(model, index, features, [1], batch_size=2)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
    for name, values in model.state_dict().items():
        torch.testing.assert_close(values, before[name], rtol=0, atol=0)


def test_grouping_preserves_each_spatial_component():
    values = torch.arange(2 * 12 * 2 * 2).reshape(2, 12, 2, 2).float()
    grouped = grouped_spatial(values, True)
    assert grouped.shape == (2, 4, 3, 4)
    for component in range(4):
        for unit in range(3):
            np.testing.assert_array_equal(grouped[:, :, unit, component], values[:, component * 3 + unit].reshape(2, 4))
    assert unit_channels(12, 1, True) == [1, 4, 7, 10]


def test_rank_correlation_and_undefined_nearest_neighbors():
    assert spearman([1, 2, 2, 4], [4, 3, 3, 1]) == pytest.approx(-1)
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    matrix = np.array([[1, 0.7, np.nan], [0.7, 1, np.nan], [np.nan, np.nan, np.nan]])
    assert nearest_neighbor_scores(matrix) == [0.7, 0.7, None]
    scores = classification_metrics([[2, 0], [0, 2]], [0, 1])
    assert scores["accuracy"] == 1
    assert scores["cross_entropy"] == pytest.approx(np.log1p(np.exp(-2)))
