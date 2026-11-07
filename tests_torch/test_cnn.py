import io
import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(2)
from torch.nn import functional as F

from src.models.quaternion_conv_torch import QuaternionConv2d, hamilton_kernel
from src.models.cifar_cnn_torch import CifarCNN
from src.cnn_training import first_update_audit, prepare_batch, seed_everything


def reference_product(a, b):
    return np.r_[a[0] * b[0] - a[1:] @ b[1:], a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:])]


def test_all_sixteen_hamilton_basis_products():
    layer = QuaternionConv2d(1, 1, 1, padding=0)
    basis = torch.eye(4).reshape(4, 4, 1, 1)
    for right in np.eye(4):
        with torch.no_grad():
            layer.kernel.copy_(torch.tensor(right, dtype=torch.float32).reshape(4, 1, 1, 1, 1))
        result = layer(basis).detach().numpy()[:, :, 0, 0]
        np.testing.assert_array_equal(result, np.array([reference_product(left, right) for left in np.eye(4)]))


def test_multichannel_spatial_convolution_against_numpy():
    rng = np.random.default_rng(15)
    x = rng.normal(size=(1, 4, 2, 4, 5)).astype("float64")
    kernel = rng.normal(size=(4, 3, 2, 3, 3))
    layer = QuaternionConv2d(2, 3).double()
    with torch.no_grad():
        layer.kernel.copy_(torch.tensor(kernel))
    result = layer(torch.tensor(x.reshape(1, 8, 4, 5))).detach().numpy().reshape(1, 4, 3, 4, 5)
    padded = np.pad(x, ((0, 0), (0, 0), (0, 0), (1, 1), (1, 1)))
    expected = np.zeros_like(result)
    for out in range(3):
        for row in range(4):
            for col in range(5):
                for incoming in range(2):
                    for dy in range(3):
                        for dx in range(3):
                            expected[0, :, out, row, col] += reference_product(padded[0, :, incoming, row + dy, col + dx], kernel[:, out, incoming, dy, dx])
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_convolution_input_and_weight_gradients():
    torch.manual_seed(4)
    x = torch.randn(1, 8, 2, 2, dtype=torch.double, requires_grad=True)
    kernel = torch.randn(4, 2, 2, 1, 1, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: F.conv2d(a, hamilton_kernel(b)), (x, kernel))


@pytest.mark.parametrize("kind", ["quaternion", "real"])
def test_cnn_tracking_updates_shapes_and_serialization(kind):
    seed_everything(2026)
    torch.set_num_threads(2)
    model = CifarCNN(kind, q_widths=(2, 3, 4), real_widths=(4, 6, 8))
    x = torch.randn(8, 3, 32, 32)
    y = torch.arange(8) % 10
    audit = first_update_audit(model, x, y, torch.optim.AdamW(model.parameters(), lr=0.001))
    assert all(audit["checks"].values())
    if kind == "quaternion":
        assert len(audit["quaternion_layers"]) == 6
    model.eval()
    assert model(x).shape == (8, 10)
    assert model.forward_features(x).shape == (8, 16 if kind == "quaternion" else 8, 4, 4)
    data = io.BytesIO()
    torch.save(model.state_dict(), data)
    data.seek(0)
    restored = CifarCNN(kind, q_widths=(2, 3, 4), real_widths=(4, 6, 8)).eval()
    restored.load_state_dict(torch.load(data, weights_only=True))
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)


def test_augmentation_preserves_rgb_pixel_channels_and_is_repeatable():
    images = torch.zeros(4, 3, 32, 32, dtype=torch.uint8)
    images[:, 0], images[:, 1], images[:, 2] = 30, 60, 90
    mean, std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
    a = prepare_batch(images, mean, std, True, torch.Generator().manual_seed(19))
    b = prepare_batch(images, mean, std, True, torch.Generator().manual_seed(19))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(a[:, 1], 2 * a[:, 0])
    torch.testing.assert_close(a[:, 2], 3 * a[:, 0])
    assert a.shape == images.shape


def test_compact_parameters_do_not_count_the_derived_expanded_kernel():
    layer = QuaternionConv2d(2, 3)
    assert sum(p.numel() for p in layer.parameters()) == 4 * 2 * 3 * 3 * 3
    assert hamilton_kernel(layer.kernel).numel() == 4 * layer.kernel.numel()
    assert sum(p.numel() for p in CifarCNN("quaternion").parameters()) == 73898
    assert sum(p.numel() for p in CifarCNN("real").parameters()) == 72954


@pytest.mark.parametrize("kind", ["quaternion", "real"])
def test_spatial_activation_collector_preserves_channel_components(kind):
    from scripts.cifar_cnn import collect

    model = CifarCNN(kind, (2, 3, 4), (4, 6, 8)).eval()
    raw = torch.randint(0, 256, (3, 3, 32, 32), dtype=torch.uint8)
    mean, std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
    actual = collect(model, raw, mean, std)
    with torch.no_grad():
        maps = model.forward_features(raw.float() / 255).numpy()
    if kind == "quaternion":
        assert actual.shape == (3, 16, 4, 4)
        for component in range(4):
            for unit in range(4):
                np.testing.assert_array_equal(actual[:, :, unit, component], maps[:, component * 4 + unit].reshape(3, 16))
    else:
        assert actual.shape == (3, 16, 8)
        np.testing.assert_array_equal(actual, maps.transpose(0, 2, 3, 1).reshape(3, 16, 8))


def test_interrupted_training_resumes_exactly_on_cpu(tmp_path, monkeypatch):
    from scripts import cifar_cnn

    config = {"seed": 2026, "quaternion_widths": [2, 3, 4], "real_widths": [4, 6, 8],
              "dropout": [0.1, 0.2, 0.3], "epochs": 2, "batch_size": 8,
              "learning_rate": 0.001, "minimum_learning_rate": 0.00001,
              "weight_decay": 0.0001, "mixed_precision": False}
    rng = np.random.default_rng(12)
    images = {name: torch.tensor(rng.integers(0, 256, (count, 3, 32, 32), dtype=np.uint8))
              for name, count in (("train", 16), ("validation", 8), ("calibration", 8))}
    labels = {name: torch.arange(len(values)) % 10 for name, values in images.items()}
    mean, std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
    full = cifar_cnn.train_one(config, "quaternion", images, labels, mean, std, tmp_path / "full", False)
    evaluator = cifar_cnn.evaluate
    calls = 0
    def interrupt_during_second_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:  # Initial validation, epoch one, then epoch two before saving.
            raise RuntimeError("simulated training interruption")
        return evaluator(*args, **kwargs)
    monkeypatch.setattr(cifar_cnn, "evaluate", interrupt_during_second_epoch)
    with pytest.raises(RuntimeError, match="simulated training interruption"):
        cifar_cnn.train_one(config, "quaternion", images, labels, mean, std, tmp_path / "resumed", False)
    monkeypatch.setattr(cifar_cnn, "evaluate", evaluator)
    resumed = cifar_cnn.train_one(config, "quaternion", images, labels, mean, std, tmp_path / "resumed", True)
    assert full["optimizer_updates_attempted"] == resumed["optimizer_updates_attempted"] == 4
    assert full["best_epoch"] == resumed["best_epoch"]
    full_weights = torch.load(tmp_path / "full" / "model.pt", weights_only=True)["model"]
    resumed_weights = torch.load(tmp_path / "resumed" / "model.pt", weights_only=True)["model"]
    for key in full_weights:
        torch.testing.assert_close(full_weights[key], resumed_weights[key], rtol=0, atol=0)
