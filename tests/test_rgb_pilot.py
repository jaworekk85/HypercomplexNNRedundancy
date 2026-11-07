import numpy as np
import pytest

from scripts.runtime import tf
from src.data.rgb import encode_pixels, fit_channel_statistics, normalize_rgb, stratified_indices
from src.models.rgb_baselines import build_rgb_model, parameter_budgets
from src.activations.diagnostics import activation_diagnostics, effective_rank, scalar_pearson


def test_splits_are_disjoint_balanced_and_repeatable():
    labels = np.repeat(np.arange(10), 20)
    sizes = {"train": 50, "validation": 30, "calibration": 20}
    first = stratified_indices(labels, sizes, 471)
    second = stratified_indices(labels, sizes, 471)
    all_indices = np.concatenate(list(first.values()))
    assert len(np.unique(all_indices)) == len(all_indices)
    for name, indices in first.items():
        np.testing.assert_array_equal(indices, second[name])
        np.testing.assert_array_equal(np.bincount(labels[indices]), np.full(10, sizes[name] // 10))
    with pytest.raises(ValueError):
        stratified_indices(labels, {"train": 201}, 471)
    with pytest.raises(ValueError):
        stratified_indices(labels, {"train": 210}, 471)


def test_rgb_pixel_and_channel_order_is_explicit_and_lossless():
    rgb = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype="float32")
    argb = encode_pixels(rgb, "ARGB")
    abgr = encode_pixels(rgb, "ABGR")
    np.testing.assert_array_equal(argb[0, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(abgr[0, 0], [0, 3, 2, 1])
    np.testing.assert_array_equal(argb[0, 2], [0, 7, 8, 9])
    np.testing.assert_array_equal(argb[..., 1:].reshape(rgb.shape), rgb)
    np.testing.assert_array_equal(abgr[..., 1:][..., ::-1].reshape(rgb.shape), rgb)
    np.testing.assert_array_equal(encode_pixels(rgb, "RGB"), rgb.reshape(1, 12))
    assert np.all(argb[..., 0] == 0)
    assert np.all(abgr[..., 0] == 0)


def test_statistics_fit_training_only_and_are_not_refit_on_held_out_images():
    training = np.array([[[[0, 10, 20]]], [[[100, 110, 120]]]], dtype="uint8")
    held_out = np.array([[[[200, 210, 220]]]], dtype="uint8")
    statistics = fit_channel_statistics(training)
    normalized = normalize_rgb(training, statistics)
    np.testing.assert_allclose(normalized.mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(normalized.std(axis=0), 1, atol=1e-6)
    np.testing.assert_allclose(normalize_rgb(held_out, statistics), 3, atol=1e-6)
    assert statistics["fit_samples"] == 2
    with pytest.raises(ValueError, match="zero variance"):
        fit_channel_statistics(np.zeros((2, 1, 1, 3), dtype="uint8"))


def test_parameter_match_has_no_zero_input_columns_or_dummy_weights():
    budgets = parameter_budgets(1024, 32, 10)
    assert budgets["quaternion_real_parameters"] == 132490
    assert budgets["real_parameters"] == 132579
    assert budgets["real_hidden_units"] == 43
    assert abs(budgets["real_minus_quaternion_fraction"]) < 0.001


@pytest.mark.parametrize("encoding", ["RGB", "ARGB", "ABGR"])
def test_real_and_padded_quaternion_models_train_every_kernel_component(encoding):
    config = {"image_shape": [2, 2, 3], "classes": 3, "quaternion_hidden_units": 3, "activation": "relu"}
    model, collector, budgets = build_rgb_model(config, encoding, 2026)
    rgb = np.random.default_rng(14).normal(size=(16, 2, 2, 3)).astype("float32")
    inputs = encode_pixels(rgb, encoding)
    labels = tf.constant(np.arange(16) % 3)
    before = [v.numpy().copy() for v in model.trainable_variables]
    with tf.GradientTape() as tape:
        logits = model(inputs)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
    gradients = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
    tf.keras.optimizers.Adam(0.001).apply_gradients(zip(gradients, model.trainable_variables))
    for old, variable in zip(before, model.trainable_variables):
        assert np.any(old != variable.numpy())
    if encoding != "RGB":
        assert collector(inputs).shape == (16, 3, 4)
        component_updates = np.linalg.norm((model.trainable_variables[0].numpy() - before[0]).reshape(-1, 4), axis=0)
        assert (component_updates > 0).all()  # Also with an identically zero real input slot.
    else:
        assert model.input_shape == (None, 12)


def test_scalar_metric_and_rank_handle_constant_units_without_mutating_inputs():
    values = np.array([[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype="float64")
    original = values.copy()
    matrix, valid = scalar_pearson(values)
    np.testing.assert_array_equal(valid, [True, False, False])
    assert np.isnan(matrix[1:]).all()
    assert effective_rank(values)["effective_rank"] == pytest.approx(1)
    np.testing.assert_array_equal(values, original)
    report, _ = activation_diagnostics(np.zeros((8, 2, 4)))
    assert report["rank"]["effective_rank"] == 0
    assert report["scalar"]["pairs"]["valid_pairs"] == 0
    assert report["grouped"]["constant_units"] == 2


def test_equal_orthogonal_covariance_directions_have_rank_two():
    values = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype="float64")
    assert effective_rank(values)["effective_rank"] == pytest.approx(2)


def test_rgb_training_and_selected_checkpoint_artifacts_offline(tmp_path):
    from scripts.rgb_pilot import train_one

    config = {"image_shape": [2, 2, 3], "classes": 3, "quaternion_hidden_units": 3,
              "activation": "relu", "epochs": 2, "learning_rate": 0.001, "batch_size": 8}
    rng = np.random.default_rng(11)
    rgb = {split: rng.normal(size=(size, 2, 2, 3)).astype("float32")
           for split, size in (("train", 24), ("validation", 8), ("calibration", 8))}
    labels = {split: (np.arange(len(images)) % 3).astype("int32") for split, images in rgb.items()}
    report = train_one(config, "ARGB", 2026, rgb, labels, tmp_path)
    assert report["status"] == "passed"
    assert all(report["checks"].values())
    assert not report["test_evaluated"]
    assert "test" not in report
    restored = tf.keras.models.load_model(tmp_path / "model.keras")
    collector = tf.keras.Model(restored.input, restored.get_layer("quaternion_hidden").output)
    with np.load(tmp_path / "calibration_activations.npz") as arrays:
        np.testing.assert_allclose(arrays["trained"], collector(encode_pixels(rgb["calibration"], "ARGB")), atol=1e-6)
