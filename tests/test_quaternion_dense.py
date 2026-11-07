"""Core numerical invariants: arithmetic, tracking, differentiation and storage."""

from scripts.runtime import np, tf
from src.models.quaternion_dense import QuaternionDense

import pytest


def hamilton_reference(a, b):
    # Independent scalar/vector identity: (a0*b0-a.b, a0*b+b0*a+a cross b).
    return np.concatenate((
        [a[0] * b[0] - np.dot(a[1:], b[1:])],
        a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:]),
    ))


def test_all_basis_products_and_noncommutativity():
    basis = np.eye(4, dtype="float32")
    layer = QuaternionDense(1, use_bias=False)
    layer(basis[:, None, :])
    for right in basis:
        layer.kernel.assign(right[None, None, :])
        expected = np.array([hamilton_reference(left, right) for left in basis])
        np.testing.assert_array_equal(layer(basis[:, None, :]).numpy()[:, 0], expected)


@pytest.mark.parametrize("use_bias", [True, False])
def test_multiunit_sum_matches_numpy(use_bias):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(5, 2, 4)).astype("float32")
    w = rng.normal(size=(2, 3, 4)).astype("float32")
    b = rng.normal(size=(3, 4)).astype("float32") if use_bias else np.zeros((3, 4))
    layer = QuaternionDense(3, use_bias=use_bias)
    layer(x)
    layer.kernel.assign(w)
    if use_bias:
        layer.bias.assign(b)
    expected = np.array([
        [sum((hamilton_reference(sample[i], w[i, j]) for i in range(2))) + b[j]
         for j in range(3)] for sample in x
    ])
    np.testing.assert_allclose(layer(x), expected, atol=2e-6, rtol=2e-6)
    assert layer.count_params() == 24 + (12 if use_bias else 0)


def test_all_components_tracked_differentiable_and_updated_in_graph_mode():
    x = tf.random.normal((7, 2, 4))
    target = tf.random.normal((7, 3, 4))
    layer = QuaternionDense(3)
    layer(x)
    variables = layer.trainable_variables
    assert len(variables) == 2
    assert any(v is layer.kernel for v in variables)
    before = [v.numpy().copy() for v in variables]
    optimizer = tf.keras.optimizers.SGD(0.03)

    @tf.function
    def step(inputs, targets):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean((layer(inputs) - targets) ** 2)
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return gradients

    gradients = step(x, target)
    for v, old, gradient in zip(variables, before, gradients):
        assert gradient is not None
        assert np.isfinite(gradient.numpy()).all()
        assert (np.linalg.norm(gradient.numpy().reshape(-1, 4), axis=0) > 0).all()
        assert (np.linalg.norm((v.numpy() - old).reshape(-1, 4), axis=0) > 0).all()


def test_input_and_kernel_gradients_match_finite_differences():
    layer = QuaternionDense(2, use_bias=False, dtype="float64")
    x = tf.random.normal((2, 2, 4), dtype=tf.float64)
    layer(x)
    analytic, numeric = tf.test.compute_gradient(layer, [x], delta=1e-5)
    np.testing.assert_allclose(analytic[0], numeric[0], atol=1e-8, rtol=1e-8)
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(layer(x) ** 2)
    gradient = tape.gradient(loss, layer.kernel).numpy()
    original = layer.kernel.numpy().copy()
    numeric_kernel = np.zeros_like(original)
    delta = 1e-5
    for index in np.ndindex(original.shape):
        perturbed = original.copy()
        perturbed[index] += delta
        layer.kernel.assign(perturbed)
        plus = float(tf.reduce_sum(layer(x) ** 2))
        perturbed[index] -= 2 * delta
        layer.kernel.assign(perturbed)
        minus = float(tf.reduce_sum(layer(x) ** 2))
        numeric_kernel[index] = (plus - minus) / (2 * delta)
    layer.kernel.assign(original)
    np.testing.assert_allclose(gradient, numeric_kernel, atol=1e-8, rtol=1e-8)


def test_shape_activation_and_serialization(tmp_path):
    inputs = tf.keras.Input((2, 4))
    outputs = QuaternionDense(3, activation="tanh")(inputs)
    model = tf.keras.Model(inputs, outputs)
    x = tf.random.normal((5, 2, 4))
    expected = model(x).numpy()
    assert expected.shape == (5, 3, 4)
    assert np.max(np.abs(expected)) <= 1
    path = tmp_path / "quaternion.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path)
    np.testing.assert_array_equal(restored(x), expected)
    assert restored.count_params() == 36
    assert len(restored.trainable_variables) == 2


@pytest.mark.parametrize("units", [0, -1, 1.5, True])
def test_reject_invalid_width(units):
    with pytest.raises(ValueError, match="positive integer"):
        QuaternionDense(units)


def test_reject_ungrouped_input():
    with pytest.raises(ValueError):
        QuaternionDense(3)(tf.zeros((2, 8)))


def test_extra_leading_dimensions_preserve_independent_samples():
    x = tf.random.normal((2, 3, 2, 4))
    layer = QuaternionDense(5)
    actual = layer(x)
    assert actual.shape == (2, 3, 5, 4)
    flat = layer(tf.reshape(x, (6, 2, 4)))
    np.testing.assert_allclose(actual, tf.reshape(flat, (2, 3, 5, 4)), atol=1e-6)


def test_matches_upstream_forward_with_layout_conversion():
    # Compare arithmetic only; upstream optimizer tracking has its own failing audit.
    from HypercomplexKeras.Hyperdense import HyperDense

    x = tf.random.normal((5, 2, 4))
    layer = QuaternionDense(3)
    upstream = HyperDense(3)
    layer(x)
    blocked_input = tf.reshape(tf.transpose(x, (0, 2, 1)), (5, 8))
    upstream(blocked_input)
    upstream.kernel.assign(tf.transpose(layer.kernel, (2, 0, 1)))
    upstream.bias.assign(tf.reshape(tf.transpose(layer.bias), (12,)))
    reference = tf.transpose(tf.reshape(upstream(blocked_input), (5, 4, 3)), (0, 2, 1))
    np.testing.assert_allclose(layer(x), reference, atol=1e-6, rtol=1e-6)
