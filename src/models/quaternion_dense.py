"""Minimal quaternion Dense layer for the documented Keras 3 fallback.

Grouped layout: (batch, ..., input_quaternion_units, 4), components (real, i, j, k).
The convention is y_j = sum_i x_i * W_ij + b_j (right Hamilton product).
Each connection has four real trainable scalars; there is no pruning or masking.
"""

from numbers import Integral

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="hypercomplex_redundancy")
class QuaternionDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer="glorot_uniform", **kwargs):
        super().__init__(**kwargs)
        if isinstance(units, bool) or not isinstance(units, Integral) or units < 1:
            raise ValueError("units must be a positive integer (quaternion units)")
        self.units = int(units)
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = bool(use_bias)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=3, axes={-1: 4})

    def build(self, input_shape):
        if len(input_shape) < 3 or input_shape[-1] != 4:
            raise ValueError("Expected grouped quaternion input (..., units, 4)")
        input_units = input_shape[-2]
        if input_units is None or input_units < 1:
            raise ValueError("The input quaternion width must be known and positive")
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=3, axes={-2: input_units, -1: 4})
        self.kernel = self.add_weight(
            name="kernel", shape=(input_units, self.units, 4),
            initializer=self.kernel_initializer, trainable=True,
        )
        self.bias = self.add_weight(
            name="bias", shape=(self.units, 4), initializer="zeros", trainable=True,
        ) if self.use_bias else None
        super().build(input_shape)

    def call(self, inputs):
        xr, xi, xj, xk = tf.unstack(inputs, axis=-1)
        wr, wi, wj, wk = tf.unstack(self.kernel, axis=-1)
        outputs = tf.stack((
            xr @ wr - xi @ wi - xj @ wj - xk @ wk,
            xr @ wi + xi @ wr + xj @ wk - xk @ wj,
            xr @ wj - xi @ wk + xj @ wr + xk @ wi,
            xr @ wk + xi @ wj - xj @ wi + xk @ wr,
        ), axis=-1)
        if self.bias is not None:
            outputs = outputs + self.bias
        return self.activation(outputs) if self.activation is not None else outputs

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-2]) + (self.units, 4)

    def get_config(self):
        return {
            **super().get_config(),
            "units": self.units,
            "activation": tf.keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
        }
