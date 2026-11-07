"""One-hidden-layer RGB controls matched by the nearest real parameter budget."""

import math

import tensorflow as tf

from src.models.quaternion_dense import QuaternionDense


def parameter_budgets(pixels, quaternion_units, classes):
    q_count = 4 * (pixels + classes + 1) * quaternion_units + classes
    real_per_unit = 3 * pixels + classes + 1
    ideal_width = (q_count - classes) / real_per_unit
    candidates = {max(1, math.floor(ideal_width)), max(1, math.ceil(ideal_width))}
    real_width = min(candidates, key=lambda width: (abs(real_per_unit * width + classes - q_count), width))
    return {
        "quaternion_real_parameters": q_count,
        "real_hidden_units": real_width,
        "real_parameters": real_per_unit * real_width + classes,
        "real_minus_quaternion_fraction": (real_per_unit * real_width + classes - q_count) / q_count,
    }


def build_rgb_model(config, encoding, seed):
    height, width, channels = config["image_shape"]
    if channels != 3:
        raise ValueError("This pilot expects RGB images")
    pixels = height * width
    budgets = parameter_budgets(pixels, config["quaternion_hidden_units"], config["classes"])
    # Three active RGB components per pixel, for both model families.
    initializer = tf.keras.initializers.RandomNormal(stddev=math.sqrt(2.0 / (3 * pixels)), seed=seed)
    if encoding == "RGB":
        inputs = tf.keras.Input((3 * pixels,))
        hidden = tf.keras.layers.Dense(
            budgets["real_hidden_units"], activation=config["activation"],
            kernel_initializer=initializer, name="real_hidden",
        )(inputs)
        readout_inputs = hidden
        expected = budgets["real_parameters"]
    elif encoding in ("ARGB", "ABGR"):
        inputs = tf.keras.Input((pixels, 4))
        hidden = QuaternionDense(
            config["quaternion_hidden_units"], activation=config["activation"],
            kernel_initializer=initializer, name="quaternion_hidden",
        )(inputs)
        readout_inputs = tf.keras.layers.Flatten()(hidden)
        expected = budgets["quaternion_real_parameters"]
    else:
        raise ValueError("Encoding must be RGB, ARGB, or ABGR")
    logits = tf.keras.layers.Dense(
        config["classes"], kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 1),
        name="real_readout",
    )(readout_inputs)
    model = tf.keras.Model(inputs, logits)
    if model.count_params() != expected:
        raise RuntimeError("Model parameter count differs from its stated budget")
    return model, tf.keras.Model(inputs, hidden), budgets
