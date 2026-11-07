"""Shared, explicit CPU settings for the small Phase 0 checks."""

import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_HOME", str(ROOT / ".cache" / "keras"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np
import tensorflow as tf


def configure(seed):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def environment(seed):
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": {
            name: metadata.version(name)
            for name in ("tensorflow", "keras", "numpy", "matplotlib", "pytest", "HypercomplexKeras")
        },
        "seed": seed,
        "devices": [{"name": d.name, "type": d.device_type} for d in tf.config.get_visible_devices()],
        "tensorflow_build": tf.sysconfig.get_build_info(),
        "settings": {name: os.environ.get(name) for name in (
            "KERAS_BACKEND", "CUDA_VISIBLE_DEVICES", "TF_DETERMINISTIC_OPS", "TF_ENABLE_ONEDNN_OPTS"
        )},
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_hashes(path):
    data = Path(path).read_bytes()
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "git_blob_sha1": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
    }


def real_parameter_count(variables):
    return sum(int(np.prod(v.shape)) for v in variables)
