"""Download and verify the official CIFAR-10 archive into the local Keras cache."""

import argparse
import contextlib

from scripts.runtime import ROOT, tf

ARCHIVE_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
MIRROR = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-10-python.tar.gz"
OFFICIAL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("tensorflow", "official"), default="tensorflow")
    args = parser.parse_args()
    origin = MIRROR if args.source == "tensorflow" else OFFICIAL
    print(f"Downloading CIFAR-10 from {origin}; requiring official SHA-256 {ARCHIVE_SHA256}", flush=True)
    log_path = ROOT / ".cache" / "cifar10-download.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        tf.keras.utils.get_file(
            fname="cifar-10-batches-py-target", origin=origin, extract=True,
            file_hash=ARCHIVE_SHA256, hash_algorithm="sha256",
        )
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
    print(f"Verified CIFAR-10: training={train_x.shape}, test={test_x.shape}")
