"""
Loading of the MNIST digits (LeCun, Cortes and Burges) from the NumPy archive on Google's storage.

Training and test images are loaded by two separate functions, so that the code for model selection
cannot touch the test images by accident: it only ever calls load_train.

LLM-assisted
------------
Code level 2. Tool: Claude Opus 5.5 (Anthropic, Claude Code desktop app, November 2026).
Role: the loading, splitting and scaling rules are the author's; Claude drafted the archive
download with its caching and checksum check.
Verification: tests/test_mnist_data.py checks shapes, scaling and the split.
"""

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"  # value used by Keras
PATH = Path(__file__).resolve().parent.parent / "data" / "mnist.npz"
N_VALIDATION = 10_000


def _archive():
    """Open the archive, downloading it once and checking its checksum. LLM-assisted: Claude."""
    if not PATH.exists():
        PATH.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, PATH)
    if hashlib.sha256(PATH.read_bytes()).hexdigest() != SHA256:
        raise ValueError(f"{PATH} does not match the published checksum; delete it and try again")
    return np.load(PATH, allow_pickle=False)


def _flatten(images):
    """Flatten 28 x 28 images to rows of 784 pixels scaled to [0, 1]."""
    return images.reshape(len(images), -1) / 255.0


def load_train():
    """The 60 000 training images and labels."""
    with _archive() as data:
        return _flatten(data["x_train"]), data["y_train"].astype(int)


def load_test():
    """The 10 000 test images and labels, used only for the final evaluation."""
    with _archive() as data:
        return _flatten(data["x_test"]), data["y_test"].astype(int)


def train_validation_split(X, y, seed=2026):
    """Stratified split of the training images: 50 000 for fitting, 10 000 for validation."""
    return train_test_split(X, y, test_size=N_VALIDATION, stratify=y, random_state=seed)
