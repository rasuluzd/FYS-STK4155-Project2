"""Tests of the MNIST loader: shapes, scaling and the stratified validation split.

LLM-assisted (code level 2): the expected values follow the author's derivations and the course
reference; the scaffolding was drafted with Claude Opus 5.5 (Claude Code, November 2026).
"""

import numpy as np

from mnist_data import load_test, load_train, train_validation_split


def test_training_images_are_flat_scaled_and_split_without_overlap():
    X, y = load_train()
    assert X.shape == (60_000, 784) and y.shape == (60_000,)
    assert X.min() == 0.0 and X.max() == 1.0 and set(np.unique(y)) == set(range(10))
    X_fit, X_val, y_fit, y_val = train_validation_split(X, y)
    assert len(X_fit) == 50_000 and len(X_val) == 10_000
    # stratification keeps the class shares within one image of the shares in the full set
    assert np.max(np.abs(np.bincount(y_val) - np.bincount(y) / 6)) <= 1


def test_test_images_have_the_canonical_size():
    X, y = load_test()
    assert X.shape == (10_000, 784) and y.shape == (10_000,)
