"""Tests of the scikit-learn wrappers and a comparison with numbers printed in the course material.

LLM-assisted (code level 2): the expected values follow the author's derivations and the course
reference; the scaffolding was drafted with Claude Opus 5.5 (Claude Code, November 2026).
"""

import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from estimators import NeuralNetClassifier, NeuralNetRegressor
from network import NeuralNetwork, one_hot, train
from optimizers import GD, Adam
from regression import make_data, runge


def course_data():
    """The data cell of the lecture notebooks: sorted inputs, noise drawn after them. LLM-assisted: Claude."""
    rng = np.random.default_rng(2026)
    x = np.sort(rng.uniform(-1.0, 1.0, 100))
    return x[:, None], runge(x) + rng.normal(0.0, 0.1, 100)


def course_training_run(X, y):
    """1-50-1 sigmoid network, plain SGD, gamma 0.03, 1000 epochs, batches of 10, seed 2026.

    LLM-assisted: Claude.
    """
    net = NeuralNetwork([1, 50, 1], ["sigmoid", "identity"], "mse", seed=2026)
    train(net, X, y[:, None], GD(0.03), 1000, 10, seed=2026)
    return net


def test_reproduces_the_training_error_printed_in_the_course_solutions():
    """Exercises of week 44, solution of 2a: 'training MSE on all 100 points ... 0.025710'.

    With the same seed, initialization and SGD an independent implementation must give this number.
    """
    X, y = course_data()
    Xs = StandardScaler().fit_transform(X)
    net = course_training_run(Xs, y)
    assert round(float(np.mean((net.predict(Xs)[:, 0] - y) ** 2)), 6) == 0.025710


def test_reproduces_the_five_fold_errors_printed_in_the_course_solutions():
    """Same solution: the five validation MSEs 0.0313, 0.049, 0.0163, 0.0334 and 0.0371."""
    X, y = course_data()
    errors = []
    for tr, va in KFold(5, shuffle=True, random_state=2026).split(X):
        scaler = StandardScaler().fit(X[tr])
        net = course_training_run(scaler.transform(X[tr]), y[tr])
        errors.append(np.mean((net.predict(scaler.transform(X[va]))[:, 0] - y[va]) ** 2))
    assert np.allclose(np.round(errors, 4), [0.0313, 0.049, 0.0163, 0.0334, 0.0371])


def test_regressor_wrapper_equals_direct_training_with_centred_target():
    x, y = make_data(40, 0.1, 5)
    X = x[:, None]
    wrapped = NeuralNetRegressor(hidden=(8,), gamma=0.01, epochs=30, batch_size=8, seed=3).fit(X, y)
    net = NeuralNetwork([1, 8, 1], ["sigmoid", "identity"], "mse", seed=3)
    train(net, X, (y - y.mean())[:, None], Adam(0.01), 30, 8, seed=3)
    assert np.allclose(wrapped.predict(X), net.predict(X)[:, 0] + y.mean())


def test_optimizer_gd_uses_all_samples_in_one_update():
    """One epoch of 'gd' is one update, whatever batch_size says."""
    x, y = make_data(30, 0.1, 6)
    X = x[:, None]
    wrapped = NeuralNetRegressor(hidden=(4,), optimizer="gd", gamma=0.1, epochs=1, batch_size=2, seed=1).fit(X, y)
    net = NeuralNetwork([1, 4, 1], ["sigmoid", "identity"], "mse", seed=1)
    theta = net.theta - 0.1 * net.gradient(net.theta, X, (y - y.mean())[:, None])
    assert np.allclose(wrapped.net_.theta, theta)


def test_regressor_works_inside_pipeline_clone_and_grid_search():
    x, y = make_data(40, 0.1, 7)
    X = x[:, None]
    pipe = make_pipeline(StandardScaler(), NeuralNetRegressor(hidden=(4,), epochs=5))
    assert not hasattr(clone(pipe)[-1], "net_")
    assert cross_val_score(pipe, X, y, cv=2, scoring="neg_mean_squared_error").shape == (2,)
    grid = {"neuralnetregressor__gamma": [1e-3, 1e-2]}
    search = GridSearchCV(pipe, grid, cv=2, scoring="neg_mean_squared_error").fit(X, y)
    assert search.best_params_["neuralnetregressor__gamma"] in grid["neuralnetregressor__gamma"]


def test_diverged_network_gets_a_huge_finite_error():
    x, y = make_data(40, 0.1, 8)
    X = x[:, None]
    model = NeuralNetRegressor(hidden=(20,), activation="relu", optimizer="gd", gamma=1e6, epochs=200).fit(X, y)
    assert np.all(np.isfinite(model.predict(X)))
    assert np.mean((model.predict(X) - y) ** 2) > 1.0


def test_classifier_learns_digits_and_handles_labels_that_do_not_start_at_zero():
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target + 3
    model = NeuralNetClassifier(hidden=(32,), gamma=1e-3, epochs=30, batch_size=32).fit(X[:1400], y[:1400])
    assert model.score(X[1400:], y[1400:]) > 0.9
    assert set(model.predict(X[1400:])) <= set(range(3, 13))
    assert np.allclose(model.predict_proba(X[:5]).sum(axis=1), 1.0)
    assert one_hot(np.array([0, 2]), 3).tolist() == [[1, 0, 0], [0, 0, 1]]


def test_logistic_regression_is_the_network_without_hidden_layers():
    digits = load_digits()
    model = NeuralNetClassifier(hidden=(), epochs=2).fit(digits.data / 16.0, digits.target)
    assert model.net_.theta.size == 64 * 10 + 10
