"""Tests of the neural network against automatic differentiation, closed forms and scikit-learn.

LLM-assisted (code level 2): the expected values follow the author's derivations and the course
reference; the scaffolding was drafted with Claude Opus 5.5 (Claude Code, November 2026).
"""

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import grad
from sklearn.datasets import load_digits, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from network import NeuralNetwork, accuracy, one_hot, sigmoid, softmax, train
from optimizers import GD, Adam, max_learning_rate
from regression import Scaler, make_data, ols_parameters, polynomial_features, ridge_parameters

AUTOGRAD_ACTIVATIONS = {
    "sigmoid": lambda z: 1.0 / (1.0 + anp.exp(-z)),
    "relu": lambda z: anp.where(z > 0, z, 0.0),
    "leaky_relu": lambda z: anp.where(z > 0, z, 0.01 * z),
    "identity": lambda z: z,
    "softmax": lambda z: anp.exp(z) / anp.sum(anp.exp(z), axis=1, keepdims=True),
}
AUTOGRAD_COSTS = {
    "mse": lambda a, y: 0.5 * anp.sum((a - y) ** 2) / len(a),
    "cross_entropy": lambda a, y: -anp.sum(y * anp.log(a)) / len(a),
    "binary_cross_entropy": lambda a, y: -anp.sum(y * anp.log(a) + (1 - y) * anp.log(1 - a)) / len(a),
}


def independent_cost(theta, net, X, Y, lam):
    """The cost written again in autograd.numpy: forward pass only, so backprop plays no part.

    LLM-assisted: Claude.
    """
    a, k, norm = X, 0, 0.0
    for n_in, n_out, name in zip(net.sizes[:-1], net.sizes[1:], net.activations):
        W = anp.reshape(theta[k:k + n_in * n_out], (n_in, n_out))
        b = theta[k + n_in * n_out:k + n_in * n_out + n_out]
        k += n_in * n_out + n_out
        a = AUTOGRAD_ACTIVATIONS[name](anp.dot(a, W) + b)
        if net.penalty == "l2":
            norm = norm + anp.sum(W ** 2) + (anp.sum(b ** 2) if net.penalize_biases else 0.0)
        elif net.penalty == "l1":
            norm = norm + anp.sum(anp.abs(W)) + (anp.sum(anp.abs(b)) if net.penalize_biases else 0.0)
    return AUTOGRAD_COSTS[net.cost_name](a, Y) + lam * norm


def small_problem(output, cost, seed=0):
    """Random inputs and targets of the right kind for the output layer. LLM-assisted: Claude."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(20, 3))
    if cost == "cross_entropy" or output == "softmax":
        return X, one_hot(rng.integers(0, 4, 20), 4)
    if cost == "binary_cross_entropy":
        return X, rng.integers(0, 2, (20, 1)).astype(float)
    return X, rng.normal(size=(20, 2))


OUTPUT_PAIRS = [("mse", "identity"), ("cross_entropy", "softmax"), ("binary_cross_entropy", "sigmoid"),
                ("mse", "sigmoid"), ("mse", "softmax")]


@pytest.mark.parametrize("cost,output", OUTPUT_PAIRS)
@pytest.mark.parametrize("hidden", ["sigmoid", "relu", "leaky_relu"])
@pytest.mark.parametrize("penalty,biases", [(None, True), ("l2", True), ("l1", True),
                                            ("l2", False), ("l1", False)])
def test_backpropagation_equals_automatic_differentiation(cost, output, hidden, penalty, biases):
    X, Y = small_problem(output, cost)
    net = NeuralNetwork([3, 7, 6, Y.shape[1]], [hidden, hidden, output], cost, penalty, biases, seed=1)
    lam = 0.0 if penalty is None else 1e-3
    auto = grad(independent_cost)(net.theta, net, X, Y, lam)
    assert np.max(np.abs(net.gradient(net.theta, X, Y, lam) - auto)) < 1e-12


@pytest.mark.parametrize("cost,output", OUTPUT_PAIRS)
@pytest.mark.parametrize("hidden", ["sigmoid", "relu", "leaky_relu"])
def test_backpropagation_equals_finite_differences(cost, output, hidden):
    X, Y = small_problem(output, cost, seed=2)
    net = NeuralNetwork([3, 7, Y.shape[1]], [hidden, output], cost, seed=3)
    analytic = net.gradient(net.theta, X, Y)
    rng = np.random.default_rng(4)
    for i in rng.choice(net.theta.size, 15, replace=False):
        step = np.zeros(net.theta.size)
        step[i] = 1e-6
        numeric = (net.cost(net.theta + step, X, Y) - net.cost(net.theta - step, X, Y)) / 2e-6
        assert abs(numeric - analytic[i]) < 1e-6 * (abs(analytic[i]) + 1e-3)


def test_cost_method_equals_independent_cost():
    X, Y = small_problem("identity", "mse")
    net = NeuralNetwork([3, 5, 2], ["sigmoid", "identity"], "mse", "l2", seed=5)
    assert np.isclose(net.cost(net.theta, X, Y, 0.01), independent_cost(net.theta, net, X, Y, 0.01))


def test_activations_are_stable_for_large_arguments():
    z = np.array([[-1000.0, 0.0, 1000.0]])
    assert np.allclose(sigmoid(z), [[0.0, 0.5, 1.0]])
    assert np.allclose(softmax(z), [[0.0, 0.0, 1.0]])


def scaled_polynomial_problem(degree=3):
    """Standardized polynomial features and a centred target, as in project 1. LLM-assisted: Claude."""
    x, y = make_data(80, noise=0.1, seed=3)
    X = polynomial_features(x, degree)
    scaler = Scaler().fit(X, y)
    return scaler.transform(X), scaler.center(y)


def test_linear_network_recovers_closed_form_ols():
    """A network with no hidden layer and a linear output is the OLS model."""
    X, y = scaled_polynomial_problem()
    net = NeuralNetwork([3, 1], ["identity"], "mse", seed=0)
    train(net, X, y[:, None], GD(max_learning_rate(X)), epochs=30_000, batch_size=len(X))
    W, b = net.layers(net.theta)[0]
    assert np.allclose(W.ravel(), ols_parameters(X, y), rtol=1e-6)
    assert abs(b[0]) < 1e-8


def test_linear_network_with_l2_recovers_ridge_with_rescaled_penalty():
    """The cost here has a factor 1/2 on the MSE, so lam here is 2*lam in the project-1 cost."""
    X, y = scaled_polynomial_problem()
    lam = 1e-3
    net = NeuralNetwork([3, 1], ["identity"], "mse", "l2", penalize_biases=False, seed=0)
    train(net, X, y[:, None], GD(max_learning_rate(X)), epochs=30_000, batch_size=len(X), lam=lam)
    W, _ = net.layers(net.theta)[0]
    assert np.allclose(W.ravel(), ridge_parameters(X, y, 2.0 * lam), rtol=1e-6)


@pytest.mark.parametrize("C,penalty", [(np.inf, None), (1.0, "l2")])
def test_no_hidden_layers_equals_multinomial_logistic_regression(C, penalty):
    """Same predicted probabilities as scikit-learn, with and without the L2 penalty.

    scikit-learn minimises the mean loss + ||W||^2/(2 n C) and leaves the biases out.
    """
    X, y = make_classification(n_samples=300, n_features=5, n_informative=4, n_redundant=0,
                               n_classes=3, n_clusters_per_class=1, class_sep=0.5, flip_y=0.1,
                               random_state=0)
    reference = LogisticRegression(C=C, tol=1e-10, max_iter=10_000).fit(X, y)
    lam = 0.0 if penalty is None else 1.0 / (2.0 * len(X) * C)
    net = NeuralNetwork([5, 3], ["softmax"], "cross_entropy", penalty, penalize_biases=False, seed=0)
    # plain gradient descent, since Adam with a constant rate keeps moving around the minimum
    train(net, X, one_hot(y, 3), GD(0.5), epochs=8000, batch_size=len(X), lam=lam, seed=0)
    assert np.max(np.abs(net.predict(X) - reference.predict_proba(X))) < 1e-6


def test_hidden_network_competes_with_scikit_learn_on_digits():
    """Same architecture and optimizer settings should give about the same accuracy."""
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    X_tr, y_tr, X_te, y_te = X[:1400], y[:1400], X[1400:], y[1400:]
    net = NeuralNetwork([64, 32, 10], ["relu", "softmax"], "cross_entropy", seed=0)
    train(net, X_tr, one_hot(y_tr), Adam(1e-3), epochs=60, batch_size=32, seed=0)
    ref = MLPClassifier((32,), activation="relu", solver="adam", alpha=0.0, batch_size=32,
                        learning_rate_init=1e-3, max_iter=60, random_state=0, tol=0.0,
                        n_iter_no_change=1000).fit(X_tr, y_tr)
    assert abs(accuracy(net.predict(X_te), y_te) - ref.score(X_te, y_te)) < 0.03
    assert accuracy(net.predict(X_te), y_te) > 0.90


def test_network_can_drive_the_training_cost_to_zero():
    """A sanity check: without noise a small network must be able to memorize four points."""
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = np.array([[0.0], [1.0], [1.0], [0.0]])  # XOR, which no linear model can represent
    net = NeuralNetwork([2, 8, 1], ["sigmoid", "identity"], "mse", seed=0)
    train(net, X, Y, Adam(0.05), epochs=3000, batch_size=4)
    assert net.cost(net.theta, X, Y) < 1e-6


def test_parameter_count_views_and_seed():
    net = NeuralNetwork([784, 100, 10], ["relu", "softmax"], "cross_entropy", seed=7)
    assert net.theta.size == 784 * 100 + 100 + 100 * 10 + 10 == 79_510
    assert [W.shape for W, _ in net.layers(net.theta)] == [(784, 100), (100, 10)]
    assert np.array_equal(net.theta, NeuralNetwork([784, 100, 10], ["relu", "softmax"],
                                                   "cross_entropy", seed=7).theta)
    assert net.hidden_outputs(np.zeros((3, 784)))[0].shape == (3, 100)


def test_layer_changes_measure_each_layer_separately():
    net = NeuralNetwork([3, 4, 4, 1], ["sigmoid", "sigmoid", "identity"], seed=1)
    theta = net.theta.copy()
    net.layers(theta)[1][0][:] *= 3.0  # triple the weights of the second layer only
    net.layers(theta)[2][1][:] += 5.0  # biases do not count
    assert np.allclose(net.layer_changes(theta, net.theta), [0.0, 2.0, 0.0])


def test_network_keeps_its_initial_parameters_until_training_ends():
    """The probes and the distance moved compare theta with net.theta during training."""
    net = NeuralNetwork([1, 5, 1], ["sigmoid", "identity"], seed=3)
    theta0, seen = net.theta.copy(), []
    X = np.linspace(-1, 1, 20)[:, None]
    train(net, X, X ** 2, GD(0.1), 3, 5, seed=0, on_epoch=lambda theta: seen.append(np.array_equal(net.theta, theta0)))
    assert seen == [True] * 3 and not np.array_equal(net.theta, theta0)


def test_initial_weight_width_depends_on_the_number_of_inputs():
    sigmoid_net = NeuralNetwork([400, 300, 1], ["sigmoid", "identity"], seed=0)
    relu_net = NeuralNetwork([400, 300, 1], ["relu", "identity"], seed=0)
    unit_net = NeuralNetwork([400, 300, 1], ["sigmoid", "identity"], init="unit", seed=0)
    assert np.isclose(sigmoid_net.layers(sigmoid_net.theta)[0][0].var(), 1 / 400, rtol=0.03)
    assert np.isclose(relu_net.layers(relu_net.theta)[0][0].var(), 2 / 400, rtol=0.03)
    assert np.isclose(unit_net.layers(unit_net.theta)[0][0].var(), 1.0, rtol=0.03)
    assert np.all(sigmoid_net.layers(sigmoid_net.theta)[0][1] == 0.01)


def test_invalid_descriptions_are_rejected():
    with pytest.raises(ValueError):
        NeuralNetwork([1, 5, 1], ["sigmoid"])
    with pytest.raises(ValueError):
        NeuralNetwork([1, 1], ["identity"], cost="hinge")
