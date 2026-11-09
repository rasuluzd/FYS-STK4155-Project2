"""
Benchmarks of the network code that are reported in the text: the size of the differences to the
references, and the cost of a gradient compared with a forward pass. tests/ checks the same properties
with pass/fail thresholds; this script writes the actual numbers to results/verification.json.

LLM-assisted
------------
Code level 2. Tool: Claude Opus 5.5 (Anthropic, Claude Code desktop app, November 2026).
Role: the checks follow the author's derivations and the course reference; the standalone
autograd cost and the finite-difference loop were drafted with Claude.
Verification: tests/ covers the same checks with pass/fail thresholds.
"""

import time

import autograd.numpy as anp
import numpy as np
from autograd import grad
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from network import NeuralNetwork, one_hot, train
from optimizers import GD, max_learning_rate
from plot_style import save_results
from regression import Scaler, make_data, ols_parameters, polynomial_features, ridge_parameters

ACTIVATIONS = {"sigmoid": lambda z: 1.0 / (1.0 + anp.exp(-z)), "relu": lambda z: anp.where(z > 0, z, 0.0),
               "leaky_relu": lambda z: anp.where(z > 0, z, 0.01 * z), "identity": lambda z: z,
               "softmax": lambda z: anp.exp(z) / anp.sum(anp.exp(z), axis=1, keepdims=True)}


def _autograd_cost(theta, net, X, Y):
    """Forward pass and cost in autograd.numpy, written separately from the network. LLM-assisted: Claude."""
    a, k = X, 0
    for n_in, n_out, name in zip(net.sizes[:-1], net.sizes[1:], net.activations):
        W = anp.reshape(theta[k:k + n_in * n_out], (n_in, n_out))
        b = theta[k + n_in * n_out:k + n_in * n_out + n_out]
        k += n_in * n_out + n_out
        a = ACTIVATIONS[name](anp.dot(a, W) + b)
    if net.cost_name == "mse":
        return 0.5 * anp.sum((a - Y) ** 2) / len(a)
    return -anp.sum(Y * anp.log(a)) / len(a)


def gradient_differences():
    """Largest relative difference between backpropagation and Autograd, and to finite differences.

    LLM-assisted: Claude drafted the finite-difference loop.
    """
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(50, 4))
    targets = {"mse": rng.normal(size=(50, 2)), "cross_entropy": one_hot(rng.integers(0, 2, 50), 2)}
    out = {}
    for cost, output in (("mse", "identity"), ("cross_entropy", "softmax")):
        for hidden in ("sigmoid", "relu", "leaky_relu"):
            net = NeuralNetwork([4, 16, 16, 2], [hidden, hidden, output], cost, seed=2026)
            analytic = net.gradient(net.theta, X, targets[cost])
            automatic = grad(_autograd_cost)(net.theta, net, X, targets[cost])
            numeric = np.zeros(net.theta.size)
            for i in range(net.theta.size):
                step = np.zeros(net.theta.size)
                step[i] = 1e-6
                numeric[i] = (net.cost(net.theta + step, X, targets[cost]) - net.cost(net.theta - step, X, targets[cost])) / 2e-6
            scale = np.abs(analytic).max()
            out[f"{cost}_{hidden}"] = {"autograd": np.abs(analytic - automatic).max() / scale, "finite_differences": np.abs(analytic - numeric).max() / scale}
    return out


def closed_form_differences():
    """Networks without hidden layers against OLS, Ridge and logistic regression."""
    x, y = make_data(80, noise=0.1, seed=3)
    A = polynomial_features(x, 3)
    scaler = Scaler().fit(A, y)
    X, yc = scaler.transform(A), scaler.center(y)
    out = {}
    net = NeuralNetwork([3, 1], ["identity"], "mse", seed=0)
    train(net, X, yc[:, None], GD(max_learning_rate(X)), 30_000, len(X), seed=0)
    reference = ols_parameters(X, yc)
    out["ols"] = np.linalg.norm(net.layers(net.theta)[0][0].ravel() - reference) / np.linalg.norm(reference)
    net = NeuralNetwork([3, 1], ["identity"], "mse", "l2", penalize_biases=False, seed=0)
    train(net, X, yc[:, None], GD(max_learning_rate(X)), 30_000, len(X), lam=1e-3, seed=0)
    reference = ridge_parameters(X, yc, 2e-3)
    out["ridge"] = np.linalg.norm(net.layers(net.theta)[0][0].ravel() - reference) / np.linalg.norm(reference)
    Xc, yc = make_classification(n_samples=300, n_features=5, n_informative=4, n_redundant=0, n_classes=3,
                                 n_clusters_per_class=1, class_sep=0.5, flip_y=0.1, random_state=0)
    reference = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10_000).fit(Xc, yc)
    net = NeuralNetwork([5, 3], ["softmax"], "cross_entropy", seed=0)
    train(net, Xc, one_hot(yc, 3), GD(0.5), 8000, len(Xc), seed=0)
    out["logistic_probabilities"] = np.abs(net.predict(Xc) - reference.predict_proba(Xc)).max()
    return out


def gradient_cost(repeats=30):
    """Time of one gradient in units of one forward pass, for networks of very different size."""
    rng = np.random.default_rng(2026)
    rows = []
    for sizes, activations, cost, n in (([1, 50, 1], ["sigmoid", "identity"], "mse", 100),
                                        ([1, 100, 100, 1], ["sigmoid", "sigmoid", "identity"], "mse", 100),
                                        ([1, 100, 100, 100, 100, 1], ["relu"] * 4 + ["identity"], "mse", 100),
                                        ([784, 128, 10], ["relu", "softmax"], "cross_entropy", 128),
                                        ([784, 512, 10], ["relu", "softmax"], "cross_entropy", 128),
                                        ([784, 512, 512, 10], ["relu", "relu", "softmax"], "cross_entropy", 128)):
        X = rng.random((n, sizes[0]))
        Y = rng.normal(size=(n, 1)) if cost == "mse" else one_hot(rng.integers(0, 10, n))
        net = NeuralNetwork(sizes, activations, cost, seed=2026)
        forward, gradient = [], []
        for _ in range(repeats):
            start = time.perf_counter()
            net.predict(X)
            forward.append(time.perf_counter() - start)
            start = time.perf_counter()
            net.gradient(net.theta, X, Y)
            gradient.append(time.perf_counter() - start)
        rows.append({"sizes": sizes, "n_parameters": int(net.theta.size), "forward_seconds": float(np.median(forward)),
                     "gradient_seconds": float(np.median(gradient)), "ratio": float(np.median(gradient) / np.median(forward))})
    return rows


if __name__ == "__main__":
    results = {"gradients": gradient_differences(), "closed_forms": closed_form_differences(), "cost": gradient_cost()}
    save_results("verification", results)
    print(results)
