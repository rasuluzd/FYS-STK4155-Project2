"""Selected gradient and optimizer checks against known reference answers.

All five update rules are tested on one degree-three Ridge problem. Separate
tests check AD gradients, the plain-GD stability bound and
full-batch/mini-batch SGD. These cases do not establish convergence on every
degree, dataset or learning rate. Codex clarified this scope on 5 October 2026;
no executable test statements changed.

LLM-assisted (code level 4): generated with Claude (Anthropic, Claude Code), October 2026.
"""

import numpy as np
import autograd.numpy as anp
from autograd import grad
import pytest

from optimizers import (ANALYTICAL_GRADIENTS, AUTOGRAD_GRADIENTS, OPTIMIZERS, GD, Adam,
                        gradient_descent, grad_ols, make_gradient, max_learning_rate,
                        stochastic_gradient_descent)
from regression import (Scaler, make_data, ols_parameters,
                        polynomial_features, ridge_parameters)

x, y = make_data(80, noise=0.1, seed=3)
X_raw = polynomial_features(x, 3)
sc = Scaler().fit(X_raw, y)
X, yc = sc.transform(X_raw), sc.center(y)
rng = np.random.default_rng(0)


@pytest.mark.parametrize("method", ["ols", "ridge", "lasso"])
def test_autograd_equals_analytical_gradient(method):
    """LLM-assisted: generated with Claude (Claude Code, October 2026)."""
    lam = 0.01
    for _ in range(5):
        theta = rng.normal(size=X.shape[1])  # random point, almost surely no zero component
        g_ana = ANALYTICAL_GRADIENTS[method](theta, X, yc, lam)
        g_auto = AUTOGRAD_GRADIENTS[method](theta, X, yc, lam)
        assert np.max(np.abs(g_ana - g_auto)) < 1e-13 * max(1.0, np.max(np.abs(g_ana)))


def test_autograd_derivative_of_abs_at_zero_is_a_subgradient():
    """LLM-assisted: generated with Claude (Claude Code, October 2026)."""
    d = grad(lambda t: anp.abs(t))(0.0)
    assert -1.0 <= d <= 1.0


def test_plain_gd_converges_below_and_diverges_above_bound():
    """LLM-assisted: generated with Claude (Claude Code, October 2026)."""
    theta_ols = ols_parameters(X, yc)
    eta_max = max_learning_rate(X)
    _, ok = gradient_descent(make_gradient("ols", X, yc), np.zeros(3), GD(0.9 * eta_max), 20000,
                             theta_ref=theta_ols, tol=1e-8)
    _, bad = gradient_descent(make_gradient("ols", X, yc), np.zeros(3), GD(1.05 * eta_max), 20000,
                              theta_ref=theta_ols)
    assert ok["converged"] and bad["diverged"]


@pytest.mark.parametrize("name,eta", [("gd", 0.05), ("momentum", 0.05), ("adagrad", 0.5),
                                      ("rmsprop", 0.01), ("adam", 0.02)])
def test_all_optimizers_reach_ridge_closed_form(name, eta):
    """LLM-assisted: generated with Claude (Claude Code, October 2026)."""
    lam = 1e-3
    theta_ridge = ridge_parameters(X, yc, lam)
    theta, info = gradient_descent(make_gradient("ridge", X, yc, lam), np.zeros(3),
                                   OPTIMIZERS[name](eta), 200000, theta_ref=theta_ridge, tol=1e-5)
    assert info["converged"], (name, info["errors"][-1])


def test_sgd_with_full_batch_equals_gd():
    """LLM-assisted: generated with Claude (Claude Code, October 2026)."""
    eta, n_epochs = 0.05, 50
    theta_gd, _ = gradient_descent(make_gradient("ols", X, yc), np.zeros(3), GD(eta), n_epochs)
    theta_sgd, _ = stochastic_gradient_descent(grad_ols, X, yc, 0.0, np.zeros(3), GD(eta),
                                               n_epochs, batch_size=len(yc), seed=1)
    assert np.allclose(theta_gd, theta_sgd)


def test_sgd_adam_gets_close_to_ols():
    """LLM-assisted: generated with Claude (Claude Code, October 2026)."""
    theta_ols = ols_parameters(X, yc)
    theta, info = stochastic_gradient_descent(grad_ols, X, yc, 0.0, np.zeros(3), Adam(0.01),
                                              n_epochs=400, batch_size=10, seed=2,
                                              theta_ref=theta_ols)
    assert info["errors"][-1] < 0.05
