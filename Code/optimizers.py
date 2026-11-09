"""
Gradient-based optimisation for parts e-h: cost functions, analytical and Autograd gradients,
the Hessian bound on the learning rate, the update rules (plain GD, momentum, AdaGrad, RMSprop,
Adam) and full-batch and mini-batch drivers (lecture notes, Ch. 4; weeks 37-38).

update(g, eta) returns the step and the driver performs theta = theta - step. Costs are mean
squared errors, so the Ridge gradient adds 2*lam*theta, also for a mini-batch.

LLM-assisted
------------
Code level 3. Claude (Claude Code, October 2026) generated the class structure and the GD/SGD
driver loops. The author wrote the cost functions, gradients and update rules. Docstrings added
with Claude and OpenAI Codex.
Verification: tests/test_optimizers.py checks analytical/AD agreement, all five update rules on a
small Ridge problem and selected GD/SGD properties.
"""

import numpy as np
import autograd.numpy as anp
from autograd import grad as _autograd_grad


# ----------------------------------------------------------------------------------------------
# Cost functions (autograd.numpy, so they can be differentiated automatically)
# ----------------------------------------------------------------------------------------------
def cost_ols(theta, X, y, lam=0.0):
    """Return mean squared residuals; lam is ignored for a common calling signature."""
    return anp.mean((y - anp.dot(X, theta)) ** 2)


def cost_ridge(theta, X, y, lam):
    """Add lam times the squared slope norm to the mean squared residuals."""
    return cost_ols(theta, X, y) + lam * anp.sum(theta**2)


def cost_lasso(theta, X, y, lam):
    """Add the L1 slope penalty to the mean squared residuals."""
    return cost_ols(theta, X, y) + lam * anp.sum(anp.abs(theta))


COSTS = {"ols": cost_ols, "ridge": cost_ridge, "lasso": cost_lasso}


# ----------------------------------------------------------------------------------------------
# Analytical gradients
# ----------------------------------------------------------------------------------------------
def grad_ols(theta, X, y, lam=0.0):
    """Return (2/n) X.T @ (X @ theta - y), using the supplied batch length."""
    return (2.0 / X.shape[0]) * (X.T @ (X @ theta - y))


def grad_ridge(theta, X, y, lam):
    """Add 2*lam*theta to the OLS gradient (book 4.4, week37 Exercise 4b)."""
    return grad_ols(theta, X, y) + 2.0 * lam * theta


def grad_lasso(theta, X, y, lam):
    """Return the MSE gradient plus lam*sign(theta), choosing sign(0)=0.

    np.sign(0) = 0, which is a valid subgradient of |theta_j| at theta_j = 0.
    """
    return grad_ols(theta, X, y) + lam * np.sign(theta)


ANALYTICAL_GRADIENTS = {"ols": grad_ols, "ridge": grad_ridge, "lasso": grad_lasso}

# Automatic differentiation: autograd traces the Python cost function and applies the chain
# rule in reverse mode. grad(f) differentiates with respect to the first argument (theta).
AUTOGRAD_GRADIENTS = {name: _autograd_grad(cost) for name, cost in COSTS.items()}


def make_gradient(method, X, y, lam=0.0, use_autograd=False):
    """Bind one training problem, leaving only theta as the gradient argument."""
    g = AUTOGRAD_GRADIENTS[method] if use_autograd else ANALYTICAL_GRADIENTS[method]
    return lambda theta: g(theta, X, y, lam)


# ----------------------------------------------------------------------------------------------
# Hessian and the learning-rate bound
# ----------------------------------------------------------------------------------------------
def hessian(X, lam=0.0):
    """Return H=(2/n) X.T @ X + 2*lam*I for the OLS/Ridge quadratic."""
    n, p = X.shape
    return (2.0 / n) * (X.T @ X) + 2.0 * lam * np.eye(p)


def max_learning_rate(X, lam=0.0):
    """Return the strict plain-GD upper bound 2/h_max (book 4.5)."""
    return 2.0 / np.linalg.eigvalsh(hessian(X, lam)).max()


# ----------------------------------------------------------------------------------------------
# Update rules. Each object returns the step that is subtracted from theta.
# ----------------------------------------------------------------------------------------------
class GD:
    """Plain gradient descent (book 4.3; week37 Exercise 4)."""

    name = "Plain GD"

    def __init__(self, eta):
        """Store the base learning rate; drivers may supply a scheduled rate to update."""
        self.eta = eta

    def reset(self, n_params):
        """Do nothing: plain GD has no accumulated state to clear."""
        pass

    def update(self, g, eta):
        """Return eta times the gradient; the driver subtracts this step from theta."""
        return eta * g


class Momentum(GD):
    """Heavy-ball momentum: change <- gamma*change + eta*g."""

    name = "Momentum"

    def __init__(self, eta, gamma=0.9):
        """Store eta and the previous-step weight gamma (beta in the course notation)."""
        self.eta, self.gamma = eta, gamma

    def reset(self, n_params):
        """Start a new run with zero previous change in every coordinate."""
        self.change = np.zeros(n_params)

    def update(self, g, eta):
        """Add the current gradient step to gamma times the previous change."""
        self.change = eta * g + self.gamma * self.change
        return self.change


class AdaGrad(GD):
    """Coordinatewise accumulated-square scaling (book 4.9; week38 Exercise 2)."""

    name = "AdaGrad"

    def __init__(self, eta, delta=1e-8):
        """Store the base step and the small denominator offset outside the square root."""
        self.eta, self.delta = eta, delta

    def reset(self, n_params):
        """Clear the sum of squared gradients once at the start of a run."""
        self.G = np.zeros(n_params)

    def update(self, g, eta):
        """Accumulate g squared coordinatewise, then divide the step by its square root."""
        self.G += g * g
        return eta * g / (self.delta + np.sqrt(self.G))


class RMSprop(GD):
    """Exponential squared-gradient memory (book 4.10; week38 Exercise 2)."""

    name = "RMSprop"

    def __init__(self, eta, rho=0.99, delta=1e-8):
        """Store eta, the squared-gradient memory rho, and the denominator offset."""
        self.eta, self.rho, self.delta = eta, rho, delta

    def reset(self, n_params):
        """Clear the moving squared-gradient average for a new run."""
        self.s = np.zeros(n_params)

    def update(self, g, eta):
        """Update the moving average and return the coordinatewise scaled step."""
        self.s = self.rho * self.s + (1.0 - self.rho) * g * g
        return eta * g / (self.delta + np.sqrt(self.s))


class Adam(GD):
    """Bias-corrected first and second moments (book 4.11; week38 Exercise 2)."""

    name = "Adam"

    def __init__(self, eta, beta1=0.9, beta2=0.999, delta=1e-8):
        """Store the two moment weights, base step and denominator offset."""
        self.eta, self.beta1, self.beta2, self.delta = eta, beta1, beta2, delta

    def reset(self, n_params):
        """Clear both moments and reset the bias-correction update counter."""
        self.m = np.zeros(n_params)
        self.v = np.zeros(n_params)
        self.t = 0

    def update(self, g, eta):
        """Update both moments, correct their zero-initialization bias, then form a step."""
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * g * g
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        return eta * m_hat / (np.sqrt(v_hat) + self.delta)


OPTIMIZERS = {"gd": GD, "momentum": Momentum, "adagrad": AdaGrad, "rmsprop": RMSprop, "adam": Adam}


# ----------------------------------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------------------------------
def _relative_error(theta, theta_ref):
    """Measure coefficient error relative to a supplied nonzero reference vector."""
    return np.linalg.norm(theta - theta_ref) / np.linalg.norm(theta_ref)


def gradient_descent(gradient, theta0, optimizer, n_iter, theta_ref=None, tol=None,
                     record_every=1):
    """Full-batch loop: compute the gradient, ask the optimizer for a step, subtract it.

    Stops after n_iter updates, or when the relative coefficient error to theta_ref is below
    tol. Returns theta and a dict with errors, iterations, converged and diverged.
    """
    theta = np.array(theta0, dtype=float)
    optimizer.reset(theta.size)
    errors = []
    converged = diverged = False
    k = 0
    for k in range(1, n_iter + 1):
        step = optimizer.update(gradient(theta), optimizer.eta)
        theta = theta - step
        if not np.all(np.isfinite(theta)) or np.abs(theta).max() > 1e12:
            diverged = True
            break
        if theta_ref is not None:
            err = _relative_error(theta, theta_ref)
            if k % record_every == 0:
                errors.append(err)
            if tol is not None and err < tol:
                converged = True
                break
    return theta, {"errors": np.array(errors), "iterations": k,
                   "converged": converged, "diverged": diverged}


def stochastic_gradient_descent(grad_fn, X, y, lam, theta0, optimizer, n_epochs, batch_size,
                                schedule=None, seed=None, theta_ref=None,
                                callback=None):
    """Mini-batch SGD: reshuffle every epoch and apply the update rule to each batch.

    The data term is averaged over the actual batch size and the optimizer state persists
    across epochs; schedule(t) gives the rate at update t. Returns theta and a dict with the
    per-epoch errors, the callback history and the number of updates.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    m = int(np.ceil(n / batch_size))
    theta = np.array(theta0, dtype=float)
    optimizer.reset(theta.size)
    errors, history = [], []
    diverged = False
    t = 0
    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        for i in range(m):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            eta = optimizer.eta if schedule is None else schedule(t)
            g = grad_fn(theta, X[idx], y[idx], lam)
            theta = theta - optimizer.update(g, eta)
            t += 1
        if not np.all(np.isfinite(theta)) or np.abs(theta).max() > 1e12:
            diverged = True
            break
        if theta_ref is not None:
            errors.append(_relative_error(theta, theta_ref))
        if callback is not None:
            history.append(callback(theta))
    return theta, {"errors": np.array(errors), "history": np.array(history), "updates": t,
                   "diverged": diverged}
