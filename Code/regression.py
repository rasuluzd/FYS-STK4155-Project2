"""
Polynomial regression on Runge's function: data and the main train/test split, design matrix,
scaling, closed-form OLS and Ridge, a scikit-learn Lasso reference, error metrics and an
estimator class.

Costs are normalised with 1/n:
    OLS   C = (1/n) ||y - X theta||^2
    Ridge C = OLS + lam ||theta||^2   ->  theta = (X^T X + n lam I)^(-1) X^T y
    Lasso C = OLS + lam ||theta||_1
so the scikit-learn references are Ridge(alpha=n*lam) and Lasso(alpha=lam/2). The design matrix
has no column of ones: X is standardised and y centred with training statistics only, and the
intercept is the training mean of y (lecture notes, Sec. 3.13).

LLM-assisted
------------
Code level 4. Generated with Claude (Claude Code, October 2026); checked_lasso_path by OpenAI
Codex (5 October 2026). Tested and checked by the author.
Verification: tests/test_regression.py compares every closed-form result with scikit-learn.
"""

import warnings
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, lasso_path
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def runge(x):
    """Runge's function f(x) = 1 / (1 + 25 x^2)."""
    return 1.0 / (1.0 + 25.0 * x**2)


def make_data(n, noise=0.1, seed=None, uniform=True):
    """Return n points (x, y) with y = f(x) + eps, eps ~ N(0, noise^2), x in [-1, 1]."""
    rng = np.random.default_rng(seed)
    if uniform:
        x = rng.uniform(-1.0, 1.0, n)
    else:
        x = np.linspace(-1.0, 1.0, n)
    y = runge(x) + noise * rng.standard_normal(n)
    return x, y


# the main data set of the report
SEED = 2026          # seed used for the main data set and all splits
N_POINTS = 100       # number of data points in the main data set
NOISE = 0.1          # standard deviation sigma of the Gaussian noise
TEST_SIZE = 0.2      # 80/20 train/test split


def main_split(n=N_POINTS, noise=NOISE, seed=SEED):
    """The main data set of the report: x ~ U[-1, 1], y = f(x) + N(0, noise^2), 80/20 split.

    LLM-assisted: generated with Claude (Claude Code, October 2026).
    """
    x, y = make_data(n, noise, seed)
    return train_test_split(x, y, test_size=TEST_SIZE, random_state=seed)


def polynomial_features(x, degree):
    """Design matrix with columns x^1, ..., x^degree (shape n x degree)."""
    x = np.asarray(x, dtype=float).ravel()
    return x[:, None] ** np.arange(1, degree + 1)[None, :]


# ----------------------------------------------------------------------------------------------
# Scaling
# ----------------------------------------------------------------------------------------------
class Scaler:
    """Standardise the columns of X and centre y with statistics from the TRAINING data."""

    def __init__(self, scale=True):
        """Choose between standardization (scale=True) and centering only."""
        self.scale = scale

    def fit(self, X, y):
        """Store column means, ddof=0 standard deviations, and target mean for one fit."""
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0) if self.scale else np.ones(X.shape[1])
        self.X_std[self.X_std == 0.0] = 1.0  # guard against constant columns
        self.y_mean = float(np.mean(y))
        return self

    def transform(self, X):
        """Apply the stored training transform without fitting new statistics."""
        return (X - self.X_mean) / self.X_std

    def center(self, y):
        """Subtract the fitted training target mean; do not estimate a mean from this input."""
        return y - self.y_mean

    def original_coefficients(self, theta):
        """Map parameters of the scaled problem back to y = beta_0 + sum_j beta_j x^j."""
        beta = theta / self.X_std
        beta0 = self.y_mean - np.sum(beta * self.X_mean)
        return beta0, beta


# ----------------------------------------------------------------------------------------------
# Closed-form solutions (on already scaled/centred data)
# ----------------------------------------------------------------------------------------------
def ols_parameters(X, y):
    """OLS solution theta = X^+ y, with the pseudoinverse computed from the SVD of X."""
    return np.linalg.pinv(X) @ y


def ridge_parameters(X, y, lam):
    """Minimiser of (1/n)||y - X theta||^2 + lam ||theta||^2."""
    n = X.shape[0]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ ((s / (s**2 + n * lam)) * (U.T @ y))


def ridge_shrinkage_factors(X, lam):
    """Shrinkage factors s_j^2 / (s_j^2 + n lam) for the singular values s_j of X."""
    s = np.linalg.svd(X, compute_uv=False)
    return s**2 / (s**2 + X.shape[0] * lam)


def lasso_sklearn_parameters(X, y, lam, max_iter=200_000, tol=1e-10):
    """Reference Lasso solution from Scikit-Learn for our cost (1/n)||y-X theta||^2 + lam||theta||_1.
    """
    model = Lasso(alpha=lam / 2.0, fit_intercept=False, max_iter=max_iter, tol=tol)
    model.fit(X, y)
    return model.coef_.copy()


def checked_lasso_path(X, y, lambdas, tol=1e-6, gap_tol=1e-7, kkt_tol=1e-7,
                       max_iter=2_000_000, refinement_iter=10_000_000):
    """Coordinate-descent Lasso path that accepts only fits passing the duality-gap and
    stationarity checks; a failed fit is continued, an unresolved one raises.

    LLM-assisted: generated by OpenAI Codex, 5 October 2026.
    """
    X = np.asfortranarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    gap_scale = max(float(np.mean(y**2)), np.finfo(float).eps)
    solver_tol = min(tol, gap_tol / gap_scale)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        _, coefs, gaps, iterations = lasso_path(
            X, y, alphas=lambdas / 2, tol=solver_tol, max_iter=max_iter, return_n_iter=True)
    initial_limits = int(np.sum(np.asarray(iterations) >= max_iter))
    residuals, refined = [], 0
    for j, lam in enumerate(lambdas):
        # For nonzero slopes: g_j + lambda*sign(theta_j)=0; at zero: |g_j|<=lambda.
        smooth = 2 * X.T @ (X @ coefs[:, j] - y) / len(y)
        residual = np.where(coefs[:, j] == 0, np.maximum(np.abs(smooth) - lam, 0),
                            np.abs(smooth + lam * np.sign(coefs[:, j])))
        violation = float(np.max(residual, initial=0))
        if abs(gaps[j]) > gap_tol or violation > kkt_tol:
            refined += 1
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                _, theta, gap, extra = lasso_path(
                    X, y, alphas=np.array([lam / 2]), coef_init=coefs[:, j].copy(),
                    tol=min(solver_tol, 1e-10), max_iter=refinement_iter, return_n_iter=True)
            coefs[:, j], gaps[j] = theta[:, 0], gap[0]
            iterations[j] += extra[0]
            smooth = 2 * X.T @ (X @ coefs[:, j] - y) / len(y)
            residual = np.where(coefs[:, j] == 0, np.maximum(np.abs(smooth) - lam, 0),
                                np.abs(smooth + lam * np.sign(coefs[:, j])))
            violation = float(np.max(residual, initial=0))
        if not np.isfinite(violation) or not np.isfinite(gaps[j]) or abs(gaps[j]) > gap_tol or violation > kkt_tol:
            raise RuntimeError(f'Unconverged Lasso candidate: lambda={lam:g}, gap={gaps[j]:.3g}, KKT={violation:.3g}')
        residuals.append(violation)
    return coefs, {'total': len(lambdas), 'unresolved': 0, 'initial_hit_limit': initial_limits,
                   'refined': refined, 'max_dual_gap': float(np.max(np.abs(gaps))),
                   'max_kkt_residual': max(residuals), 'max_sweeps': int(max(iterations)),
                   'requested_solver_tol': tol, 'dual_gap_tolerance': gap_tol, 'kkt_tolerance': kkt_tol}


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def mse(y, y_pred):
    """Mean squared error."""
    y, y_pred = np.asarray(y).ravel(), np.asarray(y_pred).ravel()
    return np.mean((y - y_pred) ** 2)


def r2(y, y_pred):
    """Coefficient of determination R^2."""
    y, y_pred = np.asarray(y).ravel(), np.asarray(y_pred).ravel()
    return 1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)


# ----------------------------------------------------------------------------------------------
# Estimator
# ----------------------------------------------------------------------------------------------
class PolynomialRegression(BaseEstimator, RegressorMixin):
    """Polynomial regression in x with OLS, Ridge or Lasso and built-in train-only scaling.

    Follows the scikit-learn estimator interface, so cross_val_score and clone refit the
    scaler inside every training fold or bootstrap sample. solver(X, y, lam) can replace the
    closed-form fit, e.g. by a gradient-descent Lasso.
    """

    def __init__(self, degree=5, method="ols", lam=0.0, scale=True, solver=None):
        """Store the hyperparameters (scikit-learn estimator convention)."""
        self.degree = degree
        self.method = method
        self.lam = lam
        self.scale = scale
        self.solver = solver

    def fit(self, x, y):
        """Fit features, scaler and chosen solver using only the observations passed here."""
        X = polynomial_features(x, self.degree)
        self.scaler_ = Scaler(self.scale).fit(X, y)
        Xs = self.scaler_.transform(X)
        yc = self.scaler_.center(np.asarray(y, dtype=float).ravel())
        if self.solver is not None:
            self.theta_ = self.solver(Xs, yc, self.lam)
        elif self.method == "ols":
            self.theta_ = ols_parameters(Xs, yc)
        elif self.method == "ridge":
            self.theta_ = ridge_parameters(Xs, yc, self.lam)
        elif self.method == "lasso":
            self.theta_ = lasso_sklearn_parameters(Xs, yc, self.lam)
        else:
            raise ValueError(f"unknown method {self.method!r}")
        self.intercept_ = self.scaler_.y_mean
        return self

    def predict(self, x):
        """Predict with the training scaler and the fitted coefficients."""
        Xs = self.scaler_.transform(polynomial_features(x, self.degree))
        return self.intercept_ + Xs @ self.theta_
