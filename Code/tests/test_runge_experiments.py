"""Checks of the regression experiments: the benchmark of project 1 and the bookkeeping of the grids.

LLM-assisted (code level 2): the expected values follow the author's derivations and the course
reference; the scaffolding was drafted with Claude Opus 5.5 (Claude Code, November 2026).
"""

import numpy as np

import plot_style
import runge_experiments as rx
from regression import make_data


def test_polynomial_benchmark_reproduces_the_project_1_table(monkeypatch, tmp_path):
    """CV MSE and independent test MSE of Table IV in the project-1 report."""
    # the benchmark saves its results; send them to a temporary folder so the tests never change results/
    monkeypatch.setattr(plot_style, "RESULTS_DIR", tmp_path)
    results = rx.polynomial_benchmark()
    assert (tmp_path / "runge_polynomials.json").exists()
    expected_cv = {"OLS": 0.0173, "Ridge": 0.0173, "Lasso": 0.0192}
    expected_test = {"OLS": 0.0113, "Ridge": 0.0113, "Lasso": 0.0132}
    x_new, y_new = make_data(2000, rx.NOISE, rx.SEED + 1)
    for name, model in rx.polynomial_models().items():
        assert round(results[name]["cv_mse"], 4) == expected_cv[name]
        model.fit(rx.X_DATA, rx.Y_DATA)
        assert round(float(np.mean((model.predict(x_new[:, None]) - y_new) ** 2)), 4) == expected_test[name]


def test_cosine_data_keep_the_inputs_and_noise_of_the_runge_data():
    x, y = make_data(30, rx.NOISE, 7)
    cosine_targets = y - rx.runge(x) + rx.cosine(x)
    assert np.allclose(cosine_targets - rx.cosine(x), y - rx.runge(x))


def test_parameter_count():
    assert rx.n_parameters((50,)) == 151
    assert rx.n_parameters((100, 100)) == 10_401
    assert rx.n_parameters((), 784, 10) == 7850


def test_one_se_rule_takes_the_largest_penalty_then_the_smallest_rate():
    mse = np.array([[0.020, 0.0205, 0.030],
                    [0.019, 0.0195, 0.040]])
    se = np.full(mse.shape, 0.002)
    cell = rx._one_se_cell(mse, se, lambdas=[0.0, 1e-3, 1e-2], gammas=[1e-3, 1e-2])
    assert cell["minimum"] == (1, 0)
    assert cell["one_se"] == (0, 1)  # lambda 1e-3 is the largest inside the band, at the smaller rate


def test_ridge_path_equals_ols_without_penalty_and_shrinks_with_penalty():
    x, y = make_data(80, 0.1, 11)
    x_new = np.linspace(-0.9, 0.9, 20)
    path = rx._ridge_path(x, y, x_new, 6, np.array([0.0, 1e3]))
    ols = rx.PolynomialRegression(6, "ols").fit(x, y).predict(x_new)
    assert np.allclose(path[0], ols, atol=1e-8)
    assert np.allclose(path[1], y.mean(), atol=1e-2)  # a huge penalty leaves only the mean


def test_cross_validation_never_sees_the_test_observations():
    """The data used for all selection is the 100-point set, disjoint from the 2000 test points."""
    x_new, _ = make_data(2000, rx.NOISE, rx.SEED + 1)
    assert len(rx.X_DATA) == 100
    assert not np.intersect1d(rx.X_DATA[:, 0], x_new).size
