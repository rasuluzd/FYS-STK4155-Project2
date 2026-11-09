"""
Regression experiments on Runge's function. Every function runs one experiment and writes its numbers
to results/runge_<name>.json, from which figures.py draws the figures.

Data: the 100 noisy points of project 1 (seed 2026). All networks and polynomials are compared on the
same ten folds, with the scaler fitted inside each fold. The two sets of 2000 independent points (seeds
2027 and 2028) are only used by final_models, after every choice has been made on the 100 points; the
first is also read by a test that reproduces the polynomial test errors of project 1.

Run all with `python runge_experiments.py`, or a few with `python runge_experiments.py overfitting`.

LLM-assisted
------------
Code level 2. Tool: Claude Opus 5.5 (Anthropic, Claude Code desktop app, November 2026).
Role: the grids, the folds, the one-standard-error rule and the protocol (same folds, scaler in
the fold, heat maps with the one-standard-error rule, test set used once) are the author's and
follow the course material; Claude drafted the joblib parallelisation and small helpers
around them.
Verification: the polynomial results reproduce the numbers of the project-1 report, see
tests/test_runge_experiments.py.
OpenAI Codex also refactored the functions marked below for readability in November 2026.
"""

import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from estimators import NeuralNetRegressor
from plot_style import load_results, save_results
from regression import (N_POINTS, NOISE, SEED, PolynomialRegression, checked_lasso_path, make_data,
                        polynomial_features, runge)

N_JOBS = 14
BATCH = 16
X_DATA, Y_DATA = make_data(N_POINTS, NOISE, SEED)
X_DATA = X_DATA[:, None]
FOLDS = KFold(10, shuffle=True, random_state=SEED)
REQUIRED_ARCHITECTURES = [(50,), (100,), (50, 50), (100, 100)]
X_PLOT = np.linspace(-1.25, 1.25, 1001)[:, None]  # also shows what happens outside the data


def n_parameters(hidden, n_in=1, n_out=1):
    """Number of weights and biases of a dense network. LLM-assisted: Claude."""
    sizes = [n_in, *hidden, n_out]
    return int(sum((a + 1) * b for a, b in zip(sizes[:-1], sizes[1:])))


def network_pipeline(**settings):
    """Scaler inside the fold, then the network. LLM-assisted: Claude."""
    return make_pipeline(StandardScaler(), NeuralNetRegressor(**settings))


def grid_search(grid, **fixed):
    """Cross-validate the network over a grid on the ten folds; axes follow the sorted keys of grid.

    Returns the CV MSE, its standard error s/sqrt(k), the training MSE, the fit time and the error
    of every fold.
    """
    grid = dict(sorted(grid.items()))
    search = GridSearchCV(network_pipeline(**fixed), {f"neuralnetregressor__{k}": v for k, v in grid.items()},
                          cv=FOLDS, scoring="neg_mean_squared_error", return_train_score=True, refit=False,
                          n_jobs=N_JOBS)
    search.fit(X_DATA, Y_DATA)
    r = search.cv_results_
    shape = tuple(len(v) for v in grid.values())
    k = FOLDS.get_n_splits()
    return {"axes": list(grid), "values": list(grid.values()),
            "mse": (-r["mean_test_score"]).reshape(shape),
            "se": (r["std_test_score"] / np.sqrt(k - 1)).reshape(shape),  # std_test_score has ddof=0
            "train": (-r["mean_train_score"]).reshape(shape),
            "fit_time": r["mean_fit_time"].reshape(shape),
            "folds": -np.array([r[f"split{j}_test_score"] for j in range(k)]).reshape(k, *shape)}


def checkpoints(epochs, count=40):
    """Roughly logarithmic list of the epochs at which curves are recorded."""
    return np.unique(np.round(np.geomspace(1, epochs, count)).astype(int)).tolist()


def _fold_curve(settings, train_index, val_index, marks):
    """Training and validation MSE, and the distance ||theta - theta_0|| that training has moved the
    parameters, at the epochs in marks, for one fold. LLM-assisted: Claude.

    LLM-assisted: OpenAI Codex, readability refactor in November 2026.
    """
    scaler = StandardScaler().fit(X_DATA[train_index])
    X_tr, X_val = scaler.transform(X_DATA[train_index]), scaler.transform(X_DATA[val_index])
    y_tr, y_val = Y_DATA[train_index], Y_DATA[val_index]
    model = NeuralNetRegressor(**settings)
    wanted = set(marks)
    log = []
    epoch = 0

    def record(theta):
        """LLM-assisted: OpenAI Codex simplified the epoch counter."""
        nonlocal epoch
        epoch += 1
        if epoch in wanted:  # model.net_.theta holds the initial parameters until training ends
            log.append((np.mean((model.predict(X_tr, theta) - y_tr) ** 2),
                        np.mean((model.predict(X_val, theta) - y_val) ** 2),
                        np.linalg.norm(theta - model.net_.theta)))

    start = time.perf_counter()
    model.fit(X_tr, y_tr, on_epoch=record)
    elapsed = time.perf_counter() - start
    log += [(1e12, 1e12, np.nan)] * (len(marks) - len(log))  # a diverged run stops early
    return np.array(log).T, elapsed


def cv_curves(configurations):
    """Learning curves on the ten folds for a list of settings dicts. LLM-assisted: Claude.

    LLM-assisted: OpenAI Codex, readability refactor in November 2026.
    """
    folds = list(FOLDS.split(X_DATA))
    marks = [checkpoints(c["epochs"]) for c in configurations]
    runs = Parallel(n_jobs=N_JOBS)(delayed(_fold_curve)(c, tr, va, m)
                                   for c, m in zip(configurations, marks) for tr, va in folds)
    out = []
    for i, (settings, epochs) in enumerate(zip(configurations, marks)):
        part = runs[i * len(folds):(i + 1) * len(folds)]
        train = np.array([p[0][0] for p in part])
        val = np.array([p[0][1] for p in part])
        distance = np.array([p[0][2] for p in part]).mean(axis=0)
        mean_train = train.mean(axis=0)
        mean_val = val.mean(axis=0)
        val_se = val.std(axis=0, ddof=1) / np.sqrt(len(folds))
        best = int(np.argmin(mean_val))
        out.append({
            "settings": settings, "epochs": epochs,
            "train": mean_train, "val": mean_val, "distance": distance,
            "best_distance": distance[best], "val_se": val_se,
            "fit_time": np.mean([p[1] for p in part]),
            "best_epoch": epochs[best], "best_val": mean_val[best],
            "best_val_se": val_se[best], "best_train": mean_train[best],
            "best_folds": val[:, best],
            "last_val": mean_val[-1], "last_val_se": val_se[-1],
            "last_train": mean_train[-1], "last_folds": val[:, -1],
        })
    return out


# ----------------------------------------------------------------------------------------------
# The polynomial benchmark of project 1
# ----------------------------------------------------------------------------------------------
LASSO_PATH = np.logspace(-1, -4, 7)  # the penalty path of project 1, ending at the selected value


def _lasso_end_of_path(Xs, yc, lam):
    """Lasso solution at the end of the warm-started path of project 1."""
    coefficients, _ = checked_lasso_path(Xs, yc, LASSO_PATH)
    return coefficients[:, -1]


def polynomial_models():
    """The models project 1 selected by ten-fold cross-validation."""
    return {"OLS": PolynomialRegression(12, "ols"),
            "Ridge": PolynomialRegression(12, "ridge", 1e-10),
            "Lasso": PolynomialRegression(15, "lasso", 1e-4, solver=_lasso_end_of_path)}


def polynomial_benchmark():
    """Cross-validated errors of the selected polynomial models, on the same folds as the networks."""
    results = {}
    for name, model in polynomial_models().items():
        cv = cross_validate(model, X_DATA, Y_DATA, cv=FOLDS, scoring="neg_mean_squared_error",
                            return_train_score=True)
        folds = -cv["test_score"]
        results[name] = {"cv_mse": folds.mean(), "cv_se": folds.std(ddof=1) / np.sqrt(len(folds)),
                         "train_mse": -cv["train_score"].mean(), "fit_time": cv["fit_time"].mean(),
                         "folds": folds, "degree": model.degree, "lam": model.lam,
                         "n_parameters": model.degree + 1}
    save_results("runge_polynomials", results)
    return results


# ----------------------------------------------------------------------------------------------
# Learning rates, optimizers and minibatches
# ----------------------------------------------------------------------------------------------
GAMMAS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
SCAN_EPOCHS = 10_000


def learning_rate_scan():
    """Learning curves of the 1-50-1 sigmoid network for four optimizers and twelve learning rates.

    All runs get the same budget of 10 000 passes over the data.

    LLM-assisted: OpenAI Codex, readability refactor in November 2026.
    """
    configurations = []
    for optimizer in ("gd", "sgd", "rmsprop", "adam"):
        for gamma in GAMMAS:
            configurations.append(dict(hidden=(50,), activation="sigmoid", optimizer=optimizer,
                                       gamma=gamma, epochs=SCAN_EPOCHS, batch_size=BATCH))
    runs = cv_curves(configurations)
    save_results("runge_learning_rate", {"gammas": GAMMAS, "runs": runs})
    return runs


def batch_scan():
    """Heat maps of the CV MSE over the batch size and the learning rate."""
    grid = {"batch_size": [4, 8, 16, 32, 90], "gamma": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]}
    results = {o: grid_search(grid, hidden=(50,), activation="sigmoid", optimizer=o, epochs=3000)
               for o in ("sgd", "adam")}
    save_results("runge_batch", results)
    return results


# ----------------------------------------------------------------------------------------------
# The architectures the comparison with OLS asks for
# ----------------------------------------------------------------------------------------------
ARCH_GAMMAS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
ARCH_EPOCHS = 10_000


def required_architectures():
    """One and two hidden layers of 50 and 100 sigmoid nodes, with Adam, learning rate by CV.

    The epoch of the lowest mean CV error gives the early-stopped model.

    LLM-assisted: OpenAI Codex, readability refactor in November 2026.
    """
    configurations = []
    for hidden in REQUIRED_ARCHITECTURES:
        for gamma in ARCH_GAMMAS:
            configurations.append(dict(hidden=hidden, activation="sigmoid", optimizer="adam",
                                       gamma=gamma, epochs=ARCH_EPOCHS, batch_size=BATCH))
    save_results("runge_architectures", {"runs": cv_curves(configurations)})


# ----------------------------------------------------------------------------------------------
# Activation functions, depth and width
# ----------------------------------------------------------------------------------------------
DEPTHS = [1, 2, 3, 4, 5]
WIDTHS = [5, 10, 25, 50, 100]
MAP_GAMMAS = {"adam": [1e-3, 3e-3, 1e-2], "sgd": [1e-2, 3e-2, 1e-1]}
MAP_EPOCHS = 3000


def activation_maps():
    """CV and training MSE over depth and width for three activations and two optimizers.

    Each cell is the best of three learning rates, so that the maps compare what an architecture can
    do and not one fixed rate that happens to suit it.
    """
    hidden = [(w,) * d for d in DEPTHS for w in WIDTHS]
    results = {"depths": DEPTHS, "widths": WIDTHS, "gammas": MAP_GAMMAS}
    for optimizer, gammas in MAP_GAMMAS.items():
        for activation in ("sigmoid", "relu", "leaky_relu"):
            print("activation map", optimizer, activation, flush=True)
            results[f"{optimizer}_{activation}"] = grid_search({"gamma": gammas, "hidden": hidden},
                                                               activation=activation, optimizer=optimizer,
                                                               epochs=MAP_EPOCHS, batch_size=BATCH)
    save_results("runge_activation_maps", results)


def gradient_flow(depth=8, width=50, repeats=30):
    """Size of the weight gradient in every layer of an untrained deep network, over many seeds."""
    X = StandardScaler().fit_transform(X_DATA)
    target = (Y_DATA - Y_DATA.mean())[:, None]
    results = {"depth": depth, "width": width}
    for label, activation, init in (("sigmoid", "sigmoid", "auto"), ("sigmoid_unit", "sigmoid", "unit"),
                                    ("relu", "relu", "auto"), ("leaky_relu", "leaky_relu", "auto")):
        norms = []
        for seed in range(repeats):
            model = NeuralNetRegressor(hidden=(width,) * depth, activation=activation, init=init, epochs=0,
                                       seed=SEED + seed).fit(X, Y_DATA)
            norms.append(model.net_.layer_gradient_norms(X, target))
        results[label] = {"median": np.median(norms, axis=0), "q25": np.quantile(norms, 0.25, axis=0),
                          "q75": np.quantile(norms, 0.75, axis=0)}
    save_results("runge_gradient_flow", results)


def overfitting():
    """Training and CV error against epoch and against size, with no penalty and no early stopping.

    LLM-assisted: OpenAI Codex, readability refactor in November 2026.
    """
    families = {"sigmoid": ("sigmoid", 3e-3), "relu": ("relu", 3e-3)}
    hidden = [(2,), (5,), (10,), (25,), (50,), (100,), (50, 50), (100, 100), (100, 100, 100)]
    configurations = []
    for activation, gamma in families.values():
        for layers in hidden:
            configurations.append(dict(hidden=layers, activation=activation, optimizer="adam",
                                       gamma=gamma, epochs=30_000, batch_size=BATCH))
    runs = cv_curves(configurations)
    for run in runs:
        run["n_parameters"] = n_parameters(run["settings"]["hidden"])
    save_results("runge_overfitting", {"runs": runs})


# ----------------------------------------------------------------------------------------------
# Penalties
# ----------------------------------------------------------------------------------------------
PENALTY_GAMMAS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
PENALTY_LAMBDAS = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
PENALTY_EPOCHS = 3000


def _one_se_cell(mse, se, lambdas, gammas):
    """Minimum of a (gamma, lambda) map and the one-standard-error choice.

    Among the cells within one standard error of the minimum: the largest penalty, then the smallest
    learning rate.
    """
    i, j = np.unravel_index(np.argmin(mse), mse.shape)
    within = mse <= mse[i, j] + se[i, j]
    cells = np.argwhere(within)
    i_one, j_one = max(cells.tolist(), key=lambda c: (lambdas[c[1]], -gammas[c[0]]))
    return {"minimum": (int(i), int(j)), "one_se": (int(i_one), int(j_one)), "within": within}


def penalty_scan():
    """CV over (gamma, lambda) for L2 and L1 at the four architectures."""
    results = {"gammas": PENALTY_GAMMAS, "lambdas": PENALTY_LAMBDAS, "architectures": REQUIRED_ARCHITECTURES}
    for penalty in ("l2", "l1"):
        for hidden in REQUIRED_ARCHITECTURES:
            print("penalty", penalty, hidden, flush=True)
            r = grid_search({"gamma": PENALTY_GAMMAS, "lam": PENALTY_LAMBDAS}, hidden=hidden,
                            activation="sigmoid", optimizer="adam", penalty=penalty, epochs=PENALTY_EPOCHS,
                            batch_size=BATCH)
            r.update(_one_se_cell(r["mse"], r["se"], PENALTY_LAMBDAS, PENALTY_GAMMAS))
            results[f"{penalty}_{'_'.join(map(str, hidden))}"] = r
    save_results("runge_penalties", results)


def bias_penalty_check():
    """Does it matter whether the biases enter the penalty? Best cell of each map, both ways."""
    maps = load_results("runge_penalties")
    results = {}
    for penalty in ("l2", "l1"):
        key, best = min(((k, m["mse"][m["minimum"][0]][m["minimum"][1]]) for k, m in maps.items()
                         if k.startswith(penalty)), key=lambda t: t[1])
        m = maps[key]
        i, j = m["one_se"]
        hidden = tuple(int(v) for v in key.split("_")[1:])
        settings = dict(hidden=hidden, activation="sigmoid", optimizer="adam", penalty=penalty,
                        gamma=maps["gammas"][i], lam=maps["lambdas"][j], epochs=PENALTY_EPOCHS, batch_size=BATCH)
        both = grid_search({"penalize_biases": [True, False]}, **settings)
        results[penalty] = {"hidden": hidden, "gamma": settings["gamma"], "lam": settings["lam"],
                            "with_biases": both["mse"][0], "weights_only": both["mse"][1], "se": both["se"][0]}
    save_results("runge_bias_penalty", results)


# ----------------------------------------------------------------------------------------------
# Why the network does well: approximation without noise, and many data sets
# ----------------------------------------------------------------------------------------------
def _noise_free_network(k, seed, X, target):
    """Train a 1-k-1 sigmoid network on noise-free samples; returns the MSE against f."""
    model = NeuralNetRegressor(hidden=(k,), activation="sigmoid", optimizer="adam", gamma=3e-3,
                               epochs=60_000, batch_size=len(X), seed=seed).fit(X, target)
    return float(np.mean((model.predict(X) - target) ** 2))


def cosine(x):
    """A smooth function without poles, for which polynomials are a natural basis."""
    return np.cos(3.0 * x)


FUNCTIONS = {"runge": runge, "cosine": cosine}


def approximation_power():
    """Noise-free L2 error against the number of parameters, for polynomials and sigmoid networks.

    Polynomials: least squares in the Chebyshev basis, which stays well conditioned at high degree.
    Networks: best of five initializations.
    """
    x = np.linspace(-1.0, 1.0, 2001)
    degrees = list(range(1, 41))
    X = np.linspace(-1.0, 1.0, 401)[:, None]
    nodes = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50]
    rho = 0.2 + np.sqrt(1.04)  # parameter of the Bernstein ellipse that reaches the poles of f at +-i/5
    results = {"degrees": degrees, "nodes": nodes, "n_parameters": [3 * k + 1 for k in nodes], "rho": rho}
    for name, function in FUNCTIONS.items():
        f = function(x)
        polynomial = [float(np.mean((np.polynomial.chebyshev.chebval(x, np.polynomial.chebyshev.chebfit(x, f, d)) - f) ** 2))
                      for d in degrees]
        target = function(X[:, 0])
        runs = Parallel(n_jobs=N_JOBS)(delayed(_noise_free_network)(k, SEED + s, X, target) for k in nodes for s in range(5))
        errors = np.array(runs).reshape(len(nodes), 5)
        results[name] = {"polynomial": polynomial, "network": errors.min(axis=1), "network_all": errors}
    save_results("runge_approximation", results)


def _best_polynomial_error(x, y, x_eval, f_eval):
    """Ridge path including lambda = 0, chosen by ten-fold CV over degree and penalty; error against f."""
    lambdas = np.r_[0.0, np.logspace(-10, 0, 11)]
    degrees = np.arange(1, 21)
    scores = np.zeros((len(degrees), len(lambdas)))
    for tr, va in KFold(10, shuffle=True, random_state=SEED).split(x):
        for a, d in enumerate(degrees):
            scores[a] += ((_ridge_path(x[tr], y[tr], x[va], d, lambdas) - y[va][None, :]) ** 2).mean(axis=1)
    a, b = np.unravel_index(np.argmin(scores), scores.shape)
    prediction = _ridge_path(x, y, x_eval, degrees[a], lambdas[b:b + 1])[0]
    return float(np.mean((prediction - f_eval) ** 2)), int(degrees[a]), float(lambdas[b])


def _ridge_path(x_tr, y_tr, x_new, degree, lambdas):
    """Predictions at x_new for every penalty, with scaling from the training part."""
    A = polynomial_features(x_tr, degree)
    mean, std = A.mean(axis=0), A.std(axis=0)
    U, s, Vt = np.linalg.svd((A - mean) / std, full_matrices=False)
    uty = U.T @ (y_tr - y_tr.mean())
    A_new = (polynomial_features(x_new, degree) - mean) / std
    with np.errstate(divide="ignore", invalid="ignore"):
        shrink = np.where(s[None, :] > 1e-15 * s[0], s[None, :] / (s[None, :] ** 2 + len(x_tr) * lambdas[:, None]), 0.0)
    return y_tr.mean() + (shrink * uty[None, :]) @ Vt @ A_new.T


def _one_data_set(name, n, replicate, settings, updates):
    """One fresh data set: best polynomial and the fixed network, both scored against the function.

    The network is trained twice, with the base number of updates and with ten times as many.
    """
    function = FUNCTIONS[name]
    x, y = make_data(n, NOISE, 7_000_000 + 1000 * n + replicate)
    y = y - runge(x) + function(x)  # the same inputs and noise for both functions
    x_eval = np.linspace(-1.0, 1.0, 2001)
    f_eval = function(x_eval)
    polynomial, degree, lam = _best_polynomial_error(x, y, x_eval, f_eval)
    per_epoch = int(np.ceil(n / settings["batch_size"]))
    errors = []
    for factor in (1, 10):
        model = network_pipeline(**settings, epochs=max(1, int(np.ceil(factor * updates / per_epoch))),
                                 seed=SEED + replicate).fit(x[:, None], y)
        errors.append(float(np.mean((model.predict(x_eval[:, None]) - f_eval) ** 2)))
    return polynomial, errors[0], errors[1], degree, lam


def sample_size_study(sizes=(50, 100, 200, 500, 1000), replicates=40):
    """Best polynomial against one fixed network over many fresh data sets of each size.

    The network settings come from the CV on the main data set and are not retuned. The number of
    updates is kept constant, so all sizes get the same optimization effort.
    """
    runs = load_results("runge_architectures")["runs"]
    fifty = [r for r in runs if r["settings"]["hidden"] == [50]]
    best = min(fifty, key=lambda r: r["best_val"])
    settings = dict(hidden=(50,), activation="sigmoid", optimizer="adam", gamma=best["settings"]["gamma"], batch_size=BATCH)
    updates = best["best_epoch"] * int(np.ceil(90 / BATCH))
    results = {"sizes": sizes, "settings": {**settings, "updates": updates}}
    for name in FUNCTIONS:
        out = Parallel(n_jobs=N_JOBS)(delayed(_one_data_set)(name, n, r, settings, updates) for n in sizes for r in range(replicates))
        out = np.array(out, dtype=float).reshape(len(sizes), replicates, 5)
        results[name] = {"polynomial": out[:, :, 0], "network": out[:, :, 1], "network_long": out[:, :, 2],
                         "degree": out[:, :, 3], "lam": out[:, :, 4]}
    save_results("runge_sample_size", results)


# ----------------------------------------------------------------------------------------------
# Final evaluation on the independent observations
# ----------------------------------------------------------------------------------------------
def final_configurations():
    """The models carried to the test: the CV choice for each question asked."""
    rows = {}
    runs = load_results("runge_architectures")["runs"]
    for hidden in REQUIRED_ARCHITECTURES:
        family = [r for r in runs if tuple(r["settings"]["hidden"]) == hidden]
        best = min(family, key=lambda r: r["best_val"])
        rows["-".join(map(str, hidden)) + " sigmoid"] = {
            "settings": dict(hidden=hidden, activation="sigmoid", optimizer="adam",
                             gamma=best["settings"]["gamma"], epochs=best["best_epoch"], batch_size=BATCH),
            "cv_mse": best["best_val"], "cv_se": best["best_val_se"], "train_mse": best["best_train"],
            "folds": best["best_folds"]}
    small = load_results("runge_overfitting")["runs"]
    for nodes in (2, 5):  # the smallest networks within one standard error of the best CV error
        u = min((r for r in small if r["settings"]["hidden"] == [nodes] and r["settings"]["activation"] == "sigmoid"),
                key=lambda r: r["best_val"])
        rows[f"{nodes} sigmoid"] = {
            "settings": dict(hidden=(nodes,), activation="sigmoid", optimizer="adam", gamma=u["settings"]["gamma"],
                             epochs=u["best_epoch"], batch_size=BATCH),
            "cv_mse": u["best_val"], "cv_se": u["best_val_se"], "train_mse": u["best_train"], "folds": u["best_folds"]}
    maps = load_results("runge_penalties")
    for penalty in ("l2", "l1"):
        key = min((k for k in maps if k.startswith(penalty)),
                  key=lambda k: maps[k]["mse"][maps[k]["minimum"][0]][maps[k]["minimum"][1]])
        m = maps[key]
        i, j = m["one_se"]
        rows[f"{penalty} one-se"] = {
            "settings": dict(hidden=tuple(int(v) for v in key.split("_")[1:]), activation="sigmoid", optimizer="adam",
                             penalty=penalty, gamma=maps["gammas"][i], lam=maps["lambdas"][j],
                             epochs=PENALTY_EPOCHS, batch_size=BATCH),
            "cv_mse": m["mse"][i][j], "cv_se": m["se"][i][j], "train_mse": m["train"][i][j],
            "folds": np.array(m["folds"])[:, i, j]}
    activation = load_results("runge_activation_maps")
    for name in ("relu", "leaky_relu"):
        m = activation[f"adam_{name}"]
        mse = np.array(m["mse"])
        g, h = np.unravel_index(np.argmin(mse), mse.shape)
        hidden = tuple(m["values"][1][h])
        rows[f"best {name}"] = {
            "settings": dict(hidden=hidden, activation=name, optimizer="adam", gamma=m["values"][0][g],
                             epochs=MAP_EPOCHS, batch_size=BATCH),
            "cv_mse": mse[g, h], "cv_se": m["se"][g][h], "train_mse": m["train"][g][h],
            "folds": np.array(m["folds"])[:, g, h]}
    return rows


def _test_network(settings, seed, X_new, y_new, f_new, X_fresh, y_fresh):
    """Refit on the 100 points with one initialization and score on the independent data."""
    start = time.perf_counter()
    model = network_pipeline(**settings, seed=seed).fit(X_DATA, Y_DATA)
    elapsed = time.perf_counter() - start
    prediction = model.predict(X_new)
    return (float(np.mean((prediction - y_new) ** 2)), float(np.mean((prediction - f_new) ** 2)),
            float(np.mean((model.predict(X_DATA) - Y_DATA) ** 2)), elapsed, model.predict(X_PLOT),
            float(np.mean((model.predict(X_fresh) - y_fresh) ** 2)))


def final_models(n_initializations=10):
    """Refit every selected model on all 100 points and score it on the 2000 independent observations.

    This is the only function that reads those observations. Seed 2027 is the test set of project 1. A
    few pilot runs, which were not saved, printed errors on it while the grids were being sized, so a
    second set with seed 2028, which no other code reads, checks the result.
    """
    x_new, y_new = make_data(2000, NOISE, SEED + 1)
    X_new, f_new = x_new[:, None], runge(x_new)
    x_fresh, y_fresh = make_data(2000, NOISE, SEED + 2)
    X_fresh = x_fresh[:, None]
    polynomials = load_results("runge_polynomials")
    results = {"polynomial": {}, "network": {}}
    for name, model in polynomial_models().items():
        start = time.perf_counter()
        model.fit(X_DATA, Y_DATA)
        elapsed = time.perf_counter() - start
        prediction = model.predict(X_new)
        results["polynomial"][name] = {**polynomials[name], "test_mse": np.mean((prediction - y_new) ** 2),
                                       "true_mse": np.mean((prediction - f_new) ** 2), "seconds": elapsed,
                                       "plot": model.predict(X_PLOT), "fresh_mse": np.mean((model.predict(X_fresh) - y_fresh) ** 2)}
    for name, row in final_configurations().items():
        print("final", name, flush=True)
        runs = Parallel(n_jobs=N_JOBS)(delayed(_test_network)(row["settings"], SEED + s, X_new, y_new, f_new, X_fresh, y_fresh)
                                       for s in range(n_initializations))
        test = np.array([r[0] for r in runs])
        true = np.array([r[1] for r in runs])
        results["network"][name] = {**row, "n_parameters": n_parameters(row["settings"]["hidden"]),
                                    "fresh_mse": np.mean([r[5] for r in runs]),
                                    "test_mse": test.mean(), "test_std": test.std(ddof=1),
                                    "true_mse": true.mean(), "true_std": true.std(ddof=1),
                                    "train_all": np.mean([r[2] for r in runs]), "seconds": np.mean([r[3] for r in runs]),
                                    "plot_first": runs[0][4], "plot_all": [r[4] for r in runs]}
    best = results["network"]["50-50 sigmoid"]["settings"]
    start = time.perf_counter()
    library = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=best["hidden"], activation="logistic", solver="adam", alpha=0.0, learning_rate_init=best["gamma"],
        batch_size=best["batch_size"], max_iter=best["epochs"], tol=0.0, n_iter_no_change=10**9,
        random_state=SEED)).fit(X_DATA, Y_DATA)
    results["library"] = {"seconds": time.perf_counter() - start,
                          "test_mse": np.mean((library.predict(X_new) - y_new) ** 2),
                          "true_mse": np.mean((library.predict(X_new) - f_new) ** 2),
                          "hidden": best["hidden"]}
    results["x_plot"] = X_PLOT[:, 0]
    results["noise_variance_test"] = np.mean((y_new - f_new) ** 2)
    save_results("runge_final", results)
    return results


def features():
    """What the network has learned: output weight times hidden output, against the polynomial terms.

    The smallest network the one-standard-error rule allows is refitted on all 100 points, so that every
    node can be drawn.
    """
    architectures = load_results("runge_overfitting")["runs"]
    small = min((r for r in architectures if r["settings"]["hidden"] == [2] and r["settings"]["activation"] == "sigmoid"),
                key=lambda r: r["best_val"])
    scaler = StandardScaler().fit(X_DATA)
    model = NeuralNetRegressor(hidden=(2,), activation="sigmoid", optimizer="adam",
                               gamma=small["settings"]["gamma"], epochs=small["best_epoch"], batch_size=BATCH,
                               seed=SEED).fit(scaler.transform(X_DATA), Y_DATA)
    X_scaled = scaler.transform(X_PLOT)
    a = model.net_.hidden_outputs(X_scaled)[0]
    W_out, b_out = model.net_.layers(model.net_.theta)[-1]
    contributions = a * W_out[:, 0]
    ols = polynomial_models()["OLS"].fit(X_DATA, Y_DATA)
    A = (polynomial_features(X_PLOT[:, 0], 12) - ols.scaler_.X_mean) / ols.scaler_.X_std
    save_results("runge_features", {"x": X_PLOT[:, 0], "network_terms": contributions.T, "network_constant": float(b_out[0]) + model.y_mean_,
                                    "network_sum": model.predict(X_scaled), "polynomial_terms": (A * ols.theta_).T,
                                    "polynomial_constant": ols.intercept_, "polynomial_sum": ols.predict(X_PLOT),
                                    "hidden": (2,), "epochs": small["best_epoch"], "gamma": small["settings"]["gamma"],
                                    "weights": model.net_.layers(model.net_.theta)[0][0], "biases": model.net_.layers(model.net_.theta)[0][1],
                                    # what an L2 penalty would charge for these weights, against the data term of the cost
                                    "squared_norm": float(np.sum(model.net_.theta ** 2)),
                                    "half_train_mse": float(0.5 * np.mean((model.predict(scaler.transform(X_DATA)) - Y_DATA) ** 2))})


def sparsity():
    """Size of the weights after training without penalty, with L2 and with L1, on all 100 points."""
    maps = load_results("runge_penalties")
    results = {}
    settings = dict(hidden=(50,), activation="sigmoid", optimizer="adam", gamma=3e-3, epochs=PENALTY_EPOCHS, batch_size=BATCH)
    X = StandardScaler().fit_transform(X_DATA)
    for label, penalty, lam in (("none", None, 0.0), ("l2", "l2", 1e-3), ("l1", "l1", 1e-3)):
        weights = []
        for seed in range(10):
            model = NeuralNetRegressor(penalty=penalty, lam=lam, seed=SEED + seed, **settings).fit(X, Y_DATA)
            weights.append(np.sort(np.abs(model.net_.layers(model.net_.theta)[-1][0][:, 0]))[::-1])
        results[label] = {"median": np.median(weights, axis=0), "train_mse": float(np.mean((model.predict(X) - Y_DATA) ** 2))}
    results["lambdas_in_scan"] = maps["lambdas"]
    save_results("runge_sparsity", results)


def _layer_changes(settings, seed, marks):
    """Relative change of each layer's weights at the epochs in marks, on all 100 points, and the final
    training MSE.

    LLM-assisted: OpenAI Codex, readability refactor in November 2026.
    """
    X = StandardScaler().fit_transform(X_DATA)
    model = NeuralNetRegressor(**settings, seed=seed)
    wanted = set(marks)
    log = []
    epoch = 0

    def record(theta):
        """LLM-assisted: OpenAI Codex simplified the epoch counter."""
        nonlocal epoch
        epoch += 1
        if epoch in wanted:
            log.append(model.net_.layer_changes(theta, model.net_.theta))

    model.fit(X, Y_DATA, on_epoch=record)
    log += [[np.nan] * (len(settings["hidden"]) + 1)] * (len(marks) - len(log))  # a diverged run stops early
    return log, float(np.mean((model.predict(X) - Y_DATA) ** 2))


def mechanisms(seeds=10):
    """Measurements behind the explanations in the report, on the 100 observations only.

    (1) How far training moves the parameters from their initial values, next to the CV error, for the
    sigmoid networks of overfitting(), rerun with the same settings and folds. (2) How much plain SGD and
    Adam change the weights of each layer of a sigmoid network with five hidden layers of 50 nodes, at each
    learning rate of the activation maps.
    """
    hidden = [tuple(r["settings"]["hidden"]) for r in load_results("runge_overfitting")["runs"]
              if r["settings"]["activation"] == "sigmoid"]
    distance = cv_curves([dict(hidden=h, activation="sigmoid", optimizer="adam", gamma=3e-3, epochs=30_000,
                               batch_size=BATCH) for h in hidden])
    for run in distance:
        run["n_parameters"] = n_parameters(run["settings"]["hidden"])
    deep, marks = (50,) * 5, checkpoints(MAP_EPOCHS)
    changes = []
    for optimizer, gammas in MAP_GAMMAS.items():
        for gamma in gammas:
            settings = dict(hidden=deep, activation="sigmoid", optimizer=optimizer, gamma=gamma, epochs=MAP_EPOCHS,
                            batch_size=BATCH)
            out = Parallel(n_jobs=N_JOBS)(delayed(_layer_changes)(settings, SEED + s, marks) for s in range(seeds))
            changes.append({"optimizer": optimizer, "gamma": gamma, "epochs": marks,
                            "median": np.median([o[0] for o in out], axis=0), "train_mse": [o[1] for o in out]})
    save_results("runge_mechanisms", {"distance": distance, "hidden": deep, "layer_changes": changes})


EXPERIMENTS = {"polynomial_benchmark": polynomial_benchmark, "learning_rate_scan": learning_rate_scan,
               "batch_scan": batch_scan, "required_architectures": required_architectures,
               "activation_maps": activation_maps, "gradient_flow": gradient_flow, "overfitting": overfitting,
               "penalty_scan": penalty_scan, "bias_penalty_check": bias_penalty_check,
               "approximation_power": approximation_power, "sample_size_study": sample_size_study,
               "final_models": final_models, "features": features, "sparsity": sparsity, "mechanisms": mechanisms}

if __name__ == "__main__":
    import runge_experiments  # functions from a named module can be sent to the worker processes

    for name in sys.argv[1:] or EXPERIMENTS:
        start_time = time.time()
        runge_experiments.EXPERIMENTS[name]()
        print(f"{name} finished in {time.time() - start_time:.0f} s", flush=True)
