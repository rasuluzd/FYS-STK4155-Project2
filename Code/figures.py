"""
All figures of the report, drawn from the saved results in results/*.json and written as PDF to
report/figures/. Run `python figures.py`, or `python figures.py learning_rate` for a single figure.

Heat maps use one rule everywhere: darker is better, a square marks the best cell, a circle the choice
of the one-standard-error rule, and bold numbers are within one standard error of the best cell.

LLM-assisted
------------
Code level 2. Tool: Claude Opus 5.5 (Anthropic, Claude Code desktop app, November 2026).
Role: the panels, and what each figure shows, are the author's choices; Claude drafted small
helpers among the definitions: the tick labels, the annotated heat map and the panel letters.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.patches import Circle, Rectangle

from plot_style import BLUES, COLORS, DOUBLE, GREY, INK, SINGLE, load_results, ordered_colors, save, set_style

NOISE_VARIANCE = 0.01
OPTIMIZER_STYLE = {"gd": ("Plain GD", COLORS[0], "o"), "sgd": ("SGD", COLORS[3], "^"),
                   "rmsprop": ("SGD + RMSprop", COLORS[1], "s"), "adam": ("SGD + Adam", COLORS[2], "D")}
ACTIVATION_STYLE = {"sigmoid": ("Sigmoid", COLORS[0], "o"), "relu": ("ReLU", COLORS[1], "s"),
                    "leaky_relu": ("Leaky ReLU", COLORS[2], "^")}
DARK_IS_LOW = LinearSegmentedColormap.from_list("dark_is_low", BLUES(np.linspace(0.0, 1.0, 256))[::-1])
DARK_IS_HIGH = LinearSegmentedColormap.from_list("dark_is_high", BLUES(np.linspace(0.0, 1.0, 256)))


def gamma_label(value):
    """Short tick label for a learning rate or penalty, e.g. 3e-4. LLM-assisted: Claude."""
    return "0" if value == 0 else f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e")


def heat_map(ax, values, x_labels, y_labels, x_title, y_title, cmap, norm, fmt=".1f", scale=1e3, ceiling=None,
             bold=None, square=None, circle=None, font=5.6):
    """Annotated seaborn heat map. Values above ceiling are drawn in grey and marked with a dash.

    LLM-assisted: Claude.
    """
    values = np.asarray(values, dtype=float)
    shown = values * scale
    mask = np.zeros(values.shape, dtype=bool) if ceiling is None else values > ceiling
    labels = np.array([[("–" if mask[i, j] else format(shown[i, j], fmt)) for j in range(values.shape[1])]
                       for i in range(values.shape[0])])
    ax.set_facecolor("#d9d8d2")
    sns.heatmap(shown, mask=mask, annot=labels, fmt="", cmap=cmap, norm=norm, cbar=False, ax=ax, square=False,
                linewidths=0.4, linecolor="white", xticklabels=x_labels, yticklabels=y_labels,
                annot_kws={"fontsize": font})
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if mask[i, j]:
                ax.text(j + 0.5, i + 0.5, "–", ha="center", va="center", fontsize=font, color="#52514e")
    if bold is not None:
        for text, is_bold in zip([t for t in ax.texts if t.get_text() != "–"], np.asarray(bold)[~mask].ravel()):
            text.set_fontweight("bold" if is_bold else "normal")
    if square is not None:
        ax.add_patch(Rectangle((square[1] + 0.04, square[0] + 0.04), 0.92, 0.92, fill=False, ec=COLORS[1], lw=1.1, zorder=5))
    if circle is not None:
        ax.add_patch(Circle((circle[1] + 0.5, circle[0] + 0.5), 0.42, fill=False, ec=COLORS[3], lw=1.1, zorder=5))
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.tick_params(length=0)
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)
    ax.grid(False)
    ax.invert_yaxis()


def panel_label(ax, text):
    """Panel letter in the upper left corner, outside the axes. LLM-assisted: Claude."""
    ax.set_title(text, loc="left", fontsize=8, fontweight="bold", pad=3)


# ----------------------------------------------------------------------------------------------
# Regression
# ----------------------------------------------------------------------------------------------
def fit_comparison():
    """Fits, their errors against f, and what happens outside the data."""
    final = load_results("runge_final")
    from regression import make_data, runge
    x, y = make_data(100, 0.1, 2026)
    grid = np.array(final["x_plot"])
    ols = np.array(final["polynomial"]["OLS"]["plot"])
    net = np.array(final["network"]["50-50 sigmoid"]["plot_all"])
    small = np.array(final["network"]["50 sigmoid"]["plot_all"])
    inside = np.abs(grid) <= 1.0
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE, 2.3), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ax[0].plot(x, y, "o", ms=2.2, color=GREY, label="100 observations", zorder=1)
    ax[0].plot(grid[inside], runge(grid[inside]), color=INK, lw=1.0, ls=":", label="$g(x)$", zorder=4)
    ax[0].plot(grid[inside], ols[inside], color=COLORS[0], label="OLS, degree 12", zorder=2)
    ax[0].plot(grid[inside], np.median(net, axis=0)[inside], color=COLORS[1], label="network 1-50-50-1", zorder=3)
    ax[0].set(xlabel="$x$", ylabel="$y$")
    ax[0].legend(loc="upper left", fontsize=5.8, handlelength=1.6)
    ax[0].set_ylim(-0.35, 2.15)  # room above the peak for the legend
    ax[0].set_yticks([0.0, 0.5, 1.0, 1.5])
    ax[1].plot(grid[inside], np.abs(ols - runge(grid))[inside], color=COLORS[0], label="OLS, degree 12")
    ax[1].plot(grid[inside], np.abs(np.median(net, axis=0) - runge(grid))[inside], color=COLORS[1], label="network 1-50-50-1")
    ax[1].plot(grid[inside], np.abs(np.median(small, axis=0) - runge(grid))[inside], color=COLORS[2], label="network 1-50-1", lw=1.0)
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-3, 3.0)
    ax[1].set(xlabel="$x$", ylabel=r"$|\hat g(x)-g(x)|$")
    ax[1].legend(loc="upper center", fontsize=5.8, handlelength=1.6)
    ax[2].axvspan(-1.25, -1.0, color="#e1e0d9", lw=0)
    ax[2].axvspan(1.0, 1.25, color="#e1e0d9", lw=0)
    ax[2].plot(grid, runge(grid), color=INK, lw=1.0, ls=":", label="$g(x)$", zorder=4)
    ax[2].plot(grid, ols, color=COLORS[0], label="OLS, degree 12")
    for row in net:
        ax[2].plot(grid, row, color=COLORS[1], lw=0.5, alpha=0.6)
    ax[2].plot([], [], color=COLORS[1], lw=0.8, label="network, 10 initializations")
    ax[2].set_ylim(-1.0, 1.5)
    ax[2].set_xlim(-1.25, 1.25)
    ax[2].set(xlabel="$x$", ylabel=r"$\hat g(x)$")
    ax[2].legend(loc="lower center", fontsize=5.8, handlelength=1.6)
    for a, letter in zip(ax, "abc"):
        panel_label(a, f"({letter})")
    fig.tight_layout(w_pad=0.6)
    save(fig, "runge_fits")


def learned_features():
    """Terms of the sum: polynomial powers against what two hidden nodes learned."""
    r = load_results("runge_features")
    x = np.array(r["x"])
    inside = np.abs(x) <= 1.0
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE, 2.25))
    terms = np.array(r["polynomial_terms"])
    colors = ordered_colors(terms.shape[0], light=0.2, dark=1.0)
    for term, color in zip(terms, colors):
        ax[0].plot(x[inside], term[inside], color=color, lw=0.8)
    ax[0].plot(x[inside], np.array(r["polynomial_sum"])[inside], color=COLORS[1], lw=1.5, label="sum (OLS, degree 12)")
    ax[0].set_yscale("symlog", linthresh=1.0)
    ax[0].set(xlabel="$x$", ylabel=r"term $\theta_j\,\tilde x_j$")
    ax[0].legend(loc="upper center", fontsize=6)
    terms = np.array(r["network_terms"])
    for k, term in enumerate(terms):
        ax[1].plot(x[inside], term[inside], color=COLORS[(k + 2) % len(COLORS)], lw=0.9)
    ax[1].plot(x[inside], np.array(r["network_sum"])[inside], color=COLORS[1], lw=1.5, label="sum (network 1-2-1)")
    ax[1].set(xlabel="$x$", ylabel=r"term $w_j^{2}\,a_j^{1}(x)$")
    ax[1].legend(loc="upper right", fontsize=6)
    for a, letter in zip(ax, "ab"):
        panel_label(a, f"({letter})")
    fig.tight_layout()
    save(fig, "runge_features")


def approximation():
    """Noise-free approximation error against size, and the error against f over many data sets."""
    r = load_results("runge_approximation")
    s = load_results("runge_sample_size")
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE, 2.3), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    p = np.array(r["degrees"]) + 1
    for name, label, style in (("runge", "Runge", "-"), ("cosine", r"$\cos 3x$", "--")):
        ax[0].plot(p, r[name]["polynomial"], style, color=COLORS[0], lw=1.1, label=f"polynomial, {label}")
        ax[0].plot(r["n_parameters"], r[name]["network"], style, color=COLORS[1], marker="s", ms=2.5, lw=1.1, label=f"1-$M$-1 network, {label}")
    errors = np.array(r["runge"]["polynomial"])
    even = np.array(r["degrees"]) == 20
    ax[0].plot(p, errors[even][0] * np.array(r["rho"]) ** (-2.0 * (np.array(r["degrees"]) - 20)), ":", color=INK, lw=0.9, label=r"$\rho^{-2d}$, $\rho=1.22$")
    ax[0].set_yscale("log")
    ax[0].set_xlim(0, 42)
    ax[0].set_ylim(1e-10, 10.0)  # headroom above the data keeps the legend off the curves
    ax[0].set(xlabel="number of parameters", ylabel="MSE against the function, no noise")
    ax[0].legend(loc="upper right", fontsize=5.3, handlelength=2.2)
    sizes = np.array(s["sizes"])
    for a, name in zip(ax[1:], ("runge", "cosine")):
        for key, label, color, marker, style in (("polynomial", "best polynomial by CV", COLORS[0], "o", "-"),
                                                 ("network", "network, same updates", COLORS[1], "s", "-"),
                                                 ("network_long", "network, 10 times as many", COLORS[1], "^", "--")):
            data = np.array(s[name][key])
            a.plot(sizes, np.median(data, axis=1), color=color, marker=marker, ms=2.8, ls=style, lw=1.1, label=label)
            if key != "network_long":
                a.fill_between(sizes, np.quantile(data, 0.25, axis=1), np.quantile(data, 0.75, axis=1), color=color, alpha=0.16, lw=0)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set(xlabel="number of observations $n$", ylabel="MSE against the function")
    ax[2].set_ylim(top=3e-2)  # room above the bands for the legend
    ax[2].legend(loc="upper right", fontsize=5.5)
    panel_label(ax[0], "(a)")
    panel_label(ax[1], "(b) Runge")
    panel_label(ax[2], r"(c) $\cos 3x$")
    fig.tight_layout(w_pad=0.5)
    save(fig, "runge_approximation")


def learning_rate():
    """CV error against learning rate for the four optimizers, and the learning curves at the best rate."""
    r = load_results("runge_learning_rate")
    gammas = np.array(r["gammas"])
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE, 2.45))
    best = {}
    for name, (label, color, marker) in OPTIMIZER_STYLE.items():
        runs = [u for u in r["runs"] if u["settings"]["optimizer"] == name]
        last = np.array([u["last_val"] for u in runs])
        se = np.array([u["last_val_se"] for u in runs])
        last_plot = np.where(last > 1.0, np.nan, last)
        ax[0].errorbar(gammas, last_plot, yerr=np.where(last > 1.0, np.nan, se), color=color, marker=marker, ms=3, capsize=1.5, lw=1.1, label=label)
        best[name] = runs[int(np.argmin(last))]
    ax[0].axhline(NOISE_VARIANCE, color=GREY, lw=0.8, ls=":")
    ax[0].text(1.3e-5, NOISE_VARIANCE * 1.08, r"$\sigma_\varepsilon^2$", fontsize=7, color="#52514e", va="bottom")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_ylim(0.0085, 0.3)
    ax[0].set(xlabel=r"learning rate $\gamma$", ylabel="CV MSE after 10 000 epochs")
    ax[0].legend(loc="upper center", ncol=2, fontsize=6)
    for name, (label, color, marker) in OPTIMIZER_STYLE.items():
        u = best[name]
        epochs = np.array(u["epochs"])
        val = np.where(np.array(u["val"]) > 1.0, np.nan, u["val"])
        ax[1].plot(epochs, val, color=color, label=f"{label}, $\\gamma$={u['settings']['gamma']:g}")
        ax[1].plot(epochs, u["train"], color=color, ls="--", lw=0.8)
    ax[1].axhline(NOISE_VARIANCE, color=GREY, lw=0.8, ls=":")
    ax[1].plot([], [], color=GREY, ls="--", lw=0.8, label="training MSE")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_ylim(0.0085, 0.2)
    ax[1].set(xlabel="epoch", ylabel="MSE (solid: CV)")
    ax[1].legend(loc="lower left", bbox_to_anchor=(0.0, 0.07), fontsize=6)  # just above the noise line
    for a, letter in zip(ax, "ab"):
        panel_label(a, f"({letter})")
    fig.tight_layout()
    save(fig, "runge_learning_rate")


def batch_maps():
    """CV MSE over the batch size and the learning rate."""
    r = load_results("runge_batch")
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE, 1.95))
    norm = LogNorm(vmin=10, vmax=110)
    for a, name, letter in zip(ax, ("sgd", "adam"), "ab"):
        m = r[name]
        mse = np.array(m["mse"])
        best = np.unravel_index(np.argmin(mse), mse.shape)
        within = mse <= mse[best] + np.array(m["se"])[best]
        heat_map(a, mse, [gamma_label(g) for g in m["values"][1]], m["values"][0], r"learning rate $\gamma$", "batch size $B$",
                 DARK_IS_LOW, norm, fmt=".1f", ceiling=1.0, bold=within, square=best)
        panel_label(a, f"({letter}) {OPTIMIZER_STYLE[name][0]}")
    fig.tight_layout()
    save(fig, "runge_batch")


def activation_maps():
    """CV MSE over depth and width for three activations and two optimizers."""
    r = load_results("runge_activation_maps")
    depths, widths = r["depths"], r["widths"]
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 3.3), sharex=True, sharey=True)
    norm = LogNorm(vmin=10, vmax=110)
    for row, optimizer in zip(axes, ("sgd", "adam")):
        for a, name in zip(row, ACTIVATION_STYLE):
            m = r[f"{optimizer}_{name}"]
            mse = np.array(m["mse"]).min(axis=0).reshape(len(depths), len(widths))
            heat_map(a, mse, widths, depths, "", "", DARK_IS_LOW, norm, ceiling=1.0, square=np.unravel_index(np.argmin(mse), mse.shape))
            panel_label(a, f"{ACTIVATION_STYLE[name][0]}, {'plain SGD' if optimizer == 'sgd' else 'Adam'}")
    for a in axes[1]:
        a.set_xlabel("nodes per layer")
    for row in axes:
        row[0].set_ylabel("hidden layers")
    fig.tight_layout(h_pad=0.8, w_pad=0.5)
    save(fig, "runge_activation_maps")


def gradient_flow():
    """Size of the weight gradient per layer in an untrained deep network."""
    r = load_results("runge_gradient_flow")
    layers = np.arange(1, r["depth"] + 2)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.2))
    entries = (("sigmoid", r"Sigmoid, $\mathcal{N}(0,1/M)$", COLORS[0], "o"), ("sigmoid_unit", r"Sigmoid, $\mathcal{N}(0,1)$", COLORS[4], "v"),
               ("relu", r"ReLU, $\mathcal{N}(0,2/M)$", COLORS[1], "s"), ("leaky_relu", r"Leaky ReLU, $\mathcal{N}(0,2/M)$", COLORS[2], "^"))
    for key, label, color, marker in entries:
        ax.plot(layers, r[key]["median"], color=color, marker=marker, ms=3, label=label)
        ax.fill_between(layers, r[key]["q25"], r[key]["q75"], color=color, alpha=0.16, lw=0)
    ax.plot(layers, r["sigmoid"]["median"][-1] * 4.0 ** (-(layers[-1] - layers)), ":", color=GREY, lw=1.0, label=r"factor $1/4$ per layer")
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 60)
    ax.set_xticks(layers)
    ax.set(xlabel="layer $l$ (input side to output side)", ylabel=r"$\|\partial C/\partial W^{l}\|_F$")
    ax.legend(loc="lower right", fontsize=5.8)
    fig.tight_layout()
    save(fig, "runge_gradient_flow")


def overfitting():
    """Training and CV error against epoch and size."""
    r = load_results("runge_overfitting")["runs"]
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE, 2.35))
    chosen = {"[5]": ("1-5-1", COLORS[0]), "[100, 100, 100]": ("1-100-100-100-1", COLORS[1])}
    for u in r:
        key = str(u["settings"]["hidden"])
        if u["settings"]["activation"] == "sigmoid" and key in chosen:
            label, color = chosen[key]
            e = np.array(u["epochs"])
            ax[0].plot(e, u["val"], color=color, label=f"{label}, CV")
            ax[0].plot(e, u["train"], color=color, ls="--", lw=0.9, label=f"{label}, training")
            ax[0].plot(u["best_epoch"], u["best_val"], marker="v", color=color, ms=5, ls="none", zorder=5)
    ax[0].axhline(NOISE_VARIANCE, color=GREY, lw=0.8, ls=":")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_ylim(0.005, 0.4)  # room above the curves for the legend
    ax[0].set(xlabel="epoch", ylabel="MSE")
    ax[0].legend(loc="upper right", fontsize=6)
    for name, (label, color, marker) in {k: ACTIVATION_STYLE[k] for k in ("sigmoid", "relu")}.items():
        family = sorted([u for u in r if u["settings"]["activation"] == name], key=lambda u: u["n_parameters"])
        p = [u["n_parameters"] for u in family]
        ax[1].errorbar(p, [u["last_val"] for u in family], yerr=[u["last_val_se"] for u in family], color=color, marker=marker, ms=3, capsize=1.5, label=f"{label}, CV")
        ax[1].plot(p, [u["last_train"] for u in family], color=color, marker=marker, ms=3, ls="--", lw=0.9, mfc="white", label=f"{label}, training")
    ax[1].axhline(NOISE_VARIANCE, color=GREY, lw=0.8, ls=":")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_ylim(0.005, 0.4)
    ax[1].set(xlabel="number of parameters", ylabel="MSE after 30 000 epochs")
    ax[1].legend(loc="upper center", ncol=2, fontsize=6)
    for a, letter in zip(ax, "ab"):
        panel_label(a, f"({letter})")
    fig.tight_layout()
    save(fig, "runge_overfitting")


def penalty_maps():
    """CV and training MSE over learning rate and penalty, at the best architecture of each penalty."""
    r = load_results("runge_penalties")
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 3.25), sharex=True, sharey=True)
    norm = LogNorm(vmin=10, vmax=110)
    x_labels = [gamma_label(v) for v in r["lambdas"]]
    y_labels = [gamma_label(v) for v in r["gammas"]]
    for row, penalty in zip(axes, ("l2", "l1")):
        keys = [k for k in r if k.startswith(penalty)]
        key = min(keys, key=lambda k: r[k]["mse"][r[k]["minimum"][0]][r[k]["minimum"][1]])
        m = r[key]
        architecture = "1-" + "-".join(key.split("_")[1:]) + "-1"
        heat_map(row[0], m["mse"], x_labels, y_labels, "", "", DARK_IS_LOW, norm, ceiling=1.0, bold=m["within"], square=m["minimum"], circle=m["one_se"])
        heat_map(row[1], m["train"], x_labels, y_labels, "", "", DARK_IS_LOW, norm, ceiling=1.0)
        panel_label(row[0], f"{penalty.upper()}, {architecture}: CV MSE")
        panel_label(row[1], f"{penalty.upper()}, {architecture}: training MSE")
        row[0].set_ylabel(r"learning rate $\gamma$")
    for a in axes[1]:
        a.set_xlabel(r"penalty $\lambda$")
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, "runge_penalties")


def sparsity():
    """Sorted size of the output weights with no penalty, L2 and L1."""
    r = load_results("runge_sparsity")
    fig, ax = plt.subplots(figsize=(SINGLE, 2.05))
    for key, label, color, style in (("none", "no penalty", COLORS[0], "-"), ("l2", r"$L_2$, $\lambda=10^{-3}$", COLORS[1], "-"),
                                     ("l1", r"$L_1$, $\lambda=10^{-3}$", COLORS[2], "-")):
        ax.plot(np.arange(1, len(r[key]["median"]) + 1), r[key]["median"], color=color, ls=style, label=label)
    ax.set_yscale("log")
    ax.set_ylim(top=20)  # room above the curves for the legend
    ax.set(xlabel="output weight, sorted by size", ylabel=r"$|w_{j1}^{2}|$")
    ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    save(fig, "runge_sparsity")


# ----------------------------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------------------------
def mnist_optimizers():
    """Validation accuracy against learning rate and epoch for the four optimizers."""
    r = load_results("mnist_optimizers")
    gammas = np.array(r["gammas"])
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE, 2.35))
    for name, (label, color, marker) in OPTIMIZER_STYLE.items():
        runs = [u for u in r["runs"] if u["optimizer"] == name]
        accuracy = np.array([u["last_val"] for u in runs]) * 100
        ax[0].plot(gammas, accuracy, color=color, marker=marker, ms=3, label=label)
        best = max(runs, key=lambda u: u["last_val"])
        epochs = np.arange(1, len(best["val"]) + 1)
        ax[1].plot(epochs, np.array(best["val"]) * 100, color=color, label=f"{label}, $\\gamma$={best['gamma']:g}")
    ax[0].axhline(10, color=GREY, lw=0.8, ls=":")
    ax[0].text(1.2e-5, 2.0, "guessing", fontsize=6, color="#52514e", va="bottom")
    ax[0].set_xscale("log")
    ax[0].set(xlabel=r"learning rate $\gamma$", ylabel="validation accuracy (%)")
    ax[0].set_ylim(0, 128)  # room above 100 % for the legend
    ax[0].set_yticks(range(0, 101, 20))
    ax[0].legend(loc="upper center", ncol=2, fontsize=6)
    ax[1].set_xscale("log")
    ax[1].set_ylim(60, 100)
    ax[1].set(xlabel="epoch (passes over the fitting images)", ylabel="validation accuracy (%)")
    ax[1].legend(loc="center left", fontsize=6)
    for a, letter in zip(ax, "ab"):
        panel_label(a, f"({letter})")
    fig.tight_layout()
    save(fig, "mnist_optimizers")


def mnist_architectures():
    """Validation accuracy over depth and width for three activations, and over depth with SGD and Adam."""
    r = load_results("mnist_architectures")
    depth = load_results("mnist_depth")
    fig = plt.figure(figsize=(DOUBLE, 3.6))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05])
    norm = Normalize(vmin=95.0, vmax=98.5)
    depths, widths = r["depths"], r["widths"]
    for k, name in enumerate(ACTIVATION_STYLE):
        a = fig.add_subplot(grid[0, k])
        runs = [u for u in r["runs"] if u["activation"] == name]
        values = np.array([u["best_val"] for u in runs]).reshape(len(depths), len(widths))
        heat_map(a, values, widths, depths, "nodes per layer", "hidden layers" if k == 0 else "", DARK_IS_HIGH, norm, fmt=".2f", scale=100.0,
                 square=np.unravel_index(np.argmax(values), values.shape))
        panel_label(a, f"({'abc'[k]}) {ACTIVATION_STYLE[name][0]}, Adam")
    for k, (optimizer, title) in enumerate((("sgd", r"plain SGD, $\gamma=0.1$"), ("adam", r"Adam, $\gamma=10^{-3}$"))):
        a = fig.add_subplot(grid[1, k])
        for name, (label, color, marker) in ACTIVATION_STYLE.items():
            runs = [u for u in depth["runs"] if u["optimizer"] == optimizer and u["activation"] == name]
            a.plot(depth["depths"], np.array([u["last_val"] for u in runs]) * 100, color=color, marker=marker, ms=3, label=label)
        a.axhline(10, color=GREY, lw=0.8, ls=":")
        a.set_ylim(0, 100)
        a.set(xlabel="hidden layers of 128 nodes", ylabel="validation accuracy (%)" if k == 0 else "")
        a.legend(loc="center left", fontsize=6)
        panel_label(a, f"({'de'[k]}) {title}")
    a = fig.add_subplot(grid[1, 2])
    runs = [u for u in r["runs"] if u["activation"] == "relu" and len(u["hidden"]) == 1]
    a.plot([u["hidden"][0] for u in runs], np.array([u["last_val"] for u in runs]) * 100, color=COLORS[1], marker="s", ms=3, label="validation")
    a.plot([u["hidden"][0] for u in runs], np.array([u["last_train"] for u in runs]) * 100, color=COLORS[1], marker="s", ms=3, ls="--", mfc="white", lw=0.9, label="training")
    a.set_xscale("log")
    a.set_xticks(widths)
    a.set_xticklabels(widths)
    a.xaxis.set_minor_formatter(plt.NullFormatter())
    a.set(xlabel="nodes in one ReLU layer", ylabel="accuracy (%)")
    a.legend(loc="lower right", fontsize=6)
    panel_label(a, "(f) one ReLU layer, Adam")
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, "mnist_architectures")


def mnist_penalties():
    """Validation accuracy over learning rate and penalty."""
    r = load_results("mnist_penalties")
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE, 1.95), sharey=True)
    norm = Normalize(vmin=96.0, vmax=98.6)
    gammas, lambdas = r["gammas"], r["lambdas"]
    best_all = max(u["best_val"] for p in ("l2", "l1") for u in r[p])
    se = np.sqrt(best_all * (1 - best_all) / 10_000)
    for a, penalty, letter in zip(ax, ("l2", "l1"), "ab"):
        values = np.array([u["best_val"] for u in r[penalty]]).reshape(len(gammas), len(lambdas))
        heat_map(a, values, [gamma_label(v) for v in lambdas], [gamma_label(v) for v in gammas], r"penalty $\lambda$",
                 r"learning rate $\gamma$" if letter == "a" else "", DARK_IS_HIGH, norm, fmt=".2f", scale=100.0,
                 bold=values >= best_all - se, square=np.unravel_index(np.argmax(values), values.shape))
        panel_label(a, f"({letter}) {penalty.upper()} penalty")
    fig.tight_layout(w_pad=0.6)
    save(fig, "mnist_penalties")


def mnist_scaling_pairs():
    """What bad scaling and a mismatched output layer and cost do to the training."""
    r = load_results("mnist_scaling_pairs")
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE, 2.25))
    labels = {(1.0, "auto"): ("pixels in [0, 1], $\\mathcal{N}(0,1/M)$", COLORS[0]), (255.0, "auto"): ("pixels in [0, 255], $\\mathcal{N}(0,1/M)$", COLORS[1]),
              (1.0, "unit"): ("pixels in [0, 1], $\\mathcal{N}(0,1)$", COLORS[2]), (255.0, "unit"): ("pixels in [0, 255], $\\mathcal{N}(0,1)$", COLORS[3])}
    for key, (label, color) in labels.items():
        family = [u for u in r["scaling"] if (u["input_scale"], u["init"]) == key]
        best = max(family, key=lambda u: u["last_val"])
        ax[0].plot(np.arange(1, len(best["val"]) + 1), np.array(best["val"]) * 100, color=color, label=label)
    ax[0].axhline(10, color=GREY, lw=0.8, ls=":")
    ax[0].set_ylim(0, 100)
    ax[0].set(xlabel="epoch", ylabel="validation accuracy (%)")
    ax[0].legend(loc="center right", fontsize=5.8)
    for cost, label, color in (("cross_entropy", "softmax + cross entropy", COLORS[0]), ("mse", "softmax + squared error", COLORS[1])):
        family = [u for u in r["pairs"] if u["cost"] == cost]
        best = max(family, key=lambda u: u["last_val"])
        ax[1].plot(np.arange(1, len(best["val"]) + 1), np.array(best["val"]) * 100, color=color, label=f"{label}, $\\gamma$={best['gamma']:g}")
    ax[1].set_ylim(80, 99)
    ax[1].set(xlabel="epoch", ylabel="validation accuracy (%)")
    ax[1].legend(loc="lower right", fontsize=6)
    for a, letter in zip(ax, "ab"):
        panel_label(a, f"({letter})")
    fig.tight_layout()
    save(fig, "mnist_scaling_pairs")


def mnist_confusion():
    """Confusion matrix of the selected network with per-class precision and recall."""
    r = load_results("mnist_final")
    matrix = np.array(r["confusion_matrix"])
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE, 2.95), gridspec_kw={"width_ratios": [1.15, 1]})
    off = matrix.astype(float)
    np.fill_diagonal(off, np.nan)
    labels = np.array([[str(v) if (v > 0) else "" for v in row] for row in matrix])
    sns.heatmap(off, annot=labels, fmt="", cmap=DARK_IS_HIGH, vmin=0, vmax=np.nanmax(off), cbar=False, ax=ax[0], linewidths=0.4, linecolor="white",
                annot_kws={"fontsize": 6}, square=True)
    for i in range(10):
        ax[0].add_patch(Rectangle((i, i), 1, 1, fill=True, fc="#e1e0d9", ec="white", lw=0.4, zorder=1))
    ax[0].set(xlabel="predicted digit", ylabel="true digit")
    ax[0].tick_params(length=0, axis="y", labelrotation=0)
    ax[0].grid(False)
    digits = np.arange(10)
    per = r["per_class"]
    ax[1].errorbar(digits - 0.12, np.array(per["precision"]) * 100, yerr=np.array(per["precision_se"]) * 100, ls="none", marker="o", ms=3.5, capsize=1.5, color=COLORS[0], label="precision (column)")
    ax[1].errorbar(digits + 0.12, np.array(per["recall"]) * 100, yerr=np.array(per["recall_se"]) * 100, ls="none", marker="s", ms=3.5, capsize=1.5, color=COLORS[1], label="recall (row)")
    ax[1].axhline(r["network"]["accuracy"] * 100, color=GREY, lw=0.8, ls=":", label="overall accuracy")
    ax[1].set_xticks(digits)
    ax[1].set_ylim(94.5, 100.0)  # room below the points for the legend
    ax[1].set(xlabel="digit", ylabel="per-class score (%)")
    ax[1].legend(loc="lower left", fontsize=6)
    for a, letter in zip(ax, "ab"):
        panel_label(a, f"({letter})")
    fig.tight_layout()
    save(fig, "mnist_confusion")


def mnist_errors():
    """The most confident errors of the network, and the templates logistic regression has learned."""
    r = load_results("mnist_final")
    images = np.array(r["images_shown"]["images"])
    index = list(r["images_shown"]["index"])
    all_index = list(r["errors"]["index"])
    fig = plt.figure(figsize=(DOUBLE, 2.0))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.08)
    left = outer[0].subgridspec(2, 5, wspace=0.05, hspace=0.05)
    weights = np.array(r["baseline_weights"])
    limit = np.abs(weights).max() * 0.6
    for k in range(10):
        a = fig.add_subplot(left[k // 5, k % 5])
        a.imshow(weights[k], cmap=sns.diverging_palette(250, 15, s=85, l=45, as_cmap=True), vmin=-limit, vmax=limit)
        a.set_axis_off()
        a.set_title(str(k), fontsize=6, pad=1)
    right = outer[1].subgridspec(2, 5, wspace=0.05, hspace=0.3)
    for k in range(10):
        a = fig.add_subplot(right[k // 5, k % 5])
        a.imshow(images[k], cmap="gray_r")
        a.set_axis_off()
        j = all_index.index(index[k])
        a.set_title(f"{r['errors']['true'][j]} → {r['errors']['predicted'][j]}", fontsize=6, pad=1)
    save(fig, "mnist_errors")


# keys are the names of the PDF files in report/figures/
FIGURES = {"runge_fits": fit_comparison, "runge_features": learned_features, "runge_approximation": approximation,
           "runge_learning_rate": learning_rate, "runge_batch": batch_maps, "runge_activation_maps": activation_maps,
           "runge_gradient_flow": gradient_flow, "runge_overfitting": overfitting, "runge_penalties": penalty_maps,
           "runge_sparsity": sparsity, "mnist_optimizers": mnist_optimizers, "mnist_architectures": mnist_architectures,
           "mnist_penalties": mnist_penalties, "mnist_scaling_pairs": mnist_scaling_pairs,
           "mnist_confusion": mnist_confusion, "mnist_errors": mnist_errors}

if __name__ == "__main__":
    set_style()
    for figure_name in sys.argv[1:] or FIGURES:
        FIGURES[figure_name]()
        print("drew", figure_name, flush=True)
