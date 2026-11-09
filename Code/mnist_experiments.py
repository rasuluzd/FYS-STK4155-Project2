"""
Classification experiments on MNIST. Every function runs one experiment and writes its numbers to
results/mnist_<name>.json, from which figures.py draws the figures.

All choices are made with the 50 000 fitting images and the 10 000 validation images. The 10 000 test
images are only read by final_evaluation, which is run once, after all the choices have been made.

Run all except final_evaluation with `python mnist_experiments.py`, or a few with
`python mnist_experiments.py baseline_scan`.

LLM-assisted
------------
Code level 2. Tool: Claude Opus 5.5 (Anthropic, Claude Code desktop app, November 2026).
Role: the splits, the settings and the evaluation protocol are the author's and follow the
course material; Claude drafted the parallelisation and the probes around them. The
scikit-learn reference for logistic regression uses the settings suggested in the course's
project text.
OpenAI Codex also refactored the functions marked below for readability in November 2026.
"""

import sys
import time
from functools import lru_cache

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier

from estimators import NeuralNetClassifier, update_rule
from mnist_data import load_test, load_train, train_validation_split
from network import NeuralNetwork, accuracy, one_hot, train
from plot_style import load_results, save_results

SEED = 2026
N_JOBS = 14
BATCH = 128
N_MONITOR = 10_000  # fitting images on which the training accuracy is followed


@lru_cache(maxsize=1)
def fit_and_validation():
    """The stratified split of the training images, cached once per worker. LLM-assisted: Claude."""
    X, y = load_train()
    return train_validation_split(X, y, seed=SEED)


def run(hidden=(128,), activation="relu", output="softmax", cost="cross_entropy", penalty=None, lam=0.0,
        init="auto", optimizer="adam", gamma=1e-3, batch_size=BATCH, epochs=15, seed=SEED, input_scale=1.0,
        probe=None):
    """Train one network and follow the validation and training accuracy epoch by epoch.

    input_scale multiplies the pixels, which lie in [0, 1]; 255 gives the raw grey levels.
    probe(net, theta, X_val), if given, is also recorded before training and after every epoch.

    LLM-assisted: OpenAI Codex, readability refactor in November 2026.
    """
    X_fit, X_val, y_fit, y_val = fit_and_validation()
    X_fit = input_scale * X_fit
    X_val = input_scale * X_val
    net = NeuralNetwork([784, *hidden, 10], [activation] * len(hidden) + [output], cost, penalty, seed=seed, init=init)
    rule, batch = update_rule(optimizer, gamma, batch_size, len(X_fit))
    val = []
    tr = []
    probes = []

    def record(theta):
        with np.errstate(all="ignore"):
            val.append(accuracy(net.predict(X_val, theta), y_val))
            tr.append(accuracy(net.predict(X_fit[:N_MONITOR], theta), y_fit[:N_MONITOR]))
            if probe is not None:
                probes.append(probe(net, theta, X_val))

    if probe is not None:
        probes.append(probe(net, net.theta, X_val))
    start = time.perf_counter()
    with np.errstate(all="ignore"):
        info = train(net, X_fit, one_hot(y_fit), rule, epochs, batch, lam, seed, record)
    elapsed = time.perf_counter() - start
    best = int(np.argmax(val)) if val else 0
    result = {
        "hidden": hidden, "activation": activation, "output": output, "cost": cost,
        "penalty": penalty, "lam": lam, "init": init, "optimizer": optimizer,
        "gamma": gamma, "batch_size": batch, "epochs": epochs, "input_scale": input_scale,
        "val": val, "train": tr, "best_epoch": best + 1,
        "best_val": val[best] if val else 0.0,
        "last_val": val[-1] if val else 0.0,
        "last_train": tr[-1] if tr else 0.0,
        "seconds": elapsed, "diverged": bool(info["diverged"]),
    }
    if probe is not None:
        result["probe"] = probes
    return result


def run_all(configurations):
    """Run a list of settings dicts in parallel, one thread per process. LLM-assisted: Claude."""
    return Parallel(n_jobs=N_JOBS)(delayed(run)(**c) for c in configurations)


GAMMAS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]


def baseline_scan():
    """Logistic regression, the network without hidden layers: learning rate and L2 penalty.

    LLM-assisted: OpenAI Codex, readability refactor in November 2026.
    """
    gammas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    lambdas = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    configurations = []
    for gamma in gammas:
        for lam in lambdas:
            configurations.append(dict(hidden=(), penalty="l2", lam=lam, gamma=gamma, epochs=30))
    runs = run_all(configurations)
    save_results("mnist_baseline", {"gammas": gammas, "lambdas": lambdas, "runs": runs})


def optimizer_scan():
    """One hidden layer of 128 ReLU nodes: four optimizers over twelve learning rates.

    Plain gradient descent makes one update per epoch, so it gets 200 epochs instead of 15; the curves
    show how far it has come after the same number of passes over the data.

    LLM-assisted: OpenAI Codex, readability refactor in November 2026.
    """
    configurations = []
    for optimizer in ("gd", "sgd", "rmsprop", "adam"):
        epochs = 200 if optimizer == "gd" else 15
        for gamma in GAMMAS:
            configurations.append(dict(optimizer=optimizer, gamma=gamma, epochs=epochs))
    runs = run_all(configurations)
    save_results("mnist_optimizers", {"gammas": GAMMAS, "runs": runs})


DEPTHS = [1, 2, 3, 4]
WIDTHS = [32, 64, 128, 256, 512]


def architecture_maps():
    """Validation accuracy over depth and width for the three activations, Adam at a fixed rate.

    LLM-assisted: OpenAI Codex, readability refactor in November 2026.
    """
    configurations = []
    for activation in ("sigmoid", "relu", "leaky_relu"):
        for depth in DEPTHS:
            for width in WIDTHS:
                configurations.append(dict(hidden=(width,) * depth, activation=activation,
                                           gamma=1e-3, epochs=15))
    runs = run_all(configurations)
    save_results("mnist_architectures", {"depths": DEPTHS, "widths": WIDTHS, "runs": runs})


def depth_study():
    """Deep narrow networks: where does each activation stop training, with and without Adam?

    LLM-assisted: OpenAI Codex, readability refactor in November 2026.
    """
    depths = [1, 2, 3, 4, 6, 8, 10]
    configurations = []
    for optimizer, gamma in (("sgd", 0.1), ("adam", 1e-3)):
        for activation in ("sigmoid", "relu", "leaky_relu"):
            for depth in depths:
                configurations.append(dict(hidden=(128,) * depth, activation=activation,
                                           optimizer=optimizer, gamma=gamma, epochs=15))
    runs = run_all(configurations)
    save_results("mnist_depth", {"depths": depths, "runs": runs})


def scaling_and_pairs():
    """What bad input scaling, bad weight scaling and a mismatched output layer and cost cost.

    Plain SGD is used, since Adam would hide part of the effect of scaling.

    LLM-assisted: OpenAI Codex, readability refactor in November 2026.
    """
    scaling = []
    for gamma in (0.01, 0.1, 1.0):
        for input_scale, init in ((1.0, "auto"), (255.0, "auto"), (1.0, "unit"), (255.0, "unit")):
            scaling.append(dict(hidden=(128, 128), activation="sigmoid", optimizer="sgd",
                                gamma=gamma, input_scale=input_scale, init=init, epochs=15))

    pairs = []
    for cost, gammas in (("cross_entropy", (0.03, 0.1, 0.3, 1.0)),
                         ("mse", (0.03, 0.1, 0.3, 1.0, 3.0, 10.0))):
        for gamma in gammas:
            pairs.append(dict(hidden=(128,), activation="relu", optimizer="sgd",
                              gamma=gamma, output="softmax", cost=cost, epochs=15))
    save_results("mnist_scaling_pairs", {"scaling": run_all(scaling), "pairs": run_all(pairs)})


def chosen_architecture():
    """Smallest network within one standard error of the best validation accuracy in the maps.

    The standard error of an accuracy p on n images is sqrt(p(1-p)/n).

    LLM-assisted: OpenAI Codex, readability refactor in November 2026.
    """
    runs = load_results("mnist_architectures")["runs"]
    best = max(runs, key=lambda r: r["best_val"])
    se = np.sqrt(best["best_val"] * (1 - best["best_val"]) / 10_000)
    inside = [r for r in runs if r["best_val"] >= best["best_val"] - se]

    choice = min(inside, key=lambda r: (sum(r["hidden"]), -r["best_val"]))
    return best, choice, se


def penalty_maps():
    """Validation accuracy over learning rate and penalty for the chosen network.

    LLM-assisted: OpenAI Codex, readability refactor in November 2026.
    """
    _, choice, _ = chosen_architecture()
    gammas = [1e-4, 3e-4, 1e-3, 3e-3]
    lambdas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
    settings = dict(hidden=tuple(choice["hidden"]), activation=choice["activation"], epochs=20)
    results = {"gammas": gammas, "lambdas": lambdas, "hidden": settings["hidden"], "activation": settings["activation"]}
    for penalty in ("l2", "l1"):
        configurations = []
        for gamma in gammas:
            for lam in lambdas:
                configurations.append(dict(settings, penalty=penalty, lam=lam, gamma=gamma))
        results[penalty] = run_all(configurations)
    save_results("mnist_penalties", results)


def final_settings():
    """The models carried to the test, from the validation results only.

    LLM-assisted: OpenAI Codex, readability refactor in November 2026.
    """
    penalties = load_results("mnist_penalties")
    penalty = max(("l2", "l1"), key=lambda p: max(r["best_val"] for r in penalties[p]))
    candidates = penalties[penalty]
    best = max(candidates, key=lambda r: r["best_val"])
    se = np.sqrt(best["best_val"] * (1 - best["best_val"]) / 10_000)
    # one-standard-error rule: inside the band, the largest penalty, then the fewest epochs
    inside = [r for r in candidates if r["best_val"] >= best["best_val"] - se]
    network = max(inside, key=lambda r: (r["lam"], -r["best_epoch"]))
    baseline = max(load_results("mnist_baseline")["runs"], key=lambda r: r["best_val"])
    keys = ("hidden", "activation", "optimizer", "gamma", "penalty", "lam", "batch_size")
    selected_settings = []
    for selected in (network, baseline):
        settings = {key: selected[key] for key in keys}
        settings["hidden"] = tuple(settings["hidden"])
        settings["epochs"] = selected["best_epoch"]
        selected_settings.append(settings)
    validation = {
        "best_val_network": network["best_val"],
        "best_val_baseline": baseline["best_val"],
        "best_val_any": best["best_val"],
        "se": se,
    }
    return selected_settings[0], selected_settings[1], validation


def _final_fit(settings, seed, X, y, X_test):
    """Fit on all 60 000 images and return the probabilities for the test images."""
    start = time.perf_counter()
    model = NeuralNetClassifier(seed=seed, **settings).fit(X, y)
    elapsed = time.perf_counter() - start
    return model.predict_proba(X_test), elapsed, float(model.score(X, y)), model.net_.theta


def _library_network(network, activation, seed, X, y, X_test):
    """scikit-learn's MLPClassifier with the training settings of our network; test probabilities, time, training accuracy.

    scikit-learn adds alpha*W/B to the mean gradient of a batch of B samples, and here 2*lam*W is added,
    so alpha = 2*lam*B. It has no L1 penalty, never penalizes the biases, has no leaky ReLU, and draws
    its own initial weights.
    """
    alpha = 2.0 * network["lam"] * network["batch_size"] if network["penalty"] == "l2" else 0.0
    start = time.perf_counter()
    model = MLPClassifier(hidden_layer_sizes=network["hidden"], activation=activation, solver="adam", alpha=alpha,
                          learning_rate_init=network["gamma"], batch_size=network["batch_size"],
                          max_iter=network["epochs"], tol=0.0, n_iter_no_change=10**9, random_state=seed).fit(X, y)
    return model.predict_proba(X_test), time.perf_counter() - start, float(model.score(X, y))


def _summary(probabilities, y_test, seconds, train_accuracy):
    """Accuracy and its binomial standard error."""
    p = accuracy(probabilities, y_test)
    return {"accuracy": p, "se": float(np.sqrt(p * (1 - p) / len(y_test))), "seconds": seconds, "train_accuracy": train_accuracy}


def final_evaluation(n_seeds=5):
    """Fit the selected models on all training images and score them on the test images, once."""
    X, y = load_train()
    X_test, y_test = load_test()
    network, baseline, validation = final_settings()
    results = {"settings": {"network": network, "baseline": baseline}, "validation": validation}

    runs = Parallel(n_jobs=min(N_JOBS, n_seeds))(delayed(_final_fit)(network, SEED + s, X, y, X_test) for s in range(n_seeds))
    accuracies = [accuracy(r[0], y_test) for r in runs]
    results["network"] = {**_summary(runs[0][0], y_test, runs[0][1], runs[0][2]), "accuracies": accuracies,
                          "mean": float(np.mean(accuracies)), "std": float(np.std(accuracies, ddof=1)),
                          "n_parameters": int(runs[0][3].size)}
    probabilities = runs[0][0]

    lr = Parallel(n_jobs=min(N_JOBS, n_seeds))(delayed(_final_fit)(baseline, SEED + s, X, y, X_test) for s in range(n_seeds))
    lr_acc = [accuracy(r[0], y_test) for r in lr]
    results["baseline"] = {**_summary(lr[0][0], y_test, lr[0][1], lr[0][2]), "accuracies": lr_acc, "mean": float(np.mean(lr_acc)),
                           "std": float(np.std(lr_acc, ddof=1)), "n_parameters": int(lr[0][3].size)}

    reference = dict(hidden=(100,), activation="relu", optimizer="sgd", gamma=0.1, epochs=20, batch_size=100)
    ref = _final_fit(reference, SEED, X, y, X_test)
    results["reference_100"] = {**_summary(ref[0], y_test, ref[1], ref[2]), "settings": reference}

    start = time.perf_counter()
    library_lr = LogisticRegression(solver="saga", max_iter=1000, random_state=42).fit(X, y)
    results["library_logistic"] = {**_summary(library_lr.predict_proba(X_test), y_test, time.perf_counter() - start,
                                              float(library_lr.score(X, y))), "iterations": int(library_lr.n_iter_[0])}

    # scikit-learn has no leaky ReLU, so its network gets sigmoid nodes here; library_validation compares
    # with its ReLU network on the validation images
    library = _library_network(network, "logistic", SEED, X, y, X_test)
    results["library_network"] = _summary(library[0], y_test, *library[1:])

    predicted = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(y_test, predicted)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, predicted)
    results["confusion_matrix"] = matrix
    results["baseline_confusion_matrix"] = confusion_matrix(y_test, np.argmax(lr[0][0], axis=1))
    results["per_class"] = {"precision": precision, "recall": recall, "f1": f1, "support": support,
                            "precision_se": np.sqrt(precision * (1 - precision) / matrix.sum(axis=0)),
                            "recall_se": np.sqrt(recall * (1 - recall) / support)}
    wrong = np.where(predicted != y_test)[0]
    confidence = probabilities[wrong, predicted[wrong]]
    results["errors"] = {"count": int(wrong.size), "index": wrong, "true": y_test[wrong], "predicted": predicted[wrong],
                         "confidence": confidence, "median_confidence": float(np.median(confidence)),
                         "median_confidence_correct": float(np.median(probabilities[predicted == y_test].max(axis=1)))}
    results["images_shown"] = {"index": wrong[np.argsort(-confidence)[:18]],
                               "images": X_test[wrong[np.argsort(-confidence)[:18]]].reshape(-1, 28, 28)}
    weights = lr[0][3]
    results["baseline_weights"] = weights[:7840].reshape(784, 10).T.reshape(10, 28, 28)
    save_results("mnist_final", results)
    return results


def library_validation(n_seeds=5):
    """Like-for-like comparison with scikit-learn on the validation images, without the test images.

    Our selected network and scikit-learn's network with ReLU nodes, the closest it has to our leaky ReLU,
    are both trained on the 50 000 fitting images with the selected settings, for n_seeds seeds each, and
    scored on the 10 000 validation images after the selected number of epochs.
    """
    network = final_settings()[0]
    X_fit, X_val, y_fit, y_val = fit_and_validation()
    ours = [dict(network, seed=SEED + s) for s in range(n_seeds)]
    jobs = [delayed(run)(**c) for c in ours] + [delayed(_library_network)(network, "relu", SEED + s, X_fit, y_fit, X_val)
                                               for s in range(n_seeds)]
    out = Parallel(n_jobs=2 * n_seeds)(jobs)
    def summary(accuracies):
        return {"accuracies": accuracies, "mean": float(np.mean(accuracies)),
                "std": float(np.std(accuracies, ddof=1)),
                "se_one_run": float(np.sqrt(np.mean(accuracies) * (1 - np.mean(accuracies)) / len(y_val)))}
    save_results("mnist_library_validation", {
        "settings": network, "n_validation": len(y_val),
        "ours": summary([r["last_val"] for r in out[:n_seeds]]),
        "scikit_learn_relu": summary([accuracy(r[0], y_val) for r in out[n_seeds:]]) | {"seconds": [r[1] for r in out[n_seeds:]]}})


def saturation(net, theta, X):
    """Per hidden sigmoid layer: share of outputs on X below 0.01 or above 0.99, where the derivative is
    below 0.01, and the mean derivative relative to its largest value 1/4. LLM-assisted: Claude."""
    outputs = net.hidden_outputs(X, theta)
    return {"saturated": [float(np.mean((a < 0.01) | (a > 0.99))) for a in outputs],
            "derivative": [float(np.mean(4.0 * a * (1.0 - a))) for a in outputs]}


def weight_changes(net, theta, X):
    """Relative change of each layer's weights; net.theta holds the initial weights until training ends.

    LLM-assisted: Claude."""
    return net.layer_changes(theta, net.theta)


def mechanisms():
    """Measurements behind the explanations in the report, on the fitting and validation images only.

    Saturation of the sigmoid layers in the scaling study, at each variant's best learning rate; how far
    plain SGD and Adam move the weights of each layer of a deep sigmoid network; how confident the
    untrained network is; and how similar the mean images of the digits are.
    """
    best = {}
    for r in load_results("mnist_scaling_pairs")["scaling"]:
        key = (r["input_scale"], r["init"])
        if key not in best or r["best_val"] > best[key]["best_val"]:
            best[key] = r
    scaling = [dict(hidden=(128, 128), activation="sigmoid", optimizer="sgd", gamma=r["gamma"], input_scale=r["input_scale"],
                    init=r["init"], epochs=15, probe=saturation) for r in best.values()]
    depth = [dict(hidden=(128,) * d, activation="sigmoid", optimizer=o, gamma=g, epochs=15, probe=weight_changes)
             for o, g in (("sgd", 0.1), ("adam", 1e-3)) for d in (6, 10)]
    runs = run_all(scaling + depth)

    X_fit, X_val, y_fit, y_val = fit_and_validation()
    p = NeuralNetwork([784, 128, 10], ["relu", "softmax"], seed=SEED).predict(X_val)  # untrained network of scaling_and_pairs
    wrong = np.argmax(p, axis=1) != y_val
    initial = {"median_largest_probability": float(np.median(p.max(axis=1))), "largest_probability": float(p.max()),
               "share_wrong_above_half": float(np.mean(wrong & (p.max(axis=1) > 0.5))),
               "share_wrong_above_0.9": float(np.mean(wrong & (p.max(axis=1) > 0.9)))}

    means = np.array([X_fit[y_fit == k].mean(axis=0) for k in range(10)])
    unit = means / np.linalg.norm(means, axis=1, keepdims=True)
    cosine = unit @ unit.T
    ranking = sorted(([int(i), int(j), float(cosine[i, j])] for i in range(10) for j in range(i + 1, 10)), key=lambda t: -t[2])
    save_results("mnist_mechanisms", {"saturation": runs[:len(scaling)], "depth": runs[len(scaling):], "initial_confidence": initial,
                                      "mean_image_cosine": cosine, "pair_ranking": ranking})


EXPERIMENTS = {"baseline_scan": baseline_scan, "optimizer_scan": optimizer_scan, "architecture_maps": architecture_maps,
               "depth_study": depth_study, "scaling_and_pairs": scaling_and_pairs, "penalty_maps": penalty_maps,
               "mechanisms": mechanisms, "library_validation": library_validation, "final_evaluation": final_evaluation}
SELECTION = [k for k in EXPERIMENTS if k != "final_evaluation"]  # never read the test images

if __name__ == "__main__":
    import mnist_experiments  # functions from a named module can be sent to the worker processes

    for name in sys.argv[1:] or SELECTION:
        start_time = time.time()
        mnist_experiments.EXPERIMENTS[name]()
        print(f"{name} finished in {time.time() - start_time:.0f} s", flush=True)
