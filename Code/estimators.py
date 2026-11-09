"""
The neural network wrapped as scikit-learn estimators.

With this interface Pipeline keeps the scaler inside every fold, and KFold, cross_validate and
GridSearchCV do the folds, the grid and the bookkeeping. The network itself is in network.py.

Optimizer names: "gd" is plain gradient descent on all the samples at once, "sgd" the same update
rule on minibatches, and "rmsprop" and "adam" the adaptive rules on minibatches.

LLM-assisted
------------
Code level 2. Tool: Claude Opus 5.5 (Anthropic, Claude Code desktop app, November 2026).
Role: the estimator interface and the rules for fitting and predicting are the author's,
following the estimator rules given in the course material; Claude drafted the two wrapper
endpoints around them.
Verification: tests/test_estimators.py compares the wrappers with direct calls to the network.
OpenAI Codex (November 2026) also refactored update_rule and the two fit methods for readability.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from network import NeuralNetwork, one_hot, train
from optimizers import GD, Adam, RMSprop

UPDATE_RULES = {"gd": GD, "sgd": GD, "rmsprop": RMSprop, "adam": Adam}


def update_rule(optimizer, gamma, batch_size, n_samples):
    """The update rule of project 1 and the batch size that goes with an optimizer name.

    LLM-assisted: Claude supplied the helper; OpenAI Codex refactored it for readability.
    """
    rule = UPDATE_RULES[optimizer](gamma)
    batch = n_samples if optimizer == "gd" else batch_size
    return rule, batch


class NeuralNetRegressor(BaseEstimator, RegressorMixin):
    """Regression network: linear output and half-squared error, with a centred target.

    The target is centred with the training mean, as in project 1, so that the penalty never acts on
    the mean level of the data. Predictions are returned in the original units.

    LLM-assisted: drafted with Claude (Claude Code, November 2026); see the module docstring.
    """

    def __init__(self, hidden=(50,), activation="sigmoid", optimizer="adam", gamma=1e-3, penalty=None,
                 lam=0.0, penalize_biases=True, epochs=1000, batch_size=16, init="auto", seed=2026):
        """Store the hyperparameters under their own names, as scikit-learn requires."""
        self.hidden = hidden
        self.activation = activation
        self.optimizer = optimizer
        self.gamma = gamma
        self.penalty = penalty
        self.lam = lam
        self.penalize_biases = penalize_biases
        self.epochs = epochs
        self.batch_size = batch_size
        self.init = init
        self.seed = seed

    def fit(self, X, y, on_epoch=None):
        """Build the network from the seed and train it, calling on_epoch after each epoch.

        LLM-assisted: OpenAI Codex made the network setup explicit.
        """
        self.y_mean_ = float(np.mean(y))
        sizes = [X.shape[1], *self.hidden, 1]
        activations = [self.activation] * len(self.hidden) + ["identity"]
        self.net_ = NeuralNetwork(sizes, activations, cost="mse", penalty=self.penalty,
                                  penalize_biases=self.penalize_biases, init=self.init, seed=self.seed)
        rule, batch = update_rule(self.optimizer, self.gamma, self.batch_size, len(X))
        with np.errstate(all="ignore"):  # a diverging run is a result, reported by the large error
            train(self.net_, X, (y - self.y_mean_)[:, None], rule, self.epochs, batch, self.lam,
                  self.seed, on_epoch)
        return self

    def predict(self, X, theta=None):
        """Predictions as a 1D array; a diverged network gets a huge finite error instead of NaN."""
        output = self.net_.predict(X, theta)[:, 0] + self.y_mean_
        return np.nan_to_num(output, nan=1e6, posinf=1e6, neginf=-1e6)


class NeuralNetClassifier(BaseEstimator, ClassifierMixin):
    """Classification network: softmax output and cross entropy. hidden=() is logistic regression.

    LLM-assisted: drafted with Claude (Claude Code, November 2026); see the module docstring.
    """

    def __init__(self, hidden=(128,), activation="relu", optimizer="adam", gamma=1e-3, penalty=None,
                 lam=0.0, penalize_biases=True, epochs=20, batch_size=128, init="auto", seed=2026):
        """Store the hyperparameters under their own names, as scikit-learn requires."""
        self.hidden = hidden
        self.activation = activation
        self.optimizer = optimizer
        self.gamma = gamma
        self.penalty = penalty
        self.lam = lam
        self.penalize_biases = penalize_biases
        self.epochs = epochs
        self.batch_size = batch_size
        self.init = init
        self.seed = seed

    def fit(self, X, y, on_epoch=None):
        """Build the network from the seed and train it on integer labels.

        LLM-assisted: OpenAI Codex made the network setup and label encoding explicit.
        """
        self.classes_ = np.unique(y)
        sizes = [X.shape[1], *self.hidden, len(self.classes_)]
        activations = [self.activation] * len(self.hidden) + ["softmax"]
        self.net_ = NeuralNetwork(sizes, activations, cost="cross_entropy", penalty=self.penalty,
                                  penalize_biases=self.penalize_biases, init=self.init, seed=self.seed)
        rule, batch = update_rule(self.optimizer, self.gamma, self.batch_size, len(X))
        with np.errstate(all="ignore"):
            labels = np.searchsorted(self.classes_, y)
            targets = one_hot(labels, len(self.classes_))
            train(self.net_, X, targets, rule, self.epochs, batch, self.lam, self.seed, on_epoch)
        return self

    def predict_proba(self, X, theta=None):
        """Class probabilities, one row per sample."""
        return np.nan_to_num(self.net_.predict(X, theta), nan=1.0 / len(self.classes_))

    def predict(self, X, theta=None):
        """The most probable class of each sample."""
        return self.classes_[np.argmax(self.predict_proba(X, theta), axis=1)]
