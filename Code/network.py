"""
A feed-forward neural network written from scratch with NumPy, trained with backpropagation.

Conventions (lecture notes, Ch. 8): samples are stored in the rows of the input, the weight matrix
of layer l has shape (M_{l-1}, M_l), and a layer computes Z^l = A^{l-1} W^l + b^l, A^l = f(Z^l).

All weights and biases live in one flat vector theta, and NeuralNetwork.gradient has the signature
grad_fn(theta, X, y, lam) of the project-1 optimizers. The update rules and the mini-batch driver in
optimizers.py are therefore reused without any change: they never see that the gradient comes from
a network.

LLM-assisted
------------
Code level 2. Tool: Claude Opus 5.5 (Anthropic, Claude Code desktop app, November 2026).
Role: the author provided the core activations, costs, forward pass, backward pass and penalties.
Claude helped debug the backward loop and supplied supporting helpers. OpenAI Codex (Codex
desktop app, November 2026) refactored parameter unpacking, layer evaluation and gradient assembly.
The affected methods describe that assistance; the file-level declaration does not apply to
every function.
Verification: tests/test_network.py compares the gradient with Autograd and with finite differences
for every activation, cost and penalty, and recovers closed-form OLS and Ridge and scikit-learn's
logistic regression with networks that have no hidden layer.
"""

import numpy as np

from optimizers import stochastic_gradient_descent


# ----------------------------------------------------------------------------------------------
# Activation functions and their derivatives
# ----------------------------------------------------------------------------------------------
def sigmoid(z):
    """Logistic function, evaluated so that large |z| cannot overflow."""
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def sigmoid_prime(z):
    """Derivative s(1 - s) of the logistic function."""
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z):
    """Rectified linear unit max(0, z)."""
    return np.maximum(z, 0.0)


def relu_prime(z):
    """Derivative of ReLU, with the value 0 assigned at the kink z = 0."""
    return (z > 0).astype(float)


LEAKY_SLOPE = 0.01  # slope alpha of the negative branch, the default in the lecture notes


def leaky_relu(z):
    """Leaky ReLU: z for z > 0 and alpha*z otherwise."""
    return np.where(z > 0, z, LEAKY_SLOPE * z)


def leaky_relu_prime(z):
    """Derivative of leaky ReLU, with the value alpha assigned at z = 0."""
    return np.where(z > 0, 1.0, LEAKY_SLOPE)


def identity(z):
    """Linear output activation."""
    return z


def identity_prime(z):
    """Derivative of the linear activation."""
    return np.ones_like(z)


def softmax(z):
    """Softmax of each row. Subtracting the largest score prevents overflow."""
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


# softmax has no element-wise derivative; output_error handles it directly
ACTIVATIONS = {"sigmoid": (sigmoid, sigmoid_prime), "relu": (relu, relu_prime),
               "leaky_relu": (leaky_relu, leaky_relu_prime), "identity": (identity, identity_prime),
               "softmax": (softmax, None)}
HE_FAMILY = ("relu", "leaky_relu")


# ----------------------------------------------------------------------------------------------
# Cost functions, averaged over the samples
# ----------------------------------------------------------------------------------------------
def half_squared_error(a, y):
    """(1/2n) sum_i sum_k (a_ik - y_ik)^2, which is half the MSE for one output."""
    return 0.5 * np.sum((a - y) ** 2) / len(a)


def cross_entropy(a, y):
    """Multiclass cross entropy with one-hot targets."""
    return -np.sum(y * np.log(np.clip(a, 1e-300, 1.0))) / len(a)


def binary_cross_entropy(a, y):
    """Binary cross entropy, the two-class special case."""
    a = np.clip(a, 1e-15, 1.0 - 1e-15)
    return -np.sum(y * np.log(a) + (1.0 - y) * np.log(1.0 - a)) / len(a)


COSTS = {"mse": half_squared_error, "cross_entropy": cross_entropy,
         "binary_cross_entropy": binary_cross_entropy}
MATCHED_PAIRS = {("mse", "identity"), ("cross_entropy", "softmax"), ("binary_cross_entropy", "sigmoid")}


def output_error(cost, activation, z, a, y):
    """Per-sample output error delta^L = dC/dz^L, before dividing by the number of samples.

    The three matched pairs reduce to a - y, without any derivative of f that could vanish. The
    mismatched pairs are kept to measure what the mismatch costs.
    """
    if (cost, activation) in MATCHED_PAIRS:
        return a - y
    if cost != "mse":
        raise ValueError(f"output error not implemented for cost {cost!r} with output {activation!r}")
    residual = a - y
    if activation == "softmax":  # the softmax Jacobian a_k (delta_kj - a_j) is not diagonal
        return a * residual - a * np.sum(a * residual, axis=1, keepdims=True)
    return ACTIVATIONS[activation][1](z) * residual


# ----------------------------------------------------------------------------------------------
# The network
# ----------------------------------------------------------------------------------------------
class NeuralNetwork:
    """Dense network with a free choice of layers, nodes, activations, cost and penalty.

    sizes       : nodes per layer, input and output included; [1, 50, 1] has one hidden layer.
                  [784, 10] has no hidden layer, which is logistic regression.
    activations : one name per layer after the input, see ACTIVATIONS.
    cost        : a key of COSTS.
    penalty     : None, "l1" or "l2"; lam is passed to cost() and gradient().
    penalize_biases : whether the biases enter the penalty together with the weights.
    init        : "auto" draws weights from N(0, 2/M_in) for the ReLU family and N(0, 1/M_in)
                  otherwise; "unit" draws them from N(0, 1), to study what bad scaling does.
    seed        : seed of the PCG64 generator that draws the initial weights.

    See the module and individual methods for the scope of LLM assistance.
    """

    def __init__(self, sizes, activations, cost="mse", penalty=None, penalize_biases=True,
                 init="auto", seed=None):
        """Check the network description and draw its initial parameters.

        LLM-assisted: OpenAI Codex, readability refactor.
        """
        if len(activations) != len(sizes) - 1:
            raise ValueError("need one activation for each layer after the input")
        if cost not in COSTS or penalty not in (None, "l1", "l2") or init not in ("auto", "unit"):
            raise ValueError("unknown cost, penalty or init")
        self.sizes = list(sizes)
        self.activations = list(activations)
        self.cost_name = cost
        self.penalty = penalty
        self.penalize_biases = penalize_biases
        self.functions = [ACTIVATIONS[name] for name in activations]
        self.theta = self._initial_parameters(init, seed)

    def _initial_parameters(self, init, seed):
        """Normal weights scaled by the number of inputs, with biases 0.01.

        LLM-assisted: OpenAI Codex, readability refactor.
        """
        rng = np.random.default_rng(seed)
        parts = []
        for n_in, n_out, name in zip(self.sizes[:-1], self.sizes[1:], self.activations):
            if init == "unit":
                std = 1.0
            else:
                std = np.sqrt((2.0 if name in HE_FAMILY else 1.0) / n_in)
            parts.append(rng.normal(0.0, std, n_in * n_out))
            parts.append(np.full(n_out, 0.01))
        return np.concatenate(parts)

    def layers(self, theta):
        """List of (W, b) views into a parameter or gradient vector.

        LLM-assisted: OpenAI Codex made the unpacking steps explicit.
        """
        layers = []
        start = 0
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            end_weights = start + n_in * n_out
            end_biases = end_weights + n_out
            W = theta[start:end_weights].reshape(n_in, n_out)
            b = theta[end_weights:end_biases]
            layers.append((W, b))
            start = end_biases
        return layers

    def predict(self, X, theta=None):
        """Feed-forward pass; returns the output layer A^L.

        LLM-assisted: OpenAI Codex, readability refactor.
        """
        if theta is None:
            theta = self.theta
        a = X
        for (W, b), (f, _) in zip(self.layers(theta), self.functions):
            a = f(a @ W + b)
        return a

    def hidden_outputs(self, X, theta=None):
        """Outputs A^l of every hidden layer.

        LLM-assisted: OpenAI Codex, readability refactor.
        """
        if theta is None:
            theta = self.theta
        a = X
        outputs = []
        for (W, b), (f, _) in zip(self.layers(theta)[:-1], self.functions):
            a = f(a @ W + b)
            outputs.append(a)
        return outputs

    def cost(self, theta, X, Y, lam=0.0):
        """Data term plus lam times the penalty.

        LLM-assisted: OpenAI Codex, readability refactor.
        """
        value = COSTS[self.cost_name](self.predict(X, theta), Y)
        if self.penalty is None or lam == 0.0:
            return value
        parts = [W for W, _ in self.layers(theta)]
        if self.penalize_biases:
            parts += [b for _, b in self.layers(theta)]
        if self.penalty == "l2":
            norm = sum(np.sum(p ** 2) for p in parts)
        else:
            norm = sum(np.sum(np.abs(p)) for p in parts)
        return value + lam * norm

    def _penalty_gradient(self, p, lam):
        """Gradient of lam*||p||^2, or the subgradient of lam*||p||_1 with sign(0) = 0."""
        if self.penalty == "l2":
            return 2.0 * lam * p
        if self.penalty == "l1":
            return lam * np.sign(p)
        return 0.0

    def gradient(self, theta, X, Y, lam=0.0):
        """Gradient of cost() with respect to theta by backpropagation.

        One forward sweep stores every Z^l and A^l. One backward sweep then applies, for each layer,
        dC/dW^l = (A^{l-1})^T Delta^l,  dC/db^l = column sums of Delta^l,
        Delta^{l-1} = (Delta^l (W^l)^T) o f'(Z^{l-1}),
        starting from the output error Delta^L.

        LLM-assisted: Claude helped debug the backward pass. OpenAI Codex
        refactored the backward loop and gradient assembly for readability.
        """
        layers = self.layers(theta)
        a = [X]
        z = []
        for (W, b), (f, _) in zip(layers, self.functions):
            z.append(a[-1] @ W + b)
            a.append(f(z[-1]))

        delta = output_error(self.cost_name, self.activations[-1], z[-1], a[-1], Y) / len(X)
        gradients = [None] * len(layers)
        for layer in range(len(layers) - 1, -1, -1):
            W, b = layers[layer]
            dW = a[layer].T @ delta + self._penalty_gradient(W, lam)
            db = np.sum(delta, axis=0)
            if self.penalize_biases:
                db = db + self._penalty_gradient(b, lam)
            gradients[layer] = (dW, db)
            if layer > 0:
                derivative = self.functions[layer - 1][1]
                delta = (delta @ W.T) * derivative(z[layer - 1])

        flat_gradients = []
        for dW, db in gradients:
            flat_gradients.append(dW.ravel())
            flat_gradients.append(db.ravel())
        return np.concatenate(flat_gradients)

    def layer_gradient_norms(self, X, Y, theta=None):
        """Frobenius norm of dC/dW^l for every layer.

        LLM-assisted: Claude, diagnostic helper.
        """
        g = self.gradient(self.theta if theta is None else theta, X, Y)
        return [float(np.linalg.norm(dW)) for dW, _ in self.layers(g)]

    def layer_changes(self, theta, theta0):
        """Relative change ||W^l - W^l_0|| / ||W^l_0|| of the weights of every layer since theta0.

        LLM-assisted: Claude.
        """
        return [float(np.linalg.norm(W - W0) / np.linalg.norm(W0))
                for (W, _), (W0, _) in zip(self.layers(theta), self.layers(theta0))]


def train(net, X, Y, optimizer, epochs, batch_size, lam=0.0, seed=None, on_epoch=None):
    """Fit net with the mini-batch driver of project 1; batch_size >= n gives plain gradient descent.

    on_epoch(theta) is called after every epoch. The minibatches are drawn by reshuffling with a
    PCG64 generator with the given seed. Returns the information dict of the driver.

    LLM-assisted: Claude.
    """
    theta, info = stochastic_gradient_descent(net.gradient, X, Y, lam, net.theta, optimizer, epochs,
                                              min(batch_size, len(X)), seed=seed, callback=on_epoch)
    net.theta = theta
    return info


def one_hot(y, n_classes=10):
    """Integer labels to one-hot rows. LLM-assisted: Claude."""
    return np.eye(n_classes)[y]


def accuracy(probabilities, y):
    """Share of samples whose most probable class equals the integer label y. LLM-assisted: Claude."""
    return float(np.mean(np.argmax(probabilities, axis=1) == y))
