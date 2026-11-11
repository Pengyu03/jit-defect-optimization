"""Smooth, nonconvex, full-batch weighted binary cross-entropy objective."""
from __future__ import annotations

import numpy as np
from scipy.special import expit


class TanhMLP:
    """One hidden layer; L2 penalizes every parameter, including biases.

    Parameters are packed as W (d,h), b (h,), v (h,), c (1,).
    Sample weights are fixed from training labels and normalized by their sum.
    """

    def __init__(self, X, y, hidden=32, l2=1e-4, positive_weight=None):
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.d, self.h = self.X.shape[1], int(hidden)
        self.l2 = float(l2)
        n1 = int(self.y.sum())
        n0 = len(self.y) - n1
        if not n1 or not n0:
            raise ValueError("Both training classes are required")
        self.positive_weight = n0 / n1 if positive_weight is None else float(positive_weight)
        weights = np.where(self.y == 1, self.positive_weight, 1.0)
        self.weights = weights / weights.sum()
        self.size = self.d * self.h + 2 * self.h + 1

    def unpack(self, theta):
        k = self.d * self.h
        return theta[:k].reshape(self.d, self.h), theta[k:k+self.h], theta[k+self.h:k+2*self.h], theta[-1]

    def initialize(self, seed):
        rng = np.random.default_rng(seed)
        wlim = np.sqrt(6.0 / (self.d + self.h))
        vlim = np.sqrt(6.0 / (self.h + 1))
        return np.concatenate((rng.uniform(-wlim, wlim, self.d*self.h),
                               np.zeros(self.h), rng.uniform(-vlim, vlim, self.h), [0.0]))

    def logits(self, theta, X=None):
        W, b, v, c = self.unpack(theta)
        inputs = self.X if X is None else np.asarray(X, dtype=np.float64)
        return np.tanh(inputs @ W + b) @ v + c

    def predict_proba(self, theta, X=None):
        return expit(self.logits(theta, X))

    def fun(self, theta):
        z = self.logits(theta)
        return float(self.weights @ (np.logaddexp(0.0, z) - self.y*z)
                     + 0.5*self.l2*np.dot(theta, theta))

    def value_grad(self, theta):
        W, b, v, c = self.unpack(theta)
        H = np.tanh(self.X @ W + b)
        z = H @ v + c
        loss = self.weights @ (np.logaddexp(0.0, z) - self.y*z)
        loss += 0.5*self.l2*np.dot(theta, theta)
        delta = self.weights * (expit(z) - self.y)
        dh = (delta[:, None] * v[None, :]) * (1.0-H*H)
        grad = np.concatenate(((self.X.T @ dh).ravel(), dh.sum(axis=0), H.T @ delta, [delta.sum()]))
        grad += self.l2 * theta
        return float(loss), grad
