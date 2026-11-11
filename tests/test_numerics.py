"""Numerical correctness checks independent of the ApacheJIT experiment results."""
from __future__ import annotations

import time

import numpy as np
import pytest

from src.model import TanhMLP
from src.optimizers import optimize


def small_problem(l2=0.07):
    rng = np.random.default_rng(2107)
    X = rng.normal(size=(9, 3))
    y = np.array([0, 0, 1, 0, 0, 0, 1, 0, 1], dtype=float)
    model = TanhMLP(X, y, hidden=4, l2=l2, positive_weight=3.7)
    theta = rng.normal(scale=0.3, size=model.size)
    return model, theta


def test_all_parameter_gradient_matches_central_differences():
    # Every block, including both bias blocks, is nonzero in this check.
    model, theta = small_problem()
    loss, analytical = model.value_grad(theta)
    numerical = np.empty_like(theta)
    for j in range(theta.size):
        step = 1e-6 * max(1.0, abs(theta[j]))
        right, left = theta.copy(), theta.copy()
        right[j] += step
        left[j] -= step
        numerical[j] = (model.fun(right) - model.fun(left)) / (2 * step)
    assert model.fun(theta) == pytest.approx(loss, rel=1e-14, abs=1e-14)
    np.testing.assert_allclose(analytical, numerical, rtol=3e-6, atol=2e-9)


def test_regularization_includes_all_weights_and_biases():
    regularized, theta = small_problem(l2=0.07)
    unregularized, theta2 = small_problem(l2=0.0)
    np.testing.assert_array_equal(theta, theta2)
    f1, g1 = regularized.value_grad(theta)
    f0, g0 = unregularized.value_grad(theta)
    assert f1 - f0 == pytest.approx(0.035 * (theta @ theta), abs=1e-14)
    np.testing.assert_allclose(g1 - g0, 0.07 * theta, rtol=1e-12, atol=1e-14)


def test_weighted_loss_uses_the_sum_of_weights_not_sample_count():
    model, theta = small_problem(l2=0)
    logits = model.logits(theta)
    raw_weights = np.where(model.y == 1, 3.7, 1.0)
    per_sample = np.logaddexp(0, (1 - 2 * model.y) * logits)
    expected = np.sum(raw_weights * per_sample) / np.sum(raw_weights)
    assert model.fun(theta) == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("bias", [-1000.0, 1000.0])
def test_extreme_logits_remain_finite(bias):
    model, theta = small_problem(l2=1e-4)
    theta[-1] = bias
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        loss, grad = model.value_grad(theta)
        probabilities = model.predict_proba(theta)
    assert np.isfinite(loss)
    assert np.all(np.isfinite(grad))
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert model.fun(theta) == pytest.approx(loss)


def quadratic():
    matrix = np.array([[8.0, 1.0, 0.0], [1.0, 3.0, 0.2], [0.0, 0.2, 1.0]])
    optimum = np.array([0.7, -1.3, 2.1])
    counts = {"fun_only": 0, "value_grad": 0}

    def fun(x):
        counts["fun_only"] += 1
        residual = x - optimum
        return 0.5 * residual @ matrix @ residual

    def value_grad(x):
        counts["value_grad"] += 1
        residual = x - optimum
        return 0.5 * residual @ matrix @ residual, matrix @ residual

    return fun, value_grad, optimum, matrix, counts


@pytest.mark.parametrize("method,learning_rate", [("gd", 1.0), ("momentum", 0.04),
                                                 ("adam", 0.1), ("lbfgs", 1.0)])
def test_solvers_on_spd_problem_and_trace_alignment(method, learning_rate):
    fun, value_grad, optimum, matrix, counts = quadratic()
    initial = np.array([-2.0, 2.0, -1.0])
    initial_copy = initial.copy()
    result = optimize(value_grad, fun, initial, method, {
        "time_budget_s": 5.0, "max_iterations": 1000, "grad_tol": 1e-8,
        "learning_rate": learning_rate, "snapshot_every": 7,
    })
    np.testing.assert_array_equal(initial, initial_copy)
    assert result["status"] == "converged", result["status"]
    assert result["loss"] < result["history"][0]["loss"] * 1e-10
    np.testing.assert_allclose(result["x"], optimum, atol=2e-6, rtol=0)
    expected_gradient = matrix @ (result["x"] - optimum)
    expected_loss = 0.5 * (result["x"] - optimum) @ expected_gradient
    last = result["history"][-1]
    assert last["loss"] == pytest.approx(expected_loss, abs=1e-14)
    assert last["grad_inf"] == pytest.approx(np.linalg.norm(expected_gradient, np.inf))
    assert last["iteration"] == result["iterations"]
    assert result["n_fun"] == counts["fun_only"] + counts["value_grad"]
    assert result["n_grad"] == counts["value_grad"]
    assert result["snapshots"][0]["iteration"] == 0
    np.testing.assert_array_equal(result["snapshots"][-1]["x"], result["x"])
    for snapshot in result["snapshots"]:
        point = snapshot["x"]
        reference = 0.5 * (point - optimum) @ matrix @ (point - optimum)
        row = result["history"][snapshot["iteration"]]
        assert row["loss"] == pytest.approx(reference, abs=1e-14)
        assert snapshot["elapsed_s"] == row["elapsed_s"]
    assert np.all(np.diff([row["elapsed_s"] for row in result["history"]]) >= 0)
    if method in ("gd", "lbfgs"):
        assert np.all(np.diff([row["loss"] for row in result["history"]]) <= 1e-12)


@pytest.mark.parametrize("method", ["gd", "momentum", "adam", "lbfgs"])
def test_evaluation_budget_preserves_last_evaluated_point(method):
    fun, value_grad, _, _, _ = quadratic()
    initial = np.zeros(3)
    result = optimize(value_grad, fun, initial, method, {"max_evaluations": 1})
    assert result["status"] == "evaluation_limit"
    assert result["n_fun"] == result["n_grad"] == 1
    np.testing.assert_array_equal(result["x"], initial)
    assert len(result["history"]) == 1


def test_budget_expiry_between_trial_loss_and_gradient_does_not_accept_trial():
    fun, value_grad, _, _, _ = quadratic()
    initial = np.zeros(3)
    result = optimize(value_grad, fun, initial, "gd", {
        "max_evaluations": 2, "learning_rate": 0.05,
    })
    assert result["status"] == "evaluation_limit"
    assert result["n_fun"] == 2
    assert result["n_grad"] == 1
    np.testing.assert_array_equal(result["x"], initial)
    assert result["history"][-1]["n_fun"] == 1


def test_time_budget_allows_only_one_already_started_slow_evaluation():
    calls = []

    def slow_value_grad(x):
        calls.append(x.copy())
        time.sleep(0.015)
        return float(x @ x), 2*x

    result = optimize(slow_value_grad, lambda x: float(x @ x), np.ones(2), "adam", {
        "time_budget_s": 0.005,
    })
    assert result["status"] == "time_limit"
    assert len(calls) == 1
    assert result["iterations"] == 0
    assert result["elapsed_s"] >= 0.005


def test_nonfinite_trial_returns_failure_and_preserves_previous_point():
    def value_grad(x):
        if np.max(np.abs(x)) > 3:
            return np.inf, np.full_like(x, np.nan)
        return float(x @ x), 2*x

    initial = np.ones(2)
    result = optimize(value_grad, lambda x: float(x @ x), initial, "momentum", {
        "learning_rate": 10.0,
    })
    assert result["status"] == "numerical_failure"
    assert result["diagnostics"]["error"]
    np.testing.assert_array_equal(result["x"], initial)
    assert result["history"][-1]["loss"] == 2.0


def test_lbfgs_handles_nonconvex_rosenbrock_valley():
    def fun(x):
        return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

    def value_grad(x):
        delta = x[1] - x[0]**2
        return fun(x), np.array([-2*(1-x[0])-400*x[0]*delta, 200*delta])

    result = optimize(value_grad, fun, np.array([-1.2, 1.0]), "lbfgs", {
        "time_budget_s": 5.0, "max_iterations": 500, "grad_tol": 1e-10,
        "snapshot_every": 1,
    })
    assert result["status"] == "converged"
    np.testing.assert_allclose(result["x"], np.ones(2), atol=1e-6, rtol=0)
    assert result["loss"] < 1e-14
    if result["diagnostics"]["armijo_fallbacks"] == 0:
        # Independently verify the accepted steps satisfy both Wolfe conditions.
        for before, after, row in zip(result["snapshots"][:-1], result["snapshots"][1:], result["history"][1:]):
            alpha = row["step_size"]
            direction = (after["x"]-before["x"])/alpha
            f0, g0 = value_grad(before["x"])
            f1, g1 = value_grad(after["x"])
            initial_slope = g0 @ direction
            assert f1 <= f0 + 1e-4*alpha*initial_slope + 1e-13
            assert abs(g1 @ direction) <= 0.9*abs(initial_slope) + 1e-12
