"""Full-batch solvers for a shared, differentiable objective.

The caller supplies ``value_grad(x) -> (loss, gradient)`` and ``fun(x) -> loss``.
Regularization belongs inside those functions, including for Adam (this is not
AdamW). All methods use float64 vectors. L-BFGS implements the limited-memory
two-loop recursion and uses SciPy's strong-Wolfe line search.

``n_fun`` counts actual objective computations, including calls to value_grad;
``n_grad`` counts actual gradient computations. Cached results do not increment
either counter. Thus n_fun - n_grad is the number of loss-only computations.
Elapsed time includes solver setup, line searches, recording and snapshots, but
does not include any validation/test evaluation performed by the caller.

Stopping uses ||gradient||_inf / max(1, ||initial_gradient||_inf) <= grad_tol.
It certifies only approximate first-order stationarity, not optimality.
"""

from __future__ import annotations

from collections import OrderedDict
import math
from time import perf_counter
import warnings

import numpy as np


class _BudgetExceeded(Exception):
    def __init__(self, status: str):
        self.status = status


class _NumericalFailure(Exception):
    pass


class _Oracle:
    """Count genuine computations and cache the last few line-search points."""

    def __init__(self, value_grad, fun, start, time_budget, max_evaluations):
        self.value_grad_impl = value_grad
        self.fun_impl = fun
        self.start = start
        self.time_budget = time_budget
        self.max_evaluations = max_evaluations
        self.n_fun = 0
        self.n_grad = 0
        self.cache = OrderedDict()

    def check_budget(self):
        if perf_counter() - self.start >= self.time_budget:
            raise _BudgetExceeded("time_limit")
        if self.n_fun >= self.max_evaluations:
            raise _BudgetExceeded("evaluation_limit")

    @staticmethod
    def key(x):
        return np.asarray(x, dtype=np.float64).tobytes()

    def _store(self, key, loss, grad):
        self.cache[key] = (loss, grad)
        self.cache.move_to_end(key)
        while len(self.cache) > 8:
            self.cache.popitem(last=False)

    @staticmethod
    def _validate_x(x):
        if not np.all(np.isfinite(x)):
            raise _NumericalFailure("A trial parameter vector was non-finite.")

    def fun(self, x):
        self._validate_x(x)
        key = self.key(x)
        if key in self.cache:
            return self.cache[key][0]
        self.check_budget()
        self.n_fun += 1
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            loss = float(self.fun_impl(x))
        if not math.isfinite(loss):
            raise _NumericalFailure("The objective returned a non-finite loss.")
        self._store(key, loss, None)
        return loss

    def value_grad(self, x):
        self._validate_x(x)
        key = self.key(x)
        if key in self.cache and self.cache[key][1] is not None:
            loss, grad = self.cache[key]
            return loss, grad.copy()
        self.check_budget()
        self.n_fun += 1
        self.n_grad += 1
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            loss, grad = self.value_grad_impl(x)
        loss = float(loss)
        grad = np.asarray(grad, dtype=np.float64)
        if grad.shape != x.shape:
            raise ValueError(f"Gradient shape {grad.shape} does not match {x.shape}.")
        if not math.isfinite(loss) or not np.all(np.isfinite(grad)):
            raise _NumericalFailure("The objective returned a non-finite loss/gradient.")
        grad = grad.copy()
        self._store(key, loss, grad)
        return loss, grad.copy()

    def grad(self, x):
        return self.value_grad(x)[1]


def _armijo(oracle, x, loss, grad, direction, initial_step, config):
    slope = float(np.dot(grad, direction))
    if not math.isfinite(slope) or slope >= 0:
        return None
    step = initial_step
    for _ in range(config["max_line_search_steps"]):
        candidate = x + step * direction
        if np.array_equal(candidate, x):
            return None
        trial_loss = oracle.fun(candidate)
        if trial_loss <= loss + config["armijo_c1"] * step * slope:
            return step
        step *= config["backtracking_factor"]
        if step < config["min_step"]:
            return None
    return None


def _lbfgs_direction(grad, pairs):
    """Return -H_k grad with scalar initial Hessian inverse gamma * I."""
    q = grad.copy()
    alphas = []
    for s, y, rho in reversed(pairs):
        alpha = rho * float(np.dot(s, q))
        alphas.append(alpha)
        q -= alpha * y
    if pairs:
        s, y, _ = pairs[-1]
        gamma = float(np.dot(s, y) / np.dot(y, y))
    else:
        gamma = 1.0
    r = gamma * q
    for (s, y, rho), alpha in zip(pairs, reversed(alphas)):
        beta = rho * float(np.dot(y, r))
        r += s * (alpha - beta)
    return -r


def _settings(method, config):
    settings = {
        "time_budget_s": 60.0,
        "max_iterations": 10000,
        "max_evaluations": 100000,
        "grad_tol": 1e-4,
        "learning_rate": {"gd": 1.0, "momentum": 1e-2,
                          "adam": 1e-3, "lbfgs": 1.0}[method],
        "momentum": 0.9,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "history_size": 10,
        "snapshot_every": 10,
        "armijo_c1": 1e-4,
        "wolfe_c2": 0.9,
        "backtracking_factor": 0.5,
        "max_line_search_steps": 40,
        "min_step": 1e-16,
        "curvature_threshold": 1e-10,
    }
    if config:
        unknown = set(config) - set(settings)
        if unknown:
            raise ValueError(f"Unknown solver configuration keys: {sorted(unknown)}")
        settings.update(config)
    for key in ("max_iterations", "max_evaluations", "history_size",
                "snapshot_every", "max_line_search_steps"):
        value = settings[key]
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("time_budget_s", "learning_rate", "epsilon", "min_step"):
        value = float(settings[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite.")
    if not math.isfinite(settings["grad_tol"]) or settings["grad_tol"] < 0:
        raise ValueError("grad_tol must be finite and non-negative.")
    if not 0 <= settings["curvature_threshold"] < 1:
        raise ValueError("curvature_threshold must be in [0, 1).")
    for key in ("momentum", "beta1", "beta2"):
        if not 0 <= settings[key] < 1:
            raise ValueError(f"{key} must be in [0, 1).")
    if not 0 < settings["armijo_c1"] < settings["wolfe_c2"] < 1:
        raise ValueError("Require 0 < armijo_c1 < wolfe_c2 < 1.")
    if not 0 < settings["backtracking_factor"] < 1:
        raise ValueError("backtracking_factor must be in (0, 1).")
    return settings


def optimize(value_grad, fun, x0, method, config=None):
    """Minimize an unconstrained smooth objective using one full-batch solver.

    Methods are ``gd``, ``momentum`` (heavy ball), ``adam``, and ``lbfgs``.
    Configuration keys and defaults are listed in ``_settings``. ``learning_rate``
    is the fixed step for Momentum/Adam and initial backtracking step for GD.
    For L-BFGS it only sets the Armijo fallback's initial step; SciPy's normal
    strong-Wolfe search starts with its own unit-step policy.

    ``history`` contains one record per accepted parameter vector, including the
    initial vector. Loss and gradient in each record refer to that exact vector.
    ``snapshots`` saves vectors every snapshot_every iterations plus initial and
    final accepted vectors. A budget exhausted at an unaccepted trial point
    returns the previous fully evaluated accepted vector. Evaluation counts in
    the top-level result include such unsuccessful trial computations.

    Status is one of converged, time_limit, evaluation_limit, iteration_limit,
    line_search_failed, or numerical_failure. A single objective evaluation can
    overrun the time budget; the solver does not start a new one after expiry.
    """
    method = str(method).lower().replace("-", "").replace("_", "")
    aliases = {"gradientdescent": "gd", "heavyball": "momentum"}
    method = aliases.get(method, method)
    if method not in ("gd", "momentum", "adam", "lbfgs"):
        raise ValueError(f"Unknown method: {method}")
    config = _settings(method, config)
    if method == "lbfgs":
        # Import/setup is outside the optimization timer, like model/data setup.
        from scipy.optimize import line_search

    start = perf_counter()
    x = np.asarray(x0, dtype=np.float64).copy()
    if x.ndim != 1 or not x.size:
        raise ValueError("x0 must be a nonempty one-dimensional vector.")
    oracle = _Oracle(value_grad, fun, start, config["time_budget_s"],
                     config["max_evaluations"])
    history = []
    snapshots = []
    diagnostics = {
        "wolfe_failures": 0,
        "armijo_fallbacks": 0,
        "curvature_pairs_skipped": 0,
        "direction_resets": 0,
        "line_search_warnings": [],
        "n_loss_only": 0,
        "n_value_grad": 0,
        "error": None,
    }
    iteration = 0
    status = "iteration_limit"
    loss = float("nan")
    grad = np.full_like(x, np.nan)
    gradient_scale = 1.0
    velocity = np.zeros_like(x)
    first_moment = np.zeros_like(x)
    second_moment = np.zeros_like(x)
    pairs = []

    def record(step_size, force_snapshot=False):
        elapsed = perf_counter() - start
        grad_inf = float(np.linalg.norm(grad, ord=np.inf))
        history.append({
            "iteration": iteration, "elapsed_s": elapsed, "loss": float(loss),
            "grad_norm": float(np.linalg.norm(grad)), "grad_inf": grad_inf,
            "normalized_grad_inf": grad_inf / gradient_scale,
            "n_fun": oracle.n_fun, "n_grad": oracle.n_grad,
            "step_size": float(step_size),
        })
        if force_snapshot or iteration % config["snapshot_every"] == 0:
            snapshots.append({"iteration": iteration, "elapsed_s": elapsed,
                              "x": x.copy()})

    try:
        loss, grad = oracle.value_grad(x)
        gradient_scale = max(1.0, float(np.linalg.norm(grad, ord=np.inf)))
        record(0.0, force_snapshot=True)
        for iteration_candidate in range(1, config["max_iterations"] + 1):
            if float(np.linalg.norm(grad, ord=np.inf)) / gradient_scale <= config["grad_tol"]:
                status = "converged"
                break
            oracle.check_budget()
            if method == "gd":
                direction = -grad
                step = _armijo(oracle, x, loss, grad, direction,
                               config["learning_rate"], config)
                if step is None:
                    status = "line_search_failed"
                    break
                new_x = x + step * direction
            elif method == "momentum":
                velocity = config["momentum"] * velocity + grad
                step = config["learning_rate"]
                new_x = x - step * velocity
            elif method == "adam":
                b1, b2 = config["beta1"], config["beta2"]
                first_moment = b1 * first_moment + (1 - b1) * grad
                second_moment = b2 * second_moment + (1 - b2) * grad * grad
                corrected_m = first_moment / (1 - b1 ** iteration_candidate)
                corrected_v = second_moment / (1 - b2 ** iteration_candidate)
                step = config["learning_rate"]
                new_x = x - step * corrected_m / (np.sqrt(corrected_v) + config["epsilon"])
            else:
                direction = _lbfgs_direction(grad, pairs)
                slope = float(np.dot(grad, direction))
                if not math.isfinite(slope) or slope >= 0:
                    direction = -grad
                    pairs.clear()
                    diagnostics["direction_resets"] += 1
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    search_result = line_search(
                        oracle.fun, oracle.grad, x, direction, gfk=grad,
                        old_fval=loss, c1=config["armijo_c1"],
                        c2=config["wolfe_c2"],
                        maxiter=config["max_line_search_steps"],
                    )
                if caught:
                    diagnostics["line_search_warnings"].extend(
                        str(item.message) for item in caught
                    )
                step = search_result[0]
                # A returned alpha alone is not evidence of Wolfe success:
                # SciPy returns a missing new slope when the search fails.
                if (step is None or search_result[5] is None
                        or not math.isfinite(step) or step <= 0):
                    diagnostics["wolfe_failures"] += 1
                    diagnostics["armijo_fallbacks"] += 1
                    step = _armijo(oracle, x, loss, grad, direction,
                                   config["learning_rate"], config)
                if step is None:
                    status = "line_search_failed"
                    break
                new_x = x + step * direction

            new_loss, new_grad = oracle.value_grad(new_x)
            if method == "lbfgs":
                s = new_x - x
                y = new_grad - grad
                sy = float(np.dot(s, y))
                scale = float(np.linalg.norm(s) * np.linalg.norm(y))
                if math.isfinite(sy) and sy > config["curvature_threshold"] * scale and sy > 0:
                    pairs.append((s.copy(), y.copy(), 1.0 / sy))
                    if len(pairs) > config["history_size"]:
                        pairs.pop(0)
                else:
                    diagnostics["curvature_pairs_skipped"] += 1
            x, loss, grad = new_x, new_loss, new_grad
            iteration = iteration_candidate
            record(step)
        else:
            if float(np.linalg.norm(grad, ord=np.inf)) / gradient_scale <= config["grad_tol"]:
                status = "converged"
    except _BudgetExceeded as exc:
        status = exc.status
    except (_NumericalFailure, FloatingPointError, OverflowError) as exc:
        status = "numerical_failure"
        diagnostics["error"] = str(exc)

    # Snapshot reflects the accepted vector, even if a rejected trial consumed
    # the final part of the budget. Its timestamp is the accepted-point time.
    if history and (not snapshots or snapshots[-1]["iteration"] != iteration):
        snapshots.append({"iteration": iteration,
                          "elapsed_s": history[-1]["elapsed_s"], "x": x.copy()})
    elif not snapshots:
        snapshots.append({"iteration": 0, "elapsed_s": perf_counter() - start,
                          "x": x.copy()})
    diagnostics["n_loss_only"] = oracle.n_fun - oracle.n_grad
    diagnostics["n_value_grad"] = oracle.n_grad
    return {
        "method": method, "x": x, "loss": float(loss), "gradient": grad,
        "history": history, "snapshots": snapshots, "status": status,
        "n_fun": oracle.n_fun, "n_grad": oracle.n_grad,
        "elapsed_s": perf_counter() - start, "iterations": iteration,
        "gradient_scale": gradient_scale, "diagnostics": diagnostics,
        "config": dict(config),
    }
