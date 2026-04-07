"""Small linear regression library based on cost minimization with gradient descent."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_1d_float_array(values: Iterable[float], name: str) -> np.ndarray:
    """Convert an input iterable into a non-empty one-dimensional float array."""
    array = np.asarray(values, dtype=float)

    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence of numbers.")

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    return array


def hypothesis(theta0: float, theta1: float, x):
    """Compute the linear hypothesis for one or many input values.

    Parameters
    ----------
    theta0 : float
        Intercept of the linear model.
    theta1 : float
        Slope of the linear model.
    x : float or array-like
        Input value or sequence of input values.

    Returns
    -------
    float or numpy.ndarray
        Predicted value(s) computed as ``theta0 + theta1 * x``.
    """
    x_array = np.asarray(x, dtype=float)
    predictions = theta0 + theta1 * x_array
    return float(predictions) if np.ndim(predictions) == 0 else predictions


def compute_cost(theta0: float, theta1: float, x: Iterable[float], y: Iterable[float]) -> float:
    """Compute the quadratic cost for a simple linear regression model.

    Parameters
    ----------
    theta0 : float
        Intercept of the linear model.
    theta1 : float
        Slope of the linear model.
    x : array-like
        Input feature values.
    y : array-like
        Observed target values.

    Returns
    -------
    float
        The value of the cost function ``sum((h(x) - y)^2) / (2m)``.
    """
    x_array = _as_1d_float_array(x, "x")
    y_array = _as_1d_float_array(y, "y")

    if x_array.size != y_array.size:
        raise ValueError("x and y must contain the same number of elements.")

    residuals = hypothesis(theta0, theta1, x_array) - y_array
    return float(np.sum(residuals**2) / (2 * x_array.size))


def gradient_descent(
    x: Iterable[float],
    y: Iterable[float],
    theta0: float = 0.0,
    theta1: float = 0.0,
    learning_rate: float = 0.01,
    iterations: int = 1000,
) -> tuple[float, float, list[float]]:
    """Optimize a linear model with gradient descent.

    Parameters
    ----------
    x : array-like
        Input feature values.
    y : array-like
        Observed target values.
    theta0 : float, default=0.0
        Initial value for the intercept.
    theta1 : float, default=0.0
        Initial value for the slope.
    learning_rate : float, default=0.01
        Step size used in each gradient update.
    iterations : int, default=1000
        Number of optimization steps to run.

    Returns
    -------
    tuple[float, float, list[float]]
        Final ``theta0``, final ``theta1``, and the cost history after each step.
    """
    x_array = _as_1d_float_array(x, "x")
    y_array = _as_1d_float_array(y, "y")

    if x_array.size != y_array.size:
        raise ValueError("x and y must contain the same number of elements.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0.")
    if iterations <= 0:
        raise ValueError("iterations must be greater than 0.")

    m = x_array.size
    current_theta0 = float(theta0)
    current_theta1 = float(theta1)
    cost_history: list[float] = []

    for _ in range(iterations):
        residuals = hypothesis(current_theta0, current_theta1, x_array) - y_array
        current_theta0 -= learning_rate * float(np.sum(residuals) / m)
        current_theta1 -= learning_rate * float(np.sum(residuals * x_array) / m)
        cost_history.append(compute_cost(current_theta0, current_theta1, x_array, y_array))

    return current_theta0, current_theta1, cost_history


def fit_linear_regression(
    x: Iterable[float],
    y: Iterable[float],
    theta0: float = 0.0,
    theta1: float = 0.0,
    learning_rate: float = 0.01,
    iterations: int = 1000,
) -> dict[str, float | list[float]]:
    """Fit a simple linear regression model to a dataset.

    This convenience function validates the data, runs gradient descent, and
    returns the learned parameters together with the final cost.

    Parameters
    ----------
    x : array-like
        Input feature values.
    y : array-like
        Observed target values.
    theta0 : float, default=0.0
        Initial value for the intercept.
    theta1 : float, default=0.0
        Initial value for the slope.
    learning_rate : float, default=0.01
        Step size used in each gradient update.
    iterations : int, default=1000
        Number of optimization steps to run.

    Returns
    -------
    dict[str, float | list[float]]
        Dictionary containing ``theta0``, ``theta1``, ``cost``, and ``cost_history``.
    """
    final_theta0, final_theta1, cost_history = gradient_descent(
        x=x,
        y=y,
        theta0=theta0,
        theta1=theta1,
        learning_rate=learning_rate,
        iterations=iterations,
    )

    return {
        "theta0": final_theta0,
        "theta1": final_theta1,
        "cost": cost_history[-1],
        "cost_history": cost_history,
    }


__all__ = [
    "hypothesis",
    "compute_cost",
    "gradient_descent",
    "fit_linear_regression",
]
