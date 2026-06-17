"""Basic checks for the linear regression library."""

from __future__ import annotations

import numpy as np

from linear_reg_lib import compute_cost, fit_linear_regression, gradient_descent, hypothesis


def test_hypothesis_scalar_and_vector() -> None:
    assert hypothesis(1.0, 2.0, 3.0) == 7.0
    assert np.allclose(hypothesis(1.0, 2.0, [0.0, 1.0, 2.0]), np.array([1.0, 3.0, 5.0]))


def test_compute_cost_perfect_fit() -> None:
    x = np.arange(6, dtype=float)
    y = np.arange(6, dtype=float)
    assert compute_cost(0.0, 1.0, x, y) == 0.0


def test_gradient_descent_reduces_cost() -> None:
    x = np.arange(6, dtype=float)
    y = np.arange(6, dtype=float)
    initial_cost = compute_cost(0.0, 0.0, x, y)
    _, _, cost_history = gradient_descent(x, y, learning_rate=0.01, iterations=2000)
    assert cost_history[-1] < initial_cost


def test_fit_linear_regression_recovers_expected_parameters() -> None:
    x = np.arange(6, dtype=float)
    y = np.arange(6, dtype=float)
    model = fit_linear_regression(x, y, learning_rate=0.01, iterations=2000)
    assert np.isclose(model["theta0"], 0.0, atol=1e-2)
    assert np.isclose(model["theta1"], 1.0, atol=1e-2)
    assert model["cost"] < 1e-4


def test_validation_errors() -> None:
    try:
        fit_linear_regression([], [])
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty inputs.")

    try:
        fit_linear_regression([1.0, 2.0], [1.0])
    except ValueError as exc:
        assert "same number of elements" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched input sizes.")
