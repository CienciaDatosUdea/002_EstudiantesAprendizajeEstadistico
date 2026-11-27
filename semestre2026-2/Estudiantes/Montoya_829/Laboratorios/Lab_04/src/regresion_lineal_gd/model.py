"""Implementacion de regresion lineal simple por descenso por gradiente."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearRegressionResult:
    """Parametros y diagnosticos producidos al ajustar una recta."""

    intercept: float
    slope: float
    cost: float
    iterations: int
    history: tuple[tuple[float, float, float], ...]

    def predict(self, x: np.ndarray | list[float]) -> np.ndarray:
        """Predice valores de y para los valores unidimensionales de x."""
        return linear_hypothesis(x, self.intercept, self.slope)


def _validate_xy(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Convierte x e y a vectores numericos compatibles para una regresion simple."""
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)

    if x_array.ndim != 1 or y_array.ndim != 1:
        raise ValueError("x e y deben ser vectores unidimensionales.")
    if x_array.size == 0:
        raise ValueError("x e y deben contener al menos una observacion.")
    if x_array.shape != y_array.shape:
        raise ValueError("x e y deben tener la misma cantidad de observaciones.")
    if not (np.isfinite(x_array).all() and np.isfinite(y_array).all()):
        raise ValueError("x e y deben contener valores finitos.")

    return x_array, y_array


def linear_hypothesis(x: np.ndarray | list[float], intercept: float, slope: float) -> np.ndarray:
    """Calcula h(x) = intercept + slope * x para una caracteristica."""
    return float(intercept) + float(slope) * np.asarray(x, dtype=float)


def cost_function(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    intercept: float,
    slope: float,
) -> float:
    """Calcula J = 1/(2m) * sum((h(x) - y)^2)."""
    x_array, y_array = _validate_xy(x, y)
    errors = linear_hypothesis(x_array, intercept, slope) - y_array
    return float(np.dot(errors, errors) / (2 * x_array.size))


def gradient_descent(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    learning_rate: float = 0.1,
    tolerance: float = 1e-8,
    max_iter: int = 10_000,
    intercept_initial: float = 0.0,
    slope_initial: float = 0.0,
) -> LinearRegressionResult:
    """Ajusta intercepto y pendiente mediante actualizaciones simultaneas de gradiente."""
    x_array, y_array = _validate_xy(x, y)
    if learning_rate <= 0:
        raise ValueError("learning_rate debe ser mayor que cero.")
    if tolerance <= 0:
        raise ValueError("tolerance debe ser mayor que cero.")
    if max_iter < 1:
        raise ValueError("max_iter debe ser al menos uno.")

    intercept = float(intercept_initial)
    slope = float(slope_initial)
    history = [(intercept, slope, cost_function(x_array, y_array, intercept, slope))]

    for iteration in range(1, max_iter + 1):
        errors = linear_hypothesis(x_array, intercept, slope) - y_array
        gradient_intercept = float(np.mean(errors))
        gradient_slope = float(np.mean(errors * x_array))
        next_intercept = intercept - learning_rate * gradient_intercept
        next_slope = slope - learning_rate * gradient_slope
        next_cost = cost_function(x_array, y_array, next_intercept, next_slope)
        history.append((next_intercept, next_slope, next_cost))

        if np.linalg.norm([next_intercept - intercept, next_slope - slope]) < tolerance:
            return LinearRegressionResult(
                intercept=next_intercept,
                slope=next_slope,
                cost=next_cost,
                iterations=iteration,
                history=tuple(history),
            )

        intercept, slope = next_intercept, next_slope

    raise RuntimeError("El descenso por gradiente no convergio; ajuste learning_rate o max_iter.")


def fit_linear_regression(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    **kwargs: float | int,
) -> LinearRegressionResult:
    """Funcion principal para ajustar una regresion lineal simple por gradiente."""
    return gradient_descent(x, y, **kwargs)
