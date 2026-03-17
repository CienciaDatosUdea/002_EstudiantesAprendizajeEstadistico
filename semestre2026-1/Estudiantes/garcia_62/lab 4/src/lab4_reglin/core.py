"""Funciones para ajustar una regresion lineal simple con gradiente descendente."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


HistoryType = Dict[str, List[float]]


def _prepare_xy(x: Iterable[float], y: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Convierte x e y a arreglos 1D y valida su consistencia."""
    x_arr = np.asarray(list(x), dtype=float).reshape(-1)
    y_arr = np.asarray(list(y), dtype=float).reshape(-1)

    if x_arr.size == 0 or y_arr.size == 0:
        raise ValueError("x e y no pueden estar vacios")
    if x_arr.shape != y_arr.shape:
        raise ValueError("x e y deben tener la misma longitud")

    return x_arr, y_arr


def linear_hypothesis(x: Iterable[float], theta0: float, theta1: float) -> np.ndarray:
    """Calcula la hipotesis lineal h(x) = theta0 + theta1*x."""
    x_arr = np.asarray(list(x), dtype=float).reshape(-1)
    return theta0 + theta1 * x_arr


def compute_cost(x: Iterable[float], y: Iterable[float], theta0: float, theta1: float) -> float:
    """Calcula J(theta0, theta1) = (1/(2m))*sum((h(x)-y)^2)."""
    x_arr, y_arr = _prepare_xy(x, y)
    m = x_arr.size
    errors = linear_hypothesis(x_arr, theta0, theta1) - y_arr
    return float(np.dot(errors, errors) / (2 * m))


def gradient_descent(
    x: Iterable[float],
    y: Iterable[float],
    theta0_init: float = 0.0,
    theta1_init: float = 0.0,
    alpha: float = 0.1,
    n_iter: int = 1000,
    tol: float = 1e-8,
    return_history: bool = True,
) -> Dict[str, object]:
    """Ejecuta gradiente descendente para una regresion lineal simple."""
    if alpha <= 0:
        raise ValueError("alpha debe ser mayor que cero")
    if n_iter <= 0:
        raise ValueError("n_iter debe ser mayor que cero")
    if tol < 0:
        raise ValueError("tol no puede ser negativo")

    x_arr, y_arr = _prepare_xy(x, y)
    m = x_arr.size

    theta0 = float(theta0_init)
    theta1 = float(theta1_init)

    history: Optional[HistoryType]
    if return_history:
        history = {"theta0": [], "theta1": [], "cost": []}
    else:
        history = None

    iterations = 0
    for i in range(1, n_iter + 1):
        y_hat = theta0 + theta1 * x_arr
        err = y_hat - y_arr

        grad0 = float(np.sum(err) / m)
        grad1 = float(np.sum(err * x_arr) / m)

        theta0_new = theta0 - alpha * grad0
        theta1_new = theta1 - alpha * grad1

        if history is not None:
            history["theta0"].append(theta0_new)
            history["theta1"].append(theta1_new)
            history["cost"].append(compute_cost(x_arr, y_arr, theta0_new, theta1_new))

        if max(abs(theta0_new - theta0), abs(theta1_new - theta1)) < tol:
            theta0, theta1 = theta0_new, theta1_new
            iterations = i
            break

        theta0, theta1 = theta0_new, theta1_new
        iterations = i

    return {
        "theta0": theta0,
        "theta1": theta1,
        "cost": compute_cost(x_arr, y_arr, theta0, theta1),
        "iterations": iterations,
        "history": history,
    }


def fit_linear_regression(
    x: Iterable[float],
    y: Iterable[float],
    alpha: float = 0.1,
    n_iter: int = 1000,
    tol: float = 1e-8,
    theta0_init: float = 0.0,
    theta1_init: float = 0.0,
    return_history: bool = True,
) -> Dict[str, object]:
    """Funcion principal para ajustar el modelo lineal con gradiente descendente."""
    return gradient_descent(
        x=x,
        y=y,
        theta0_init=theta0_init,
        theta1_init=theta1_init,
        alpha=alpha,
        n_iter=n_iter,
        tol=tol,
        return_history=return_history,
    )
