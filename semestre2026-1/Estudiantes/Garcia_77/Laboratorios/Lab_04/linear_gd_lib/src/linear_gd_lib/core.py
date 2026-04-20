from __future__ import annotations

from typing import Callable

import numpy as np


ArrayLike = np.ndarray


def linear_hypothesis(X: ArrayLike, theta0: float, theta1: float) -> ArrayLike:
    """Calcula la hipotesis lineal h(x)=theta0+theta1*x."""
    X_arr = np.asarray(X, dtype=float)
    return theta0 + theta1 * X_arr


def linear_cost(X: ArrayLike, y: ArrayLike, theta0: float, theta1: float) -> float:
    """Calcula el costo cuadratico medio escalado J(theta0, theta1)."""
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if X_arr.shape != y_arr.shape:
        raise ValueError("X e y deben tener la misma forma.")

    m = y_arr.size
    pred = linear_hypothesis(X_arr, theta0, theta1)
    return float((1.0 / (2.0 * m)) * np.sum((pred - y_arr) ** 2))


def run_gradient_descent(
    X: ArrayLike,
    y: ArrayLike,
    alpha: float = 0.1,
    n_iter: int = 10_000,
    tol: float = 1e-12,
) -> tuple[float, float, np.ndarray]:
    """Ejecuta gradiente descendente para ajustar theta0 y theta1."""
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if X_arr.shape != y_arr.shape:
        raise ValueError("X e y deben tener la misma forma.")
    if alpha <= 0:
        raise ValueError("alpha debe ser positivo.")
    if n_iter <= 0:
        raise ValueError("n_iter debe ser mayor que cero.")
    if tol <= 0:
        raise ValueError("tol debe ser positivo.")

    m = y_arr.size
    theta0, theta1 = 0.0, 0.0
    history: list[float] = []

    for _ in range(n_iter):
        pred = linear_hypothesis(X_arr, theta0, theta1)
        err = pred - y_arr

        grad0 = float(np.sum(err) / m)
        grad1 = float(np.sum(err * X_arr) / m)

        new_theta0 = theta0 - alpha * grad0
        new_theta1 = theta1 - alpha * grad1

        history.append(linear_cost(X_arr, y_arr, new_theta0, new_theta1))

        if np.linalg.norm([new_theta0 - theta0, new_theta1 - theta1]) < tol:
            theta0, theta1 = new_theta0, new_theta1
            break

        theta0, theta1 = new_theta0, new_theta1

    return theta0, theta1, np.asarray(history, dtype=float)


def fit_linear_gd(
    X: ArrayLike,
    y: ArrayLike,
    alpha: float = 0.1,
    n_iter: int = 10_000,
    tol: float = 1e-12,
) -> dict[str, float | np.ndarray | Callable[[ArrayLike], ArrayLike]]:
    """Ajusta el modelo lineal con gradiente descendente y retorna un diccionario de resultados."""
    theta0, theta1, history = run_gradient_descent(
        X=X,
        y=y,
        alpha=alpha,
        n_iter=n_iter,
        tol=tol,
    )

    return {
        "theta0": theta0,
        "theta1": theta1,
        "history": history,
        "predict": lambda x_new: linear_hypothesis(np.asarray(x_new, dtype=float), theta0, theta1),
    }
