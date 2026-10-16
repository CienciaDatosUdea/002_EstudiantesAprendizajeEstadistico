"""Algoritmos basicos para regresion lineal con una variable."""

import numpy as np


def hipotesis_lineal(X, theta):
    """Calcula h_theta(X) = theta_0 + theta_1 X."""
    X = np.asarray(X, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (2,):
        raise ValueError("theta debe contener [intercepto, pendiente]")
    return theta[0] + theta[1] * X


def coste(X, y, theta):
    """Calcula J(theta) = 1/(2m) * sum((h_theta(X) - y)^2)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.shape != y.shape:
        raise ValueError("X e y deben tener la misma longitud")
    if X.ndim != 1:
        raise ValueError("X e y deben ser vectores unidimensionales")
    error = hipotesis_lineal(X, theta) - y
    return float(np.sum(error ** 2) / (2 * X.size))


def descenso_gradiente(X, y, theta_inicial=None, tasa_aprendizaje=0.01,
                       tolerancia=1e-8, max_iter=10000):
    """Optimiza theta usando el gradiente analitico de la funcion de coste."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.shape != y.shape or X.ndim != 1 or X.size == 0:
        raise ValueError("X e y deben ser vectores no vacios de igual longitud")
    if tasa_aprendizaje <= 0 or tolerancia <= 0 or max_iter <= 0:
        raise ValueError("Los hiperparametros deben ser positivos")

    theta = np.zeros(2, dtype=float) if theta_inicial is None else np.asarray(theta_inicial, dtype=float).copy()
    if theta.shape != (2,):
        raise ValueError("theta_inicial debe contener dos valores")

    historia = []
    for _ in range(max_iter):
        error = hipotesis_lineal(X, theta) - y
        gradiente = np.array([
            np.mean(error),
            np.mean(error * X),
        ])
        nuevo_theta = theta - tasa_aprendizaje * gradiente
        historia.append(coste(X, y, theta))
        if np.linalg.norm(nuevo_theta - theta) <= tolerancia:
            return nuevo_theta, np.asarray(historia)
        theta = nuevo_theta

    raise RuntimeError("El descenso no convergio dentro de max_iter")


def ajustar_modelo(X, y, **kwargs):
    """Ajusta una regresion lineal y devuelve theta e historial de coste."""
    theta, historia = descenso_gradiente(X, y, **kwargs)
    return {"intercepto": float(theta[0]), "pendiente": float(theta[1]),
            "theta": theta, "historial_coste": historia}
